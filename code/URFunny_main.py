import time
import torch
import os
import math
import torch.nn as nn
import numpy as np
from utils.metric import Accuracy
from tasks.URFunny_task import Funny_Task
from utils.function_tools import save_config,get_logger,get_device,set_seed
from torch.utils.tensorboard import SummaryWriter
from sklearn import linear_model


def train(model, train_dataloader, optimizer, scheduler, cfgs, device, logger, epoch, writer, 
          last_train_score_v, last_train_score_t, visual_lr_ratio, text_lr_ratio):
    softmax = nn.Softmax(dim=1)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_batch = len(train_dataloader)
    
    train_acc, train_visual_acc, train_text_acc = 0., 0., 0.
    train_score_v, train_score_t = last_train_score_v, last_train_score_t

    for step, (feature, feature_length, index, label) in enumerate(train_dataloader):
        vision = feature[0].float().to(device)
        audio = feature[1].float().to(device) # Will be zeroed inside model
        text = feature[2].float().to(device)
        label = label.squeeze(1).to(device)
        # label = (label + 1) // 2  # Converts -1 to 0, and leaves 1 as 1 for MUStARD
        
        iteration = (epoch - 1) * len(train_dataloader) + step + 1
        optimizer.zero_grad()
        
        # Forward pass (GradMod_2Modality returns None for audio slots)
        _, _, _, _, m_v_out, m_t_out, out = model(vision, audio, text, feature_length)
        loss = loss_fn(out, label)
        
        # Calculate Log-Loss for AGM
        score_v = -torch.log(softmax(m_v_out)[range(out.size(0)), label]).mean()
        score_t = -torch.log(softmax(m_t_out)[range(out.size(0)), label]).mean()
        
        # 2-Modality Ratio (Reference: CREMA-D)
        ratio_v = math.exp(score_t.item() - score_v.item())
        ratio_t = math.exp(score_v.item() - score_t.item())
        
        optimal_ratio_v = math.exp(train_score_t - train_score_v)
        optimal_ratio_t = math.exp(train_score_v - train_score_t)
        
        # AGM Coefficients
        coeff_v = math.exp(cfgs.alpha * (min(optimal_ratio_v - ratio_v, 10)))
        coeff_t = math.exp(cfgs.alpha * (min(optimal_ratio_t - ratio_t, 10)))
        
        # Update training scores
        train_score_v = train_score_v * (iteration - 1) / iteration + score_v.item() / iteration
        train_score_t = train_score_t * (iteration - 1) / iteration + score_t.item() / iteration

        if cfgs.methods == "AGM" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
            model.update_scale(coeff_v, coeff_t) # Updated scale method
            loss.backward()
        else:
            loss.backward()

        optimizer.step()
        
        # ... (Metrics and logging) ...
        if step % 100 == 0:
            logger.info('EPOCH:[{:3d}/{:3d}]--STEP:[{:5d}/{:5d}]--{}--Loss:{:.4f}--lr:{}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Train',loss.item(),[group['lr'] for group in optimizer.param_groups]))
    
    scheduler.step()
    return train_score_v, train_score_t
    
def validate(model, validate_dataloader, cfgs, device, logger, epoch, writer):
    softmax = nn.Softmax(dim=1)
    model.eval()
    model.mode = 'eval'
    
    validate_acc, validate_v_acc, validate_t_acc = 0., 0., 0.
    total_batch = len(validate_dataloader)
    
    with torch.no_grad():
        for step, (feature, feature_length, index, label) in enumerate(validate_dataloader):
            vision, audio, text = feature[0].float().to(device), feature[1].float().to(device), feature[2].float().to(device)
            label = label.squeeze(1).to(device)
            # label = (label + 1) // 2  # Converts -1 to 0, and leaves 1 as 1 for MUStARD
            
            _, _, _, _, m_v_out, m_t_out, out = model(vision, audio, text, feature_length)
            
            validate_acc += Accuracy(softmax(out), label).item() / total_batch
            validate_v_acc += Accuracy(softmax(m_v_out), label).item() / total_batch
            validate_t_acc += Accuracy(softmax(m_t_out), label).item() / total_batch

    logger.info(f'EPOCH[{epoch}] VALID: Acc: {validate_acc:.4f} | V-Acc: {validate_v_acc:.4f} | T-Acc: {validate_t_acc:.4f}')
    return validate_acc
    

def validate_compute_weight(model,validate_dataloader,cfgs,device,logger,epoch,writer,mm_to_audio_lr,mm_to_visual_lr,mm_to_text_lr,test_audio_out,test_visual_out,test_text_out):
    softmax = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    model.mode = 'eval'
    with torch.no_grad():
        if cfgs.use_mgpu:
            model.module.net.eval()
        else:
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                model.net.eval()
            else:
                model.eval()
        ota = []
        otv = []
        ott = []
        valid_score_a = 0.
        valid_score_v = 0.
        valid_score_t = 0. 

        valid_batch_audio_loss = 0.
        valid_batch_visual_loss = 0.
        valid_batch_text_loss = 0.   
        
        validate_acc = 0.
        validate_visual_acc = 0.
        validate_audio_acc = 0.
        validate_text_acc = 0.
        total_batch = len(validate_dataloader)
        start_time = time.time()
        model.extract_mm_feature = True
        for step,(feature,feature_length,index,label) in enumerate(validate_dataloader):
            vision = feature[0].float().to(device)
            audio = feature[1].float().to(device)
            text = feature[2].float().to(device)
            label = label.squeeze(1)
            label = label.to(device)
            label = (label + 1) // 2  # Converts -1 to 0, and leaves 1 as 1 for MUStARD
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                m_a_mc,m_v_mc,m_t_mc,m_a_out,m_v_out,m_t_out,out,encoded_feature = model(vision,audio,text,feature_length)
            else:
                m_a_out,m_v_out,m_t_out,out,encoded_feature = model(vision,audio,text,feature_length)
            out_to_audio = mm_to_audio_lr.predict(encoded_feature.detach().cpu())
            out_to_visual = mm_to_visual_lr.predict(encoded_feature.detach().cpu())
            out_to_text = mm_to_text_lr.predict(encoded_feature.detach().cpu())
            ota.append(torch.from_numpy(out_to_audio))
            otv.append(torch.from_numpy(out_to_visual))
            ott.append(torch.from_numpy(out_to_text))
            
            score_visual = 0.
            score_audio = 0.
            score_text = 0.
            for k in range(out.size(0)):
                score_visual += - torch.log(softmax(m_v_out)[k][label[k]]) / m_v_out.size(0)
                score_audio += - torch.log(softmax(m_a_out)[k][label[k]]) / m_a_out.size(0)
                score_text += - torch.log(softmax(m_t_out)[k][label[k]]) / m_t_out.size(0)
            
            valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)
            valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            valid_score_t = valid_score_t * step / (step + 1) + score_text.item() / (step + 1)
            
            mean_score = (score_visual.item() + score_audio.item() + score_text.item()) / 3
            
            ratio_v = math.exp(mean_score - score_visual.item())
            ratio_a = math.exp(mean_score - score_audio.item())
            ratio_t = math.exp(mean_score - score_text.item())
            
            optimal_mean_score = (valid_score_v + valid_score_a + valid_score_t) / 3
            optimal_ratio_v = math.exp(optimal_mean_score - valid_score_v)
            optimal_ratio_a = math.exp(optimal_mean_score - valid_score_a)
            optimal_ratio_t = math.exp(optimal_mean_score - valid_score_t)
            

            loss = loss_fn(out,label)
            loss_a = loss_fn(m_a_out,label)
            loss_v = loss_fn(m_v_out,label)
            loss_t = loss_fn(m_t_out,label)
            valid_batch_audio_loss += loss_a.item() / total_batch
            valid_batch_visual_loss += loss_v.item() / total_batch
            valid_batch_text_loss += loss_t.item() / total_batch

            preds = softmax(out)
            visual_preds = softmax(m_v_out)
            audio_preds = softmax(m_a_out)
            text_preds = softmax(m_t_out)
            accuracy = Accuracy(preds,label)
            visual_acc = Accuracy(visual_preds,label)
            audio_acc = Accuracy(audio_preds,label)
            text_acc = Accuracy(text_preds,label)
            validate_acc += accuracy.item()
            validate_visual_acc += visual_acc.item()
            validate_audio_acc += audio_acc.item()
            validate_text_acc += text_acc.item()

            iteration = (epoch-1)*total_batch + step
            writer.add_scalar('Validate loss/step',loss,iteration)

            if step % 20 == 0:
                logger.info('EPOCHS[{:02d}/{:02d}]--STEP[{:02d}/{:02d}]--{}--loss:{:.4f}'.format(epoch,cfgs.EPOCHS,step,total_batch,'Validate',loss))
        model.extract_mm_feature = False    
        ota = torch.cat(ota,dim=0).float()
        otv = torch.cat(otv,dim=0).float()
        ott = torch.cat(ott,dim=0).float()

        ota = ota - test_audio_out
        otv = otv - test_visual_out
        ott = ott - test_text_out
        ba = torch.cov(test_audio_out.T) * test_audio_out.size(0)
        bv = torch.cov(test_visual_out.T) * test_visual_out.size(0)
        bt = torch.cov(test_text_out.T) * test_text_out.size(0)

        ra = torch.sum(torch.multiply(ota @ torch.pinverse(ba),ota)) / test_audio_out.size(1)
        rv = torch.sum(torch.multiply(otv @ torch.pinverse(bv),otv)) / test_visual_out.size(1)
        rt = torch.sum(torch.multiply(ott @ torch.pinverse(bt),ott)) / test_text_out.size(1)
        end_time = time.time()
        elapse_time = end_time - start_time
        validate_mean_acc = validate_acc / total_batch
        validate_mean_visual_acc = validate_visual_acc / total_batch
        validate_mean_audio_acc = validate_audio_acc / total_batch
        validate_mean_text_acc = validate_text_acc / total_batch
        
        writer.add_scalars('Accuracy(Validate)',{'acc':validate_mean_acc,
                                                'audio_acc':validate_mean_audio_acc,
                                                'visual_acc':validate_mean_visual_acc,
                                                'text_acc':validate_mean_text_acc},epoch)
        logger.info('EPOCH[{:02d}/{:02d}]-{}-elapse time:{:.2f}-validate acc:{:.4f}-validate_audio_acc:{:.4f}-validate_visual_acc:{:.4f}--validate_text_acc:{:.4f}'.format(epoch,cfgs.EPOCHS,'validate',elapse_time,validate_mean_acc,validate_mean_audio_acc,validate_mean_visual_acc,validate_mean_text_acc))

        model.mode = 'train'
        return validate_mean_acc,validate_mean_visual_acc,validate_mean_audio_acc,validate_mean_text_acc,valid_batch_visual_loss,valid_batch_audio_loss,valid_batch_text_loss

def extract_mm_feature(model,dep_dataloader,device,cfgs):
    model.mode = 'eval'
    all_feature = []
    with torch.no_grad():
        model.eval()
        total_batch = len(dep_dataloader)
        for step,(feature,feature_length,index,label) in enumerate(dep_dataloader):
            vision = feature[0].float().to(device)
            audio = feature[1].float().to(device)
            text = feature[2].float().to(device)
            label = label.squeeze(1)
            label = label.to(device)
            if cfgs.methods == "AGM" or cfgs.fusion_type == "early_fusion":
                model.net.mode = 'feature'
                classify_out,out = model.net(vision,audio,text,feature_length)
                all_feature.append(out.detach().cpu())
                model.net.mode = 'classify'
            else:
                model.extract_mm_feature = True
                out_a,out_v,out_t,out,feature = model(vision,audio,text,feature_length)
                all_feature.append(feature.detach().cpu())
        all_feature = torch.cat(all_feature,dim=0)
        return all_feature

def URFunny_main(cfgs):
    set_seed(cfgs.random_seed)
    ts = time.strftime('%Y_%m_%d %H:%M:%S',time.localtime())
    save_dir = os.path.join(cfgs.expt_dir,f"{ts}_{cfgs.expt_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_config(cfgs,save_dir)
    
    if cfgs.use_mgpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.gpu_ids
        gpu_ids = list(map(int,cfgs.gpu_ids.split(",")))
        device = get_device(cfgs.device)
    else:
        device = get_device(cfgs.device)
        
    logger = get_logger("train_logger",logger_dir=save_dir)
    logger.info(vars(cfgs))
    logger.info(f"Processed ID:{os.getpid()},Device:{device},System Version:{os.uname()}")
    
    writer = SummaryWriter(os.path.join(save_dir,'tensorboard_out'))

    task = Funny_Task(cfgs)
    train_dataloader = task.train_dataloader
    validate_dataloader = task.valid_dataloader
    dep_dataloader = task.dep_dataloader

    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler

    model.to(device)
    if cfgs.use_mgpu:
        model = torch.nn.DataParallel(model,device_ids=gpu_ids)
        model.cuda()

    best_epoch = {'epoch':0,'acc':0.}

    # --- 2-Modality Trackers ---
    train_score_v = 0.
    train_score_t = 0.
    visual_lr_ratio = 1.0
    text_lr_ratio = 1.0

    for epoch in range(1,cfgs.EPOCHS+1):
        logger.info(f'Training for epoch:{epoch}...')
        
        # FIX 1: Call train() with exactly 13 args and 2 return values
        train_score_v, train_score_t = train(
            model, train_dataloader, optimizer, scheduler, cfgs, device, logger, epoch, writer,
            train_score_v, train_score_t, visual_lr_ratio, text_lr_ratio
        )
        
        logger.info(f'Validating for epoch:{epoch}...')
        
        # FIX 2: Call validate() matching our new 2-modality signature
        validate_acc = validate(model, validate_dataloader, cfgs, device, logger, epoch, writer)

        # FIX 3: Update checkpoint saving to only save Vision & Text scores
        if validate_acc > best_epoch['acc']:
            best_epoch['acc'] = validate_acc
            best_epoch['epoch'] = epoch

            if cfgs.save_checkpoint:
                torch.save({'epoch': best_epoch['epoch'],
                            'acc': best_epoch['acc'],
                            'optimizer': optimizer.state_dict(),
                            'state_dict': model.state_dict(),
                            'train_score_v': train_score_v,
                            'train_score_t': train_score_t}, os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
        
        logger.info(f'Best epoch {best_epoch["epoch"]}, best accuracy {best_epoch["acc"]:.4f}')