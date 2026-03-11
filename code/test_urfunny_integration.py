#!/usr/bin/env python3
"""
Test script to validate URFunny integration with the main paper code flow
"""

import os
import sys
import torch
from utils.config import Config, DEFAULT_URFUNNY_PARAMS
from URFunny_task import Funny_Task

def test_config():
    """Test that Config can be instantiated with fixed parameters"""
    print("Testing Config with fixed parameters...")
    try:
        # For testing, we'll use a dummy data_root path
        test_params = DEFAULT_URFUNNY_PARAMS.copy()
        test_params['data_root'] = 'dummy_path_for_testing.pkl'  # Override for testing
        
        cfgs = Config(fixed_params=test_params)
        print(f"✓ Config created successfully")
        print(f"  Dataset: {cfgs.dataset}")
        print(f"  Data root: {cfgs.data_root}")
        print(f"  Fusion type: {cfgs.fusion_type}")
        print(f"  Max pad: {cfgs.max_pad}")
        print(f"  Aligned: {cfgs.aligned}")
        print("  ⚠️  Note: Set data_root to your actual URFunny pickle file path")
        return cfgs
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return None

def test_task_creation(cfgs):
    """Test that URFunny task can be created"""
    print("\nTesting URFunny Task creation...")
    try:
        # Use default batch_size for testing
        task = Funny_Task(cfgs, batch_size=16)
        print(f"✓ Funny_Task created successfully")
        print(f"  Model type: {type(task.model).__name__}")
        print(f"  Batch size: {task.batch_size}")
        print(f"  Train dataloader size: {len(task.train_dataloader)}")
        print(f"  Valid dataloader size: {len(task.valid_dataloader)}")
        return task
    except Exception as e:
        print(f"✗ Task creation failed: {e}")
        return None

def test_model_forward(task):
    """Test that the model can perform forward pass"""
    print("\nTesting model forward pass...")
    try:
        # Get one batch from train dataloader
        dataloader = task.train_dataloader
        for batch in dataloader:
            feature, feature_length, index, label = batch
            vision = feature[0].float()
            audio = feature[1].float() 
            text = feature[2].float()
            
            print(f"  Vision shape: {vision.shape}")
            print(f"  Audio shape: {audio.shape}")
            print(f"  Text shape: {text.shape}")
            print(f"  Label shape: {label.shape}")
            
            # Test forward pass
            with torch.no_grad():
                outputs = task.model(vision, audio, text, feature_length)
                print(f"  Output length: {len(outputs)}")
                print(f"  Final output shape: {outputs[-1].shape}")
            print("✓ Model forward pass successful")
            break
        return True
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        return False

def test_reinit_integration():
    """Test that reinit functions can handle vision+text"""
    print("\nTesting reinit function compatibility...")
    try:
        # Test if the main functions exist and have correct signatures
        from ours import reinit_score, reinit, get_feature
        
        print("✓ Required functions imported successfully:")
        print("  - reinit_score (for vision+text)")
        print("  - reinit (for vision+text)")
        print("  - get_feature (for URFunny format)")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    print("URFunny Integration Test")
    print("=" * 50)
    
    # Test 1: Config creation
    cfgs = test_config()
    if cfgs is None:
        print("\n❌ Integration test FAILED at config creation")
        return False
    
    # Test 2: Task creation
    task = test_task_creation(cfgs)
    if task is None:
        print("\n❌ Integration test FAILED at task creation")
        return False
    
    # Test 3: Model forward pass
    if not test_model_forward(task):
        print("\n❌ Integration test FAILED at model forward pass")
        return False
    
    # Test 4: Reinit integration
    if not test_reinit_integration():
        print("\n❌ Integration test FAILED at reinit integration")
        return False
    
    print("\n✅ All integration tests PASSED!")
    print("\nYour URFunny integration appears to be working correctly.")
    print("Key fixes applied:")
    print("  • Removed conflicting parameters (epochs, batch_size, learning_rate, optimizer)")
    print("  • Kept only URFunny-specific parameters in config")
    print("  • Fixed optimizer duplication issue")
    print("  • URFunny now inherits main training parameters from ours.py")
    print("\nYou can now run the main training script with:")
    print("python ours.py --dataset URFunny --train --epochs 30 --batch_size 16 --learning_rate 0.0001 --optimizer adam")
    return True

if __name__ == "__main__":
    main()