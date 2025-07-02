#!/usr/bin/env python3
"""
Quick start training scripts with optimized configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and stream output."""
    print(f"🚀 Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line.rstrip())
    
    process.wait()
    return process.returncode

def main():
    if len(sys.argv) < 2:
        print("""
🎯 Training Configurations:

Available modes:
  quick     - Fast training for testing (small batch, frequent eval)
  standard  - Balanced training (good for 8-16GB GPU)
  large     - Large scale training (for 24GB+ GPU)
  resume    - Resume from checkpoint
  
Usage:
  python run_training.py quick
  python run_training.py standard
  python run_training.py large
  python run_training.py resume <checkpoint_path>
        """)
        return
    
    mode = sys.argv[1]
    
    base_cmd = [sys.executable, "train.py"]
    
    if mode == "quick":
        # Quick training for testing
        cmd = base_cmd + [
            "--batch_size", "4",
            "--eval_batch_size", "8", 
            "--learning_rate", "3e-5",
            "--eval_freq", "100",  # Evaluate every 100 steps
            "--log_freq", "10",
            "--max_eval_batches", "20",
            "--wandb_project", "flickr30k-quick-test",
            "--run_name", "quick_test"
        ]
        
    elif mode == "standard":
        # Standard training configuration
        cmd = base_cmd + [
            "--batch_size", "8",
            "--eval_batch_size", "16",
            "--gradient_accumulation_steps", "2",  # Effective batch size: 16
            "--learning_rate", "2e-5",
            "--eval_freq", "1000",
            "--log_freq", "50",
            "--max_eval_batches", "100",
            "--wandb_project", "flickr30k-qwen-training",
            "--run_name", "qwen_0.6b_standard"
        ]
        
    elif mode == "large":
        # Large scale training
        cmd = base_cmd + [
            "--batch_size", "16",
            "--eval_batch_size", "32",
            "--gradient_accumulation_steps", "4",  # Effective batch size: 64
            "--learning_rate", "1e-5",
            "--eval_freq", "2000",
            "--log_freq", "100",
            "--max_eval_batches", "200",
            "--num_workers", "0",  # Set to 0 for Windows compatibility
            "--wandb_project", "flickr30k-qwen-large",
            "--run_name", "qwen_0.6b_large_scale"
        ]
        
    elif mode == "resume":
        if len(sys.argv) < 3:
            print("❌ Please provide checkpoint path for resume mode")
            print("Usage: python run_training.py resume checkpoints/checkpoint_step_5000.pt")
            return
            
        checkpoint_path = sys.argv[2]
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
            
        cmd = base_cmd + [
            "--resume_from_checkpoint", checkpoint_path,
            "--wandb_project", "flickr30k-qwen-training",
        ]
        
    else:
        print(f"❌ Unknown mode: {mode}")
        return
    
    # Run the training
    return_code = run_command(cmd)
    
    if return_code == 0:
        print("✅ Training completed successfully!")
    else:
        print(f"❌ Training failed with return code {return_code}")

if __name__ == "__main__":
    main() 