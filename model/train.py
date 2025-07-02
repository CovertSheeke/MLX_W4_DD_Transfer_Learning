import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import wandb
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import os

from dataset import FlickrDataset, qwen_collate_fn
from model import QwenModel


class VisionLanguageTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def setup_model(self):
        """Initialize model and tokenizer."""
        print("🔧 Setting up model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Setup special tokens for tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Add image tokens
        special_tokens = {'additional_special_tokens': ['<|im_start|>', '<|im_end|>']}
        self.tokenizer.add_special_tokens(special_tokens)
        
        self.model = QwenModel(self.config.model_name)
        self.model.to(self.device)
        
        # Resize token embeddings if we added tokens
        self.model.qwen.resize_token_embeddings(len(self.tokenizer))
        
        # Add language modeling head (don't create it dynamically in training)
        vocab_size = len(self.tokenizer)
        hidden_size = self.model.qwen.config.hidden_size
        self.model.lm_head = nn.Linear(hidden_size, vocab_size).to(self.device)
        
        print(f"📊 Model loaded on {self.device}")
        print(f"📝 Tokenizer vocab size: {len(self.tokenizer)}")
        
    def setup_data(self):
        """Setup datasets and dataloaders."""
        print("📚 Setting up datasets...")
        
        # Create datasets
        self.train_dataset = FlickrDataset(split='train')
        self.val_dataset = FlickrDataset(split='validation')
        
        print(f"📖 Train samples: {len(self.train_dataset)}")
        print(f"📖 Val samples: {len(self.val_dataset)}")
        
        # Create collate function with tokenizer
        def collate_fn(batch):
            return qwen_collate_fn(batch, self.tokenizer)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        print("⚙️ Setting up training components...")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Calculate total steps (accounting for gradient accumulation)
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * self.config.num_epochs
        
        # Learning rate scheduler
        if self.config.lr_scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.total_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        else:
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.total_steps
            )
        
        # Loss function (cross-entropy for language modeling)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        print(f"🎯 Total training steps: {self.total_steps}")
        print(f"📈 Learning rate scheduler: {self.config.lr_scheduler}")
        
    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.run_name,
            config=vars(self.config),
            resume='allow' if self.config.resume_from_checkpoint else None
        )
        
        # Watch model for gradient tracking
        wandb.watch(self.model, log='all', log_freq=self.config.log_freq)
        
    def save_checkpoint(self, is_best=False, suffix=""):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved at step {self.global_step}")
            
        # Keep only last N checkpoints to save space
        self.cleanup_checkpoints()
        
        return checkpoint_path
        
    def cleanup_checkpoints(self, keep_last=3):
        """Remove old checkpoints to save disk space."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > keep_last:
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()
                
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ Resumed from epoch {self.current_epoch}, step {self.global_step}")
        
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        images = batch['images'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            image_data=images,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Calculate loss
        logits = outputs.last_hidden_state
        # Project to vocabulary size for language modeling
        lm_logits = self.model.lm_head(logits)
        
        # Shift for causal LM: predict next token
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Only update optimizer every accumulation_steps
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm': torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf')).item()
        }
        
    def evaluate(self):
        """Run evaluation on validation set."""
        print("🔍 Running evaluation...")
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                if batch is None:
                    continue
                    
                # Move batch to device
                images = batch['images'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    image_data=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate loss
                logits = outputs.last_hidden_state
                lm_logits = self.model.lm_head(logits)
                
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Don't evaluate entire val set every time (too expensive)
                if num_batches >= self.config.max_eval_batches:
                    break
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'val_loss': avg_loss}
        
    def train(self):
        """Main training loop."""
        print("🚀 Starting training...")
        
        # Initialize wandb
        if self.config.use_wandb:
            self.init_wandb()
        
        # Load checkpoint if resuming
        if self.config.resume_from_checkpoint:
            checkpoint_path = self.config.resume_from_checkpoint
            if Path(checkpoint_path).exists():
                self.load_checkpoint(checkpoint_path)
            else:
                print(f"⚠️ Checkpoint {checkpoint_path} not found, starting fresh")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            print(f"\n📅 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training loop
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                if batch is None:
                    continue
                
                # Training step
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{metrics['lr']:.2e}",
                    'step': self.global_step
                })
                
                # Logging
                if self.global_step % self.config.log_freq == 0:
                    if self.config.use_wandb:
                        wandb.log({
                            'train/loss': metrics['loss'],
                            'train/learning_rate': metrics['lr'],
                            'train/grad_norm': metrics['grad_norm'],
                            'global_step': self.global_step,
                            'epoch': epoch
                        })
                
                # Mid-epoch evaluation and checkpointing
                if self.global_step % self.config.eval_freq == 0:
                    eval_metrics = self.evaluate()
                    
                    if self.config.use_wandb:
                        wandb.log({
                            'eval/val_loss': eval_metrics['val_loss'],
                            'global_step': self.global_step
                        })
                    
                    # Save checkpoint
                    is_best = eval_metrics['val_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = eval_metrics['val_loss']
                    
                    self.save_checkpoint(is_best=is_best)
                    
                    print(f"\n📊 Step {self.global_step} - Val Loss: {eval_metrics['val_loss']:.4f} {'🏆' if is_best else ''}")
                
                # Memory cleanup
                if self.global_step % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # End of epoch summary
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            elapsed_time = time.time() - start_time
            
            print(f"✅ Epoch {epoch + 1} completed:")
            print(f"   📈 Avg Loss: {avg_epoch_loss:.4f}")
            print(f"   ⏰ Time: {elapsed_time / 3600:.2f}h")
            print(f"   🔢 Steps: {self.global_step}")
            
            # Final epoch evaluation
            eval_metrics = self.evaluate()
            if self.config.use_wandb:
                wandb.log({
                    'epoch/train_loss': avg_epoch_loss,
                    'epoch/val_loss': eval_metrics['val_loss'],
                    'epoch/epoch': epoch,
                    'global_step': self.global_step
                })
        
        print("🎉 Training completed!")
        if self.config.use_wandb:
            wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Vision-Language Model')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B-Base')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'linear'])
    
    # Data arguments
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Logging and checkpointing
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='vision-language-training')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=1000)  # Evaluate every 1000 steps
    parser.add_argument('--max_eval_batches', type=int, default=100)  # Limit eval to save time
    
    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"qwen_vlm_{timestamp}"
    
    print("🎯 Training Configuration:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")
    
    # Create trainer and start training
    trainer = VisionLanguageTrainer(args)
    trainer.train() 