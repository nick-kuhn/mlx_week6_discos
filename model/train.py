import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import wandb
import argparse
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch.amp
import peft

from .dataset import TLDRDataset, tldr_collate_fn

class SummarizationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track current upload file
        self.current_upload_file = None
        
        # Initialize mixed precision scaler
        self.use_amp = getattr(config, 'use_amp', True) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
                
        # Initialize model and tokenizer
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        print(f"🔥 Mixed precision training: {'enabled' if self.use_amp else 'disabled'}")
        
    def setup_model(self):
        """Initialize model and tokenizer."""
        print("🔧 Setting up model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        # Setup special tokens for tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.model.to(self.device)
        
        # Resize token embeddings if we added tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        
        print(f"📊 Model loaded on {self.device}")
        print(f"📝 Tokenizer vocab size: {len(self.tokenizer)}")


        # Load PEFT adapter
        self.model = peft.get_peft_model(self.model, peft.LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, bias="none"))
        self.model.print_trainable_parameters()
        
    def setup_data(self):
        """Setup datasets and dataloaders."""
        print("📚 Setting up datasets...")
        
        # Create datasets with cache/download configuration
        self.train_dataset = TLDRDataset(
            train_path=self.config.train_path,
            tokenizer=self.tokenizer,
            split='train'
        )
        self.val_dataset = TLDRDataset(
            train_path=self.config.val_path,
            tokenizer=self.tokenizer,
            split='valid'
        )
        
        print(f"📖 Train samples: {len(self.train_dataset)}")
        print(f"📖 Val samples: {len(self.val_dataset)}")
        
        # Create collate function with tokenizer
        def collate_fn(batch):
            return tldr_collate_fn(batch, self.tokenizer)
        
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
        
        # Loss function (cross-entropy for language modeling with label smoothing)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        
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
        
        # Watch model for gradient tracking (disabled heavy logging for performance)
        # wandb.watch(self.model, log='all', log_freq=self.config.log_freq)  # Too slow - uploads 7+ GB every 50 steps!
        wandb.watch(self.model, log=None, log_freq=1000)  # Only log topology, no gradients/parameters
        
    def save_checkpoint(self, is_best=False, suffix=""):
        """Save LoRA adapter checkpoint."""
        # Save only LoRA adapters instead of full model
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'lora_state_dict': self.model.state_dict(),  # This contains only LoRA parameters
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
            'base_model_name': self.config.model_name  # Store base model name for loading
        }
        
        # Only save and upload best models to save disk space
        if is_best:
            # Check if we should actually save this best model
            should_upload = (self.config.use_wandb and self.config.upload_checkpoints and 
                           self.global_step % 2000 == 0)
            
            if should_upload:
                # Clean up previous upload if complete
                upload_slot_available = self.cleanup_previous_upload()
                
                # Check if we should skip saving to avoid disk space issues
                if not upload_slot_available and self.config.delete_after_upload:
                    print(f"⏳ Previous upload still in progress, skipping checkpoint save to avoid disk issues")
                    return None
                
                # Safe to save and upload
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                print(f"💾 Best model saved at step {self.global_step} ({best_path.stat().st_size / 1e9:.1f}GB)")
                
                upload_success = self.upload_checkpoint_to_wandb(best_path, f"best_model_step_{self.global_step}", is_best=True)
                
                if upload_success:
                    # Track this file as currently uploading
                    self.current_upload_file = str(best_path)
                    print(f"✅ Upload queued successfully (tracking for completion)")
                    return best_path
                else:
                    print(f"⚠️  Upload failed, deleting local file to save disk space")
                    if self.config.delete_after_upload:
                        best_path.unlink()
                        print(f"🗑️  Deleted local checkpoint due to upload failure")
                    return None
            else:
                # Not uploading this step - don't save to disk at all
                print(f"🏆 Best model achieved at step {self.global_step} (will save at next 2000-step interval)")
                return None
        else:
            # For non-best checkpoints, create temporary file, upload, then always delete
            temp_checkpoint_path = self.checkpoint_dir / f"temp_checkpoint_step_{self.global_step}.pt"
            torch.save(checkpoint, temp_checkpoint_path)
            
            # Upload if it's time for periodic backup
            if (self.config.use_wandb and self.config.upload_checkpoints and 
                self.global_step % self.config.checkpoint_upload_freq == 0):
                
                print(f"💾 Temporary checkpoint saved ({temp_checkpoint_path.stat().st_size / 1e9:.1f}GB)")
                upload_success = self.upload_checkpoint_to_wandb(temp_checkpoint_path, f"checkpoint_step_{self.global_step}", is_best=False)
                
                if upload_success:
                    print(f"✅ Upload successful")
                else:
                    print(f"⚠️  Upload failed, but deleting anyway to save disk space")
            
            # Always clean up temp file regardless of upload success
            if temp_checkpoint_path.exists():
                temp_checkpoint_path.unlink()
                print(f"🗑️  Deleted temporary checkpoint")
            
            return None
    
    def upload_checkpoint_to_wandb(self, checkpoint_path, artifact_name, is_best=False):
        """Upload LoRA adapter checkpoint to wandb as artifact."""
        try:
            print(f"☁️  Uploading {'best LoRA adapter' if is_best else 'LoRA checkpoint'} to wandb...")
            
            # Create artifact with LoRA-specific metadata
            artifact_type = "lora_adapter" if is_best else "lora_checkpoint"
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=f"LoRA adapter checkpoint at step {self.global_step} (val_loss: {self.best_val_loss:.4f})" if is_best 
                           else f"LoRA training checkpoint at step {self.global_step}",
                metadata={
                    "step": self.global_step,
                    "epoch": self.current_epoch,
                    "val_loss": self.best_val_loss,
                    "base_model_name": self.config.model_name,
                    "adapter_type": "lora",
                    "checkpoint_type": "lora_adapter_only"
                }
            )
            
            # Add checkpoint file
            artifact.add_file(str(checkpoint_path))
            
            # Log artifact to wandb
            wandb.log_artifact(artifact)
            print(f"✅ {'Best model' if is_best else 'Checkpoint'} uploaded successfully!")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to upload checkpoint to wandb: {e}")
            # Don't fail training if upload fails
            return False
        
    def cleanup_checkpoints(self, keep_last=1):
        """Remove old checkpoints to save disk space."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > keep_last:
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()
                print(f"🗑️  Cleaned up old checkpoint: {old_checkpoint.name}")
                
    def load_checkpoint(self, checkpoint_path):
        """Load LoRA adapter checkpoint."""
        print(f"📂 Loading LoRA checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Verify this is a LoRA checkpoint
        if 'lora_state_dict' in checkpoint:
            # Load LoRA adapter weights
            self.model.load_state_dict(checkpoint['lora_state_dict'])
            print("✅ LoRA adapter weights loaded successfully")
        else:
            # Fallback for old checkpoints with full model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("⚠️  Loaded full model state dict (old format)")
            else:
                raise ValueError("Checkpoint missing both 'lora_state_dict' and 'model_state_dict'")
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Verify base model compatibility
        if 'base_model_name' in checkpoint:
            if checkpoint['base_model_name'] != self.config.model_name:
                print(f"⚠️  Warning: Checkpoint base model ({checkpoint['base_model_name']}) "
                      f"differs from current model ({self.config.model_name})")
        
        print(f"✅ Resumed from epoch {self.current_epoch}, step {self.global_step}")
        
    def clear_tensor_cache(self):
        """Clear tensor cache to prevent memory accumulation."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Forward pass - model returns logits and loss directly
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = output.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with proper mixed precision handling
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only update optimizer every accumulation_steps
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.use_amp:
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Track optimizer step count to detect if step actually happened
                optimizer_step_count_before = self.optimizer.state_dict().get('state', {}).get(next(iter(self.optimizer.param_groups[0]['params'])), {}).get('step', 0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Check if optimizer actually stepped by comparing step counts
                optimizer_step_count_after = self.optimizer.state_dict().get('state', {}).get(next(iter(self.optimizer.param_groups[0]['params'])), {}).get('step', 0)
                
                # Only step scheduler if optimizer actually stepped (no NaN/Inf gradients)
                if optimizer_step_count_after > optimizer_step_count_before:
                    self.scheduler.step()
            else:
                # Standard gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
            
            self.optimizer.zero_grad()
        
        # Calculate gradient norm without retaining gradients (memory leak fix)
        grad_norm = 0.0
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Only calculate grad norm when we actually update (avoids memory retention)
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** (1. / 2)
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,  # Report unscaled loss for logging
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    # Forward pass - model returns logits directly
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask, 
                        labels=labels
                    )
                    loss = output.loss
                
                total_loss += loss.item()
                num_batches += 1
                
                # Memory cleanup during evaluation
                if num_batches % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
                
                # More frequent memory cleanup to prevent gradual accumulation
                if self.global_step % 20 == 0:  # Increased frequency from 100 to 20
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Explicit cleanup of batch tensors and metrics (memory leak prevention)
                del batch, metrics
                if self.global_step % 50 == 0:  # Periodic forced garbage collection
                    import gc
                    gc.collect()
                
                # Clear tensor cache periodically to prevent memory growth
                if self.global_step % 500 == 0:
                    self.clear_tensor_cache()
                    print(f"🧹 Cleared tensor cache at step {self.global_step}")
            
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
            
            # Clear tensor cache at end of epoch to prevent memory accumulation
            self.clear_tensor_cache()
            print(f"🧹 End-of-epoch cleanup completed")
        
        print("🎉 Training completed!")
        if self.config.use_wandb:
            wandb.finish()

    def is_file_being_uploaded(self, filepath):
        """Check if wandb is currently uploading/accessing a file."""
        if not filepath or not Path(filepath).exists():
            return False
            
        try:
            import subprocess
            # Check if any wandb process is accessing this file
            result = subprocess.run(['lsof', str(filepath)], 
                                  capture_output=True, text=True, timeout=5)
            
            # Look for wandb processes in the output
            lines = result.stdout.lower()
            return 'wandb' in lines or 'python' in lines
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # If lsof fails, assume not uploading (safer to proceed)
            return False
    
    def cleanup_previous_upload(self):
        """Clean up the previous upload file if upload is complete."""
        if self.current_upload_file and Path(self.current_upload_file).exists():
            if not self.is_file_being_uploaded(self.current_upload_file):
                # Upload complete, safe to delete
                Path(self.current_upload_file).unlink()
                print(f"🗑️  Cleaned up completed upload: {Path(self.current_upload_file).name}")
                self.current_upload_file = None
                return True
            else:
                print(f"⏳ Previous upload still in progress: {Path(self.current_upload_file).name}")
                return False
        return True  # No previous file to clean up


def parse_args():
    parser = argparse.ArgumentParser(description='Train Summarization Model')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B-Base')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'linear'])
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision training')

    # Data arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading (set to 0 for Windows compatibility)')
    parser.add_argument('--train_path', type=str, required=True, default='CarperAI/TLDR-dataset')
    parser.add_argument('--val_path', type=str, required=True, default='CarperAI/TLDR-dataset')
    
    # Logging and checkpointing
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='summarization-finetuning')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=500)  # Evaluate every 500 steps
    parser.add_argument('--max_eval_batches', type=int, default=100)  # Limit eval to save time
    parser.add_argument('--upload_checkpoints', action='store_true', default=False,
                        help='Upload checkpoints to wandb as artifacts for remote backup')
    parser.add_argument('--checkpoint_upload_freq', type=int, default=2000,
                        help='Upload regular checkpoints every N steps (best models always uploaded)')
    parser.add_argument('--delete_after_upload', action='store_true', default=True,
                        help='Delete local checkpoints after successful upload to save disk space')
    
    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"qwen_summarization_{timestamp}"
    
    print("🎯 Training Configuration:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")
    
    # Create trainer and start training
    trainer = SummarizationTrainer(args)
    trainer.train() 