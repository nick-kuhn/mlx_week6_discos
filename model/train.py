import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR
import wandb
import argparse
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import os
import torch.amp
from accelerate import Accelerator
import peft

from .dataset import TLDRDataset, tldr_collate_fn
from .evaluate_model import get_rouge_scores, get_examples

class SummarizationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.checkpoint_dir = Path(config.model.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track current upload file
        self.current_upload_file = None
        
        # Initialize mixed precision scaler
        self.accelerator = Accelerator(mixed_precision = "fp16" if config.training.use_amp else "no")
        self.use_amp = getattr(config.training, 'use_amp', True) #and torch.cuda.is_available() #handled by accelerator
        self.scaler = self.accelerator.scaler if self.use_amp else None
                
        # Initialize model and tokenizer
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
        # Initialize reward model if enabled
        if self.config.logging.reward_evaluation:
            self.setup_reward_model()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_reward_improvement = 0.0  # Track best reward improvement (starts at baseline)
        
        print(f"🔥 Mixed precision training: {'enabled' if self.use_amp else 'disabled'}")
        
    def setup_reward_model(self):
        """Initialize reward model for evaluation with lazy loading."""
        print("🎯 Setting up reward model...")
        
        # Store model names for lazy loading
        self.reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        self.baseline_model_name = self.config.model.name
        
        # Initialize as None - will be loaded when needed
        self.reward_model = None
        self.reward_tokenizer = None
        self.baseline_model = None
        
        print("✅ Reward model setup complete (lazy loading enabled)")
    
    def load_reward_model(self):
        """Lazy load reward model when needed."""
        if self.reward_model is None:
            print("🔄 Loading reward model...")
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_model_name
            )
            self.reward_tokenizer = AutoTokenizer.from_pretrained(
                self.reward_model_name
            )
            self.reward_model.to(self.device)
            self.reward_model.eval()
            print("✅ Reward model loaded")
        return self.reward_model, self.reward_tokenizer
    
    def load_baseline_model(self):
        """Lazy load baseline model when needed."""
        if self.baseline_model is None:
            print("🔄 Loading baseline model...")
            self.baseline_model = AutoModelForCausalLM.from_pretrained(self.baseline_model_name)
            if self.baseline_model.config.pad_token_id is None:
                self.baseline_model.config.pad_token_id = self.tokenizer.eos_token_id
            self.baseline_model.to(self.device)
            self.baseline_model.eval()
            print("✅ Baseline model loaded")
        return self.baseline_model
    
    def unload_reward_models(self):
        """Unload reward models to free memory."""
        if self.reward_model is not None:
            del self.reward_model
            del self.reward_tokenizer
            self.reward_model = None
            self.reward_tokenizer = None
            
        if self.baseline_model is not None:
            del self.baseline_model
            self.baseline_model = None
            
        torch.cuda.empty_cache()
        print("🧹 Reward models unloaded from memory")
        
    def calculate_reward_scores(self, reward_inputs):
        """Calculate reward scores for given inputs using the reward model."""
        reward_model, reward_tokenizer = self.load_reward_model()
        rewards = []
        
        with torch.no_grad():
            for text in reward_inputs:
                try:
                    inputs = reward_tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512
                    ).to(self.device)
                    
                    outputs = reward_model(**inputs)
                    # The reward model outputs logits, score is the first logit
                    score = outputs.logits[0].item()
                    rewards.append(score)
                except Exception as e:
                    print(f"⚠️ Error calculating reward score: {e}")
                    rewards.append(0.0)  # Fallback score
        
        return rewards
    
    def generate_summary(self, model, prompt_text):
        """Generate a summary using the given model."""
        try:
            inputs = self.tokenizer(
                prompt_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract only the generated part (excluding the input prompt)
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
            return generated_text
        except Exception as e:
            print(f"⚠️ Error generating summary: {e}")
            return "Error generating summary"
    
    def evaluate_with_reward_model(self, num_samples=10):
        """Evaluate model using reward model comparison."""
        print("🎯 Running reward model evaluation...")
        
        try:
            # Get sample data for evaluation
            sample_data = []
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                if batch is None:
                    continue
                    
                # Get the prompt (input text before summary)
                mask_length = (batch['labels'] == -100).sum().item()
                prompt_text = self.tokenizer.decode(
                    batch['input_ids'][0][:mask_length], 
                    skip_special_tokens=True
                )
                
                # Get reference summary
                reference_summary = self.tokenizer.decode(
                    batch['labels'][0][mask_length:], 
                    skip_special_tokens=True
                )
                
                sample_data.append({
                    'prompt': prompt_text,
                    'reference': reference_summary
                })
            
            if not sample_data:
                print("⚠️ No samples found for reward evaluation")
                return {
                    'reward_finetuned_avg': 0.0,
                    'reward_baseline_avg': 0.0,
                    'reward_improvement': 0.0,
                    'reward_improvement_std': 0.0
                }
            
            # Load baseline model for summary generation
            baseline_model = self.load_baseline_model()
            
            # Generate summaries with both models
            finetuned_summaries = []
            baseline_summaries = []
            
            for sample in sample_data:
                # Generate with current finetuned model
                finetuned_summary = self.generate_summary(self.model, sample['prompt'])
                finetuned_summaries.append(finetuned_summary)
                
                # Generate with baseline model
                baseline_summary = self.generate_summary(baseline_model, sample['prompt'])
                baseline_summaries.append(baseline_summary)
            
            # Prepare inputs for reward model
            finetuned_inputs = [
                f"{sample['prompt']} {summary}" 
                for sample, summary in zip(sample_data, finetuned_summaries)
            ]
            baseline_inputs = [
                f"{sample['prompt']} {summary}" 
                for sample, summary in zip(sample_data, baseline_summaries)
            ]
            
            # Calculate rewards
            finetuned_rewards = self.calculate_reward_scores(finetuned_inputs)
            baseline_rewards = self.calculate_reward_scores(baseline_inputs)
            
            # Calculate metrics
            reward_improvements = [f - b for f, b in zip(finetuned_rewards, baseline_rewards)]
            
            metrics = {
                'reward_finetuned_avg': np.mean(finetuned_rewards),
                'reward_baseline_avg': np.mean(baseline_rewards),
                'reward_improvement': np.mean(reward_improvements),
                'reward_improvement_std': np.std(reward_improvements)
            }
            
            print(f"📊 Reward Evaluation Results:")
            print(f"   Finetuned avg: {metrics['reward_finetuned_avg']:.4f}")
            print(f"   Baseline avg: {metrics['reward_baseline_avg']:.4f}")
            print(f"   Improvement: {metrics['reward_improvement']:.4f} ± {metrics['reward_improvement_std']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Error during reward evaluation: {e}")
            return {
                'reward_finetuned_avg': 0.0,
                'reward_baseline_avg': 0.0,
                'reward_improvement': 0.0,
                'reward_improvement_std': 0.0
            }
        
        finally:
            # Always clean up memory after evaluation
            self.unload_reward_models()
        
    def setup_model(self):
        """Initialize model and tokenizer."""
        print("🔧 Setting up model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model.name)
        # Setup special tokens for tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.model.to(self.device)
        
        # Resize token embeddings if we added tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        
        print(f"📊 Model loaded on {self.device}")
        print(f"📝 Tokenizer vocab size: {len(self.tokenizer)}")


        # Load PEFT adapter with configurable settings
        lora_config = peft.LoraConfig(
            r=getattr(self.config.advanced.lora, 'r', 8),
            lora_alpha=getattr(self.config.advanced.lora, 'alpha', 32),
            lora_dropout=getattr(self.config.advanced.lora, 'dropout', 0.05),
            bias=getattr(self.config.advanced.lora, 'bias', "none"),
            target_modules=getattr(self.config.advanced.lora, 'target_modules', None)
        )
        self.model = peft.get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def setup_data(self):
        """Setup datasets and dataloaders."""
        print("📚 Setting up datasets...")
        
        # Create datasets with cache/download configuration
        self.train_dataset = TLDRDataset(
            train_path=self.config.data.train_path,
            tokenizer=self.tokenizer,
            split='train'
        )
        self.val_dataset = TLDRDataset(
            train_path=self.config.data.val_path,
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
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        print("⚙️ Setting up training components...")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Calculate total steps (accounting for gradient accumulation)
        steps_per_epoch = len(self.train_loader) // self.config.training.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * self.config.training.num_epochs
        
        # Learning rate scheduler with warmup
        if self.config.training.lr_scheduler == 'cosine':
            # Use OneCycleLR for cosine with warmup
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=self.total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos',
                div_factor=10,  # Initial LR = max_lr / div_factor
                final_div_factor=100  # Final LR = max_lr / final_div_factor
            )
        elif self.config.training.lr_scheduler == 'cosine_legacy':
            # Legacy cosine scheduler without warmup (for comparison)
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.total_steps,
                eta_min=self.config.training.learning_rate * 0.1
            )
        else:
            # Linear scheduler with warmup using OneCycleLR
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=self.total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='linear',
                div_factor=10,  # Initial LR = max_lr / div_factor
                final_div_factor=100  # Final LR = max_lr / final_div_factor
            )
        
        # Loss function (cross-entropy for language modeling with label smoothing)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        
        print(f"🎯 Total training steps: {self.total_steps}")
        print(f"📈 Learning rate scheduler: {self.config.training.lr_scheduler}")
        
    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.logging.wandb_project,
            name=self.config.get_run_name(),
            config=self.config.dict(),
            resume='allow' if self.config.resume.resume_from_checkpoint else None
        )
        
        # Watch model for gradient tracking (disabled heavy logging for performance)
        # wandb.watch(self.model, log='all', log_freq=self.config.logging.log_freq)  # Too slow - uploads 7+ GB every 50 steps!
        wandb.watch(self.model, log=None, log_freq=1000)  # Only log topology, no gradients/parameters
        
    def save_checkpoint(self, is_best=False, suffix=""):
        """Save LoRA adapter checkpoint."""
        # Save only LoRA adapters instead of full model
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'lora_state_dict': {k: v for k, v in self.model.state_dict().items() if 'lora_' in k},  # Only LoRA parameters
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.dict(),
            'base_model_name': self.config.model.name  # Store base model name for loading
        }
        
        # Only save and upload best models to save disk space
        if is_best:
            # Check if we should actually save this best model
            should_upload = (self.config.logging.use_wandb and self.config.logging.upload_checkpoints and 
                           self.global_step % 2000 == 0)
            
            if should_upload:
                # Clean up previous upload if complete
                upload_slot_available = self.cleanup_previous_upload()
                
                # Check if we should skip saving to avoid disk space issues
                if not upload_slot_available and self.config.logging.delete_after_upload:
                    print(f"⏳ Previous upload still in progress, skipping checkpoint save to avoid disk issues")
                    return None
                
                # Safe to save and upload
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                print(f"💾 Best model saved at step {self.global_step} ({best_path.stat().st_size / 1e9:.1f}GB)")
                
                upload_success = self.upload_checkpoint_to_wandb(best_path, f"best_finetuned_model", is_best=True)
                
                if upload_success:
                    # Track this file as currently uploading
                    self.current_upload_file = str(best_path)
                    print(f"✅ Upload queued successfully (tracking for completion)")
                    return best_path
                else:
                    print(f"⚠️  Upload failed, deleting local file to save disk space")
                    if self.config.logging.delete_after_upload:
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
            if (self.config.logging.use_wandb and self.config.logging.upload_checkpoints and 
                self.global_step % self.config.logging.checkpoint_upload_freq == 0):
                
                print(f"💾 Temporary checkpoint saved ({temp_checkpoint_path.stat().st_size / 1e9:.1f}GB)")
                upload_success = self.upload_checkpoint_to_wandb(temp_checkpoint_path, f"checkpoint_finetuned_model", is_best=False)
                
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
                    "base_model_name": self.config.model.name,
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
            if checkpoint['base_model_name'] != self.config.model.name:
                print(f"⚠️  Warning: Checkpoint base model ({checkpoint['base_model_name']}) "
                      f"differs from current model ({self.config.model.name})")
        
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
            loss = loss / self.config.training.gradient_accumulation_steps
        
        # Backward pass with proper mixed precision handling
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only update optimizer every accumulation_steps
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            if self.use_amp:
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
            
            self.optimizer.zero_grad()
        
        # Calculate gradient norm without retaining gradients (memory leak fix)
        grad_norm = 0.0
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            # Only calculate grad norm when we actually update (avoids memory retention)
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** (1. / 2)
        
        return {
            'loss': loss.item() * self.config.training.gradient_accumulation_steps,  # Report unscaled loss for logging
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
                if num_batches >= self.config.logging.max_eval_batches:
                    break
        
        # Generate examples for ROUGE score calculation
        _, references, predictions = get_examples(
            self.model, 
            self.tokenizer, 
            self.val_dataset, 
            self.device, 
            num_examples=10,  # Use more examples for better ROUGE calculation
            verbose=self.config.logging.verbose_evals
        )
        
        # Calculate ROUGE scores
        rouge_scores = get_rouge_scores(predictions, references)
        
        # Calculate reward model metrics if enabled
        reward_metrics = {}
        if self.config.logging.reward_evaluation:
            reward_metrics = self.evaluate_with_reward_model(num_samples=5)  # Use fewer samples to save time
        
        avg_loss = total_loss / max(num_batches, 1)
        eval_results = {'val_loss': avg_loss, 'rouge_scores': rouge_scores}
        eval_results.update(reward_metrics)
        
        return eval_results
        
    def train(self):
        """Main training loop."""
        print("🚀 Starting training...")
        
        # Initialize wandb
        if self.config.logging.use_wandb:
            self.init_wandb()
        
        # Load checkpoint if resuming
        if self.config.resume.resume_from_checkpoint:
            checkpoint_path = self.config.resume.resume_from_checkpoint
            if Path(checkpoint_path).exists():
                self.load_checkpoint(checkpoint_path)
            else:
                print(f"⚠️ Checkpoint {checkpoint_path} not found, starting fresh")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            print(f"\n📅 Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
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
                if self.global_step % self.config.logging.log_freq == 0:
                    if self.config.logging.use_wandb:
                        wandb.log({
                            'train/loss': metrics['loss'],
                            'train/learning_rate': metrics['lr'],
                            'train/grad_norm': metrics['grad_norm'],
                            'global_step': self.global_step,
                            'epoch': epoch
                        })
                
                # Mid-epoch evaluation and checkpointing
                if self.global_step % self.config.logging.eval_freq == 0:
                    eval_metrics = self.evaluate()
                    

                    
                    if self.config.logging.use_wandb:
                        log_dict = {
                            'eval/val_loss': eval_metrics['val_loss'],
                            'eval/rouge-L': eval_metrics['rouge_scores']['rougeL'],
                            'global_step': self.global_step
                        }
                        
                        # Add reward metrics if available
                        if self.config.logging.reward_evaluation and 'reward_improvement' in eval_metrics:
                            log_dict.update({
                                'eval/reward_improvement': eval_metrics['reward_improvement'],
                                'eval/reward_finetuned_avg': eval_metrics['reward_finetuned_avg'],
                                'eval/reward_baseline_avg': eval_metrics['reward_baseline_avg']
                            })
                        
                        wandb.log(log_dict)
                    
                    # Save checkpoint - use reward improvement if available, otherwise val_loss
                    if self.config.logging.reward_evaluation and 'reward_improvement' in eval_metrics:
                        is_best = eval_metrics['reward_improvement'] > self.best_reward_improvement
                        if is_best:
                            self.best_reward_improvement = eval_metrics['reward_improvement']
                    else:
                        is_best = eval_metrics['val_loss'] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = eval_metrics['val_loss']
                    
                    self.save_checkpoint(is_best=is_best)
                    
                    print(f"\n📊 Step {self.global_step} - Val Loss: {eval_metrics['val_loss']:.4f} {'🏆' if is_best else ''}")
                    #print main rougeL score
                    print(f"🔍 ROUGE-L: {eval_metrics['rouge_scores']['rougeL']:.4f}")
                    
                    # Print reward metrics if available
                    if self.config.logging.reward_evaluation and 'reward_improvement' in eval_metrics:
                        best_indicator = '🏆' if is_best else ''
                        print(f"🎯 Reward Improvement: {eval_metrics['reward_improvement']:.4f} {best_indicator}")
                        print(f"📈 Best Reward Improvement: {self.best_reward_improvement:.4f}")

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
            if self.config.logging.use_wandb:
                epoch_log_dict = {
                    'epoch/train_loss': avg_epoch_loss,
                    'epoch/val_loss': eval_metrics['val_loss'],
                    'epoch/epoch': epoch,
                    'global_step': self.global_step
                }
                
                # Add reward metrics if available
                if self.config.logging.reward_evaluation and 'reward_improvement' in eval_metrics:
                    epoch_log_dict.update({
                        'epoch/reward_improvement': eval_metrics['reward_improvement'],
                        'epoch/reward_finetuned_avg': eval_metrics['reward_finetuned_avg'],
                        'epoch/reward_baseline_avg': eval_metrics['reward_baseline_avg']
                    })
                
                wandb.log(epoch_log_dict)
            
            # Clear tensor cache at end of epoch to prevent memory accumulation
            self.clear_tensor_cache()
            print(f"🧹 End-of-epoch cleanup completed")
        
        print("🎉 Training completed!")
        if self.config.logging.use_wandb:
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
    
    # Configuration file arguments (keep defaults for utility flags)
    parser.add_argument('--config', type=str,
                        help='Path to YAML configuration file')
    parser.add_argument('--print-config', action='store_true',
                        help='Print loaded configuration and exit')
    parser.add_argument('--validate-config', action='store_true',
                        help='Validate configuration and exit')
    parser.add_argument('--save-config', type=str,
                        help='Save current configuration to specified file and exit')
    
    # Model arguments
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--checkpoint_dir', type=str)
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--lr_scheduler', type=str, choices=['cosine', 'linear'])
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--no_use_amp', action='store_true',
                        help='Disable automatic mixed precision training')

    # Data arguments
    parser.add_argument('--num_workers', type=int,
                        help='Number of worker processes for data loading (set to 0 for Windows compatibility)')
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--val_path', type=str)
    
    # Logging and checkpointing
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--no_use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--log_freq', type=int)
    parser.add_argument('--eval_freq', type=int)
    parser.add_argument('--max_eval_batches', type=int)
    parser.add_argument('--upload_checkpoints', action='store_true',
                        help='Upload checkpoints to wandb as artifacts for remote backup')
    parser.add_argument('--no_upload_checkpoints', action='store_true')
    parser.add_argument('--checkpoint_upload_freq', type=int,
                        help='Upload regular checkpoints every N steps (best models always uploaded)')
    parser.add_argument('--delete_after_upload', action='store_true',
                        help='Delete local checkpoints after successful upload to save disk space')
    parser.add_argument('--no_delete_after_upload', action='store_true')
    parser.add_argument('--verbose-evals', action='store_true',
                        help='Show detailed evaluation output including original prompts during training')
    parser.add_argument('--reward-evaluation', action='store_true',
                        help='Enable reward model evaluation during training (compares finetuned vs baseline)')
    
    # Resume training
    parser.add_argument('--resume-from-checkpoint', type=str)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Import config modules
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import ConfigLoader, ConfigValidator
    from pydantic import ValidationError
    
    # Load configuration
    loader = ConfigLoader()
    
    try:
        if args.config:
            # Load from config file
            config = loader.load_config(args.config)
        else:
            # Create config from CLI args (backward compatibility)
            config = loader.create_config_from_args(args)
    except ValidationError as e:
        print(f"❌ Configuration validation failed:")
        print(f"   {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Apply CLI overrides if config file was used
    if args.config:
        
        # Apply CLI overrides
        cli_overrides = {}
        args_dict = vars(args)
        
        # Handle boolean flags with negative counterparts
        def handle_boolean_flag(positive_key, negative_key, config_path):
            if args_dict.get(positive_key):
                cli_overrides[config_path] = True
            elif args_dict.get(negative_key):
                cli_overrides[config_path] = False
        
        for key, value in args_dict.items():
            if key in ['config', 'print_config', 'validate_config', 'save_config']:
                continue
            if key.startswith('no_'):  # Skip negative flags, handled separately
                continue
                
            # Only apply overrides for non-None values
            if value is not None:
                # Map CLI arguments to config structure
                if key == 'model_name':
                    cli_overrides['model.name'] = value
                elif key == 'checkpoint_dir':
                    cli_overrides['model.checkpoint_dir'] = value
                elif key in ['num_epochs', 'batch_size', 'eval_batch_size', 'gradient_accumulation_steps', 
                           'learning_rate', 'weight_decay', 'max_grad_norm', 'lr_scheduler']:
                    cli_overrides[f'training.{key}'] = value
                elif key in ['train_path', 'val_path', 'num_workers']:
                    cli_overrides[f'data.{key}'] = value
                elif key in ['wandb_project', 'run_name', 'log_freq', 'eval_freq', 'max_eval_batches',
                           'checkpoint_upload_freq', 'verbose_evals', 'reward_evaluation']:
                    cli_overrides[f'logging.{key}'] = value
                elif key == 'resume_from_checkpoint':
                    cli_overrides['resume.resume_from_checkpoint'] = value
        
        # Handle boolean flags with positive/negative variants
        handle_boolean_flag('use_amp', 'no_use_amp', 'training.use_amp')
        handle_boolean_flag('use_wandb', 'no_use_wandb', 'logging.use_wandb')
        handle_boolean_flag('upload_checkpoints', 'no_upload_checkpoints', 'logging.upload_checkpoints')
        handle_boolean_flag('delete_after_upload', 'no_delete_after_upload', 'logging.delete_after_upload')
        
        # Apply overrides
        if cli_overrides:
            try:
                config = loader.apply_overrides(config, cli_overrides)
            except ValidationError as e:
                print(f"❌ CLI override validation failed:")
                print(f"   {e}")
                sys.exit(1)
    
    # Generate run name if not provided
    if config.logging.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.logging.run_name = f"qwen_summarization_{timestamp}"
    
    # Handle utility flags
    if args.print_config:
        config.print_config()
        sys.exit(0)
    
    if args.validate_config:
        validator = ConfigValidator()
        if validator.validate(config):
            validator.print_validation_results()
            print("✅ Configuration is valid!")
        else:
            validator.print_validation_results()
            sys.exit(1)
    
    if args.save_config:
        loader.save_config(config, args.save_config)
        print(f"✅ Configuration saved to {args.save_config}")
        sys.exit(0)
    
    # Validate configuration before training
    try:
        validator = ConfigValidator()
        if not validator.validate(config):
            print("❌ Configuration validation failed:")
            validator.print_validation_results()
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error during configuration validation: {e}")
        sys.exit(1)
    
    # Print configuration (always shown before training)
    config.print_config()
    
    # Create trainer and start training
    trainer = SummarizationTrainer(config)
    trainer.train() 