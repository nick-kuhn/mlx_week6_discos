#!/usr/bin/env python3
"""Debug version of training script with minimal evaluation to reproduce wandb upload issue."""

import os
# Set tokenizer parallelism before importing transformers to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import wandb
import argparse
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch.amp
from accelerate import Accelerator
import peft

from model.dataset import TLDRDataset, tldr_collate_fn
from model.evaluate_model import get_rouge_scores, get_examples
from config.loader import ConfigLoader

class DebugSummarizationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.checkpoint_dir = Path(config.model.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize mixed precision
        cuda_available = torch.cuda.is_available()
        self.use_amp = config.training.use_amp and cuda_available
        self.accelerator = Accelerator(mixed_precision="fp16" if self.use_amp else "no")
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
        self.best_reward_improvement = float('-inf')
        
        print(f"Debug trainer initialized on {self.device}")
        
    def setup_reward_model(self):
        """Initialize reward model for evaluation."""
        print("Setting up reward model for debug...")
        
        # Store model names for lazy loading
        self.reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        
        # Initialize reward model as None - will be loaded when needed
        self.reward_model = None
        self.reward_tokenizer = None
        
        # Baseline evaluation cache
        self.baseline_reward_scores = None
        self.baseline_validation_samples = None
        self._baseline_initialized = False
        
        print("Reward model setup complete")
    
    def load_reward_model(self):
        """Lazy load reward model when needed."""
        if self.reward_model is None:
            print("Loading reward model...")
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_model_name
            )
            self.reward_tokenizer = AutoTokenizer.from_pretrained(
                self.reward_model_name
            )
            self.reward_model.to(self.device)
            self.reward_model.eval()
            print("Reward model loaded")
        return self.reward_model, self.reward_tokenizer
    
    def get_validation_samples(self, num_samples=1):
        """Get minimal validation samples for debug."""
        samples = []
        for i, batch in enumerate(self.val_loader):
            if i >= num_samples or batch is None:
                break
                
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
            
            samples.append({
                'prompt': prompt_text,
                'reference': reference_summary
            })
        return samples
    
    def initialize_baseline_rewards(self, num_samples=1):
        """Initialize baseline reward scores - minimal version."""
        if self._baseline_initialized:
            return
            
        print("Initializing baseline reward scores (debug mode - 1 sample)...")
        
        # Get minimal validation samples
        self.baseline_validation_samples = self.get_validation_samples(num_samples)
        
        if not self.baseline_validation_samples:
            print("No validation samples found")
            self._baseline_initialized = True
            self.baseline_reward_scores = []
            return
        
        # Generate summaries with baseline model
        baseline_summaries = []
        
        # Check model state before adapter manipulation
        print(f"Model adapter enabled before disable_adapter: {self.model.peft_config if hasattr(self.model, 'peft_config') else 'N/A'}")
        
        for sample in self.baseline_validation_samples:
            with self.model.disable_adapter():
                summary = self.generate_summary(self.model, sample['prompt'])
                baseline_summaries.append(summary)
        
        # Check model state after adapter manipulation
        print(f"Model adapter enabled after disable_adapter: {self.model.peft_config if hasattr(self.model, 'peft_config') else 'N/A'}")
        
        # Force model back to training mode and ensure adapters are enabled
        self.model.train()
        if hasattr(self.model, 'enable_adapter'):
            self.model.enable_adapter()
            print("Explicitly re-enabled adapter after baseline evaluation")
        
        # Calculate baseline rewards
        baseline_inputs = [
            f"{sample['prompt']} {summary}" 
            for sample, summary in zip(self.baseline_validation_samples, baseline_summaries)
        ]
        
        self.baseline_reward_scores = self.calculate_reward_scores(baseline_inputs)
        self._baseline_initialized = True
        
        print(f"Baseline rewards initialized: avg={np.mean(self.baseline_reward_scores):.4f}")
        
    def calculate_reward_scores(self, reward_inputs):
        """Calculate reward scores - minimal version."""
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
                    score = outputs.logits[0].item()
                    rewards.append(score)
                except Exception as e:
                    print(f"Error calculating reward score: {e}")
                    rewards.append(0.0)
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
                    max_new_tokens=50,  # Reduced for debug
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
            return generated_text
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary"
    
    def evaluate_with_reward_model(self, num_samples=1):
        """Evaluate model using reward model - minimal version."""
        print("Running reward model evaluation (debug mode)...")
        
        try:
            # Initialize baseline rewards if not done yet
            if not self._baseline_initialized:
                self.initialize_baseline_rewards(num_samples)
            
            if not self.baseline_validation_samples or not self.baseline_reward_scores:
                print("No baseline data available")
                return self.get_default_reward_metrics()
            
            # Generate summaries with current finetuned model
            finetuned_summaries = []
            for sample in self.baseline_validation_samples:
                summary = self.generate_summary(self.model, sample['prompt'])
                finetuned_summaries.append(summary)
            
            # Prepare inputs for reward model
            finetuned_inputs = [
                f"{sample['prompt']} {summary}"
                for sample, summary in zip(self.baseline_validation_samples, finetuned_summaries)
            ]
            
            # Calculate rewards for current finetuned model
            finetuned_rewards = self.calculate_reward_scores(finetuned_inputs)
            
            # Calculate improvement using cached baseline scores
            reward_improvements = [
                f - b for f, b in zip(finetuned_rewards, self.baseline_reward_scores)
            ]
            
            # NOTE: These are the numpy values that might cause wandb upload issues
            metrics = {
                'reward_finetuned_avg': float(np.mean(finetuned_rewards)),
                'reward_baseline_avg': float(np.mean(self.baseline_reward_scores)),
                'reward_improvement': float(np.mean(reward_improvements)),
                'reward_improvement_std': float(np.std(reward_improvements))
            }
            
            print(f"Reward Evaluation Results:")
            print(f"  Finetuned avg: {metrics['reward_finetuned_avg']:.4f}")
            print(f"  Baseline avg: {metrics['reward_baseline_avg']:.4f}")
            print(f"  Improvement: {metrics['reward_improvement']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error during reward evaluation: {e}")
            return self.get_default_reward_metrics()
    
    def get_default_reward_metrics(self):
        """Return default reward metrics when evaluation fails."""
        return {
            'reward_finetuned_avg': 0.0,
            'reward_baseline_avg': 0.0,
            'reward_improvement': 0.0,
            'reward_improvement_std': 0.0
        }
        
    def setup_model(self):
        """Initialize model and tokenizer."""
        print("Setting up model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model.name)
        
        # Setup special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Load PEFT adapter
        lora_config = peft.LoraConfig(
            r=getattr(self.config.advanced.lora, 'r', 8),
            lora_alpha=getattr(self.config.advanced.lora, 'alpha', 32),
            lora_dropout=getattr(self.config.advanced.lora, 'dropout', 0.05),
            bias=getattr(self.config.advanced.lora, 'bias', "none"),
            target_modules=getattr(self.config.advanced.lora, 'target_modules', None)
        )
        self.model = peft.get_peft_model(self.model, lora_config)
        
        print(f"Model loaded on {self.device}")
        
    def setup_data(self):
        """Setup minimal datasets."""
        print("Setting up minimal datasets...")
        
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
        
        def collate_fn(batch):
            return tldr_collate_fn(batch, self.tokenizer)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
    def setup_training(self):
        """Setup training components."""
        print("Setting up training components...")
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Simple scheduler for debug
        steps_per_epoch = len(self.train_loader)
        self.total_steps = steps_per_epoch * self.config.training.num_epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=self.total_steps,
            pct_start=0.1
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        print(f"Total training steps: {self.total_steps}")
        
    def init_wandb(self):
        """Initialize wandb."""
        wandb.init(
            project=self.config.logging.wandb_project,
            name=f"debug_{self.config.get_run_name()}",
            config=self.config.model_dump(),
            settings=wandb.Settings(console="wrap")
        )
        
    def _safe_float_conversion(self, value):
        """Safely convert numpy values to native Python float for wandb compatibility."""
        import numpy as np
        import math
        
        # Handle numpy types
        if isinstance(value, (np.floating, np.integer)):
            val = float(value.item())
        elif isinstance(value, np.ndarray):
            val = float(value.item())
        else:
            val = float(value)
        
        # Handle infinity and NaN values that might cause JSON issues
        if math.isinf(val):
            return 999999.0 if val > 0 else -999999.0
        elif math.isnan(val):
            return 0.0
        else:
            return val

    def upload_checkpoint_to_wandb(self, checkpoint_path, artifact_name, is_best=False):
        """Upload checkpoint to wandb - this is where the issue might occur."""
        try:
            print(f"Uploading checkpoint to wandb: {checkpoint_path.name}")
            
            # Create metadata - using safe conversion to avoid numpy serialization issues
            metadata = {
                "step": int(self.global_step),
                "epoch": int(self.current_epoch),
                "val_loss": self._safe_float_conversion(self.best_val_loss),
                "base_model_name": str(self.config.model.name),
            }
            
            # Add reward metrics if reward evaluation is enabled
            if self.config.logging.reward_evaluation and is_best:
                metadata["reward_improvement"] = self._safe_float_conversion(self.best_reward_improvement)
                metadata["best_criteria"] = "reward_improvement"
                description = f"Best LoRA adapter at step {self.global_step} (reward_improvement: {self.best_reward_improvement:.4f})"
            else:
                metadata["best_criteria"] = "val_loss"
                description = f"LoRA adapter at step {self.global_step} (val_loss: {self.best_val_loss:.4f})"
            
            print(f"Metadata: {metadata}")
            print(f"Description: {description}")
            
            # Check checkpoint file
            if checkpoint_path.exists():
                file_size = checkpoint_path.stat().st_size
                print(f"Checkpoint file size: {file_size / 1024 / 1024:.2f} MB")
            else:
                print("ERROR: Checkpoint file doesn't exist!")
                return False
            
            # Create artifact
            print("Creating wandb artifact...")
            artifact = wandb.Artifact(
                name=artifact_name,
                type="lora_adapter" if is_best else "lora_checkpoint",
                description=description,
                metadata=metadata
            )

            print("Adding file to artifact...")
            artifact.add_file(str(checkpoint_path))
            
            print("Logging artifact to wandb...")
            wandb.log_artifact(artifact, aliases=["latest", f"step-{self.global_step}"])
            
            print("Waiting for upload to complete...")
            artifact.wait()
            print("Upload completed!")
            
            return True

        except Exception as e:
            print(f"Failed to upload checkpoint: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_checkpoint(self, is_best=False):
        """Save checkpoint - minimal version."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'lora_state_dict': {k: v for k, v in self.model.state_dict().items() if 'lora_' in k},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.model_dump(),
            'base_model_name': self.config.model.name
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / "debug_best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Best model saved: {checkpoint_path.name}")
            
            # Upload to wandb - this is where the issue occurs
            if self.config.logging.use_wandb and self.config.logging.upload_checkpoints:
                success = self.upload_checkpoint_to_wandb(
                    checkpoint_path=checkpoint_path,
                    artifact_name=f"debug_best_model_step_{self.global_step}",
                    is_best=True
                )
                if not success:
                    print("CHECKPOINT UPLOAD FAILED - THIS IS THE BUG!")
        
    def evaluate(self):
        """Minimal evaluation."""
        print("Running minimal evaluation...")
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = output.loss
                
                total_loss += loss.item()
                num_batches += 1
                
                # Only evaluate 1 batch for debug
                if num_batches >= 1:
                    break
        
        # Get minimal examples for ROUGE
        _, references, predictions = get_examples(
            self.model, 
            self.tokenizer, 
            self.val_dataset, 
            self.device, 
            num_examples=1,  # Minimal for debug
            verbose=False
        )
        
        rouge_scores = get_rouge_scores(predictions, references)
        
        # Calculate reward model metrics if enabled
        reward_metrics = {}
        if self.config.logging.reward_evaluation:
            reward_metrics = self.evaluate_with_reward_model(num_samples=1)
        
        avg_loss = total_loss / max(num_batches, 1)
        eval_results = {'val_loss': avg_loss, 'rouge_scores': rouge_scores}
        eval_results.update(reward_metrics)
        
        return eval_results
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = output.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def train(self):
        """Debug training loop."""
        print("Starting debug training...")
        
        # Initialize wandb
        if self.config.logging.use_wandb:
            self.init_wandb()
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Only train for a few steps to trigger evaluation quickly
            max_steps = 5
            
            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue
                
                if batch_idx >= max_steps:
                    break
                
                metrics = self.train_step(batch)
                self.global_step += 1
                
                print(f"Step {self.global_step}: loss={metrics['loss']:.4f}")
                
                # Log to wandb
                if self.config.logging.use_wandb:
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/learning_rate': metrics['lr'],
                        'global_step': self.global_step
                    })
                
                # Trigger evaluation early
                if self.global_step % self.config.logging.eval_freq == 0:
                    eval_metrics = self.evaluate()
                    
                    # Log evaluation metrics
                    if self.config.logging.use_wandb:
                        log_dict = {
                            'eval/val_loss': eval_metrics['val_loss'],
                            'eval/rouge-L': eval_metrics['rouge_scores']['rougeL'],
                            'global_step': self.global_step
                        }
                        
                        # Add reward metrics if available
                        if self.config.logging.reward_evaluation and 'reward_improvement' in eval_metrics:
                            log_dict.update({
                                'eval/reward_improvement': self._safe_float_conversion(eval_metrics['reward_improvement']),
                                'eval/reward_finetuned_avg': self._safe_float_conversion(eval_metrics['reward_finetuned_avg']),
                                'eval/reward_baseline_avg': self._safe_float_conversion(eval_metrics['reward_baseline_avg'])
                            })
                        
                        wandb.log(log_dict)
                    
                    # Determine if this is the best model
                    if self.config.logging.reward_evaluation and 'reward_improvement' in eval_metrics:
                        is_best = eval_metrics['reward_improvement'] > self.best_reward_improvement
                        if is_best:
                            self.best_reward_improvement = float(eval_metrics['reward_improvement'])
                    else:
                        is_best = eval_metrics['val_loss'] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = float(eval_metrics['val_loss'])
                    
                    # Save checkpoint - this is where the wandb upload issue occurs
                    self.save_checkpoint(is_best=is_best)
                    
                    print(f"Evaluation complete - Best: {is_best}")
                    
                    # Break after first evaluation to test upload quickly
                    if self.global_step >= self.config.logging.eval_freq:
                        print("Debug training complete - stopping early")
                        break
        
        print("Debug training finished!")
        if self.config.logging.use_wandb:
            wandb.finish()

def main():
    """Main function for debug training."""
    parser = argparse.ArgumentParser(description='Debug Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--reward-evaluation', action='store_true', help='Enable reward evaluation')
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency')
    parser.add_argument('--max_eval_batches', type=int, default=1, help='Max evaluation batches')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Load configuration
    loader = ConfigLoader()
    config = loader.load_config(args.config)
    
    # Apply debug overrides
    config.logging.eval_freq = args.eval_freq
    config.logging.max_eval_batches = args.max_eval_batches
    config.training.num_epochs = args.num_epochs
    
    if args.reward_evaluation:
        config.logging.reward_evaluation = True
    
    # Generate debug run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.logging.run_name = f"debug_upload_{timestamp}"
    
    print(f"Debug config: eval_freq={config.logging.eval_freq}, "
          f"max_eval_batches={config.logging.max_eval_batches}, "
          f"reward_evaluation={config.logging.reward_evaluation}")
    
    # Create trainer and start debug training
    trainer = DebugSummarizationTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()