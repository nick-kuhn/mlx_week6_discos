"""Configuration validation using Pydantic built-in validation."""

import os
import shutil
import importlib.util
from pathlib import Path
from typing import List
from pydantic import ValidationError
from .config import TrainingConfiguration


class ConfigValidator:
    """Validates training configuration settings."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: TrainingConfiguration) -> bool:
        """Validate configuration and return True if valid."""
        self.errors = []
        self.warnings = []
        
        # Pydantic already handles most validation, but we do runtime checks
        self._validate_runtime_environment()
        self._validate_paths(config)
        self._validate_system_warnings(config)
        
        return len(self.errors) == 0
    
    def get_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get list of validation warnings."""
        return self.warnings.copy()
    
    def print_validation_results(self):
        """Print validation results to console."""
        if self.errors:
            print("❌ Configuration Validation Errors:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print("⚠️ Configuration Validation Warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ Configuration validation passed successfully!")
    
    def _validate_runtime_environment(self):
        """Validate runtime environment requirements."""
        import torch
        
        # Check PyTorch installation
        if not torch.cuda.is_available():
            self.warnings.append("CUDA not available, training will be slow on CPU")
        
        # Check required packages
        required_packages = ['transformers', 'accelerate', 'peft', 'wandb', 'tqdm', 'numpy', 'rouge_score']
        missing_packages = []
        
        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        
        if missing_packages:
            self.errors.append(f"Missing required packages: {', '.join(missing_packages)}")
        
        # Check disk space
        try:
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            if free_space_gb < 10:
                self.warnings.append(f"Low disk space: {free_space_gb:.1f}GB available")
        except Exception:
            pass  # Ignore if we can't check disk space
    
    def _validate_paths(self, config: TrainingConfiguration):
        """Validate file paths and directories."""
        # Validate checkpoint directory
        try:
            checkpoint_path = Path(config.model.checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.errors.append(f"Cannot create checkpoint directory '{config.model.checkpoint_dir}': {e}")
        
        # Validate resume checkpoint
        if config.resume.resume_from_checkpoint:
            checkpoint_path = Path(config.resume.resume_from_checkpoint)
            if not checkpoint_path.exists():
                self.errors.append(f"Resume checkpoint file not found: {config.resume.resume_from_checkpoint}")
        
        # Check for local data paths
        for path_name, path_value in [('train_path', config.data.train_path), 
                                      ('val_path', config.data.val_path)]:
            if path_value and not path_value.startswith('CarperAI/') and not path_value.startswith('openai/'):
                local_path = Path(path_value)
                if not local_path.exists():
                    self.warnings.append(f"Local data path '{path_value}' does not exist")
    
    def _validate_system_warnings(self, config: TrainingConfiguration):
        """Generate system-specific warnings."""
        # Windows-specific warnings
        if os.name == 'nt' and config.data.num_workers > 0:
            self.warnings.append("num_workers > 0 may cause issues on Windows")
        
        # Wandb warnings
        if config.logging.use_wandb:
            if not os.getenv('WANDB_API_KEY'):
                self.warnings.append("WANDB_API_KEY environment variable not set")
        
        # Performance warnings
        effective_batch_size = config.training.batch_size * config.training.gradient_accumulation_steps
        if effective_batch_size > 64:
            self.warnings.append(f"Effective batch size ({effective_batch_size}) is very large")
        
        if config.training.learning_rate > 1e-2:
            self.warnings.append(f"Learning rate {config.training.learning_rate} is unusually high")
        
        # Logic warnings
        if config.logging.eval_freq < config.logging.log_freq:
            self.warnings.append("Eval frequency is less than log frequency")
        
        if config.logging.reward_evaluation and not config.logging.use_wandb:
            self.warnings.append("Reward evaluation is enabled but wandb logging is disabled")