"""Configuration loader for YAML files with Pydantic."""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, List
from pydantic import ValidationError
from .config import TrainingConfiguration


class ConfigLoader:
    """Loads and processes configuration files."""
    
    def __init__(self):
        self.env_var_pattern = re.compile(r'\$\{([^}]+)\}')
    
    def load_config(self, config_path: str) -> TrainingConfiguration:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")
        
        if config_data is None:
            config_data = {}
        
        # Process environment variables
        config_data = self._process_env_vars(config_data)
        
        # Use Pydantic to parse and validate
        try:
            config = TrainingConfiguration.parse_obj(config_data)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
        
        return config
    
    def load_multiple_configs(self, config_paths: List[str]) -> TrainingConfiguration:
        """Load multiple configuration files and merge them."""
        if not config_paths:
            return TrainingConfiguration()
        
        # Load first config as base
        merged_config = self.load_config(config_paths[0])
        
        # Merge additional configs
        for config_path in config_paths[1:]:
            overlay_config = self.load_config(config_path)
            merged_config = self._merge_configs(merged_config, overlay_config)
        
        return merged_config
    
    def _process_env_vars(self, data: Any) -> Any:
        """Process environment variables in configuration data."""
        if isinstance(data, dict):
            return {k: self._process_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_env_vars(item) for item in data]
        elif isinstance(data, str):
            return self._substitute_env_vars(data)
        else:
            return data
    
    def _substitute_env_vars(self, value: str) -> str:
        """Substitute environment variables in string values."""
        def replace_env_var(match):
            var_name = match.group(1)
            # Support default values: ${VAR_NAME:default_value}
            if ':' in var_name:
                var_name, default_value = var_name.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                env_value = os.getenv(var_name)
                if env_value is None:
                    raise ValueError(f"Environment variable '{var_name}' not found")
                return env_value
        
        return self.env_var_pattern.sub(replace_env_var, value)
    
    def _merge_configs(self, base: TrainingConfiguration, overlay: TrainingConfiguration) -> TrainingConfiguration:
        """Merge two configurations, with overlay taking precedence."""
        # Convert to dictionaries, merge, then back to Pydantic model
        base_dict = base.model_dump()
        overlay_dict = overlay.model_dump()
        
        merged_dict = self._deep_merge_dicts(base_dict, overlay_dict)
        
        return TrainingConfiguration.parse_obj(merged_dict)
    
    def _deep_merge_dicts(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: TrainingConfiguration, output_path: str):
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.model_dump()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving configuration to {output_path}: {e}")
    
    def create_config_from_args(self, args) -> TrainingConfiguration:
        """Create configuration from argparse arguments (for backward compatibility)."""
        config_dict = {}
        
        # Helper function to add non-None values
        def add_if_not_none(dict_ref, key, value):
            if value is not None:
                dict_ref[key] = value
        
        # Model config
        model_config = {}
        add_if_not_none(model_config, 'name', args.model_name)
        add_if_not_none(model_config, 'checkpoint_dir', args.checkpoint_dir)
        if model_config:
            config_dict['model'] = model_config
        
        # Training config  
        training_config = {}
        add_if_not_none(training_config, 'num_epochs', args.num_epochs)
        add_if_not_none(training_config, 'batch_size', args.batch_size)
        add_if_not_none(training_config, 'eval_batch_size', args.eval_batch_size)
        add_if_not_none(training_config, 'gradient_accumulation_steps', args.gradient_accumulation_steps)
        add_if_not_none(training_config, 'learning_rate', args.learning_rate)
        add_if_not_none(training_config, 'weight_decay', args.weight_decay)
        add_if_not_none(training_config, 'max_grad_norm', args.max_grad_norm)
        add_if_not_none(training_config, 'lr_scheduler', args.lr_scheduler)
        
        # Handle boolean flags with negative counterparts
        if args.use_amp:
            training_config['use_amp'] = True
        elif getattr(args, 'no_use_amp', False):
            training_config['use_amp'] = False
            
        if training_config:
            config_dict['training'] = training_config
        
        # Data config
        data_config = {}
        add_if_not_none(data_config, 'train_path', args.train_path)
        add_if_not_none(data_config, 'val_path', args.val_path)
        add_if_not_none(data_config, 'num_workers', args.num_workers)
        if data_config:
            config_dict['data'] = data_config
        
        # Logging config
        logging_config = {}
        add_if_not_none(logging_config, 'wandb_project', args.wandb_project)
        add_if_not_none(logging_config, 'run_name', args.run_name)
        add_if_not_none(logging_config, 'log_freq', args.log_freq)
        add_if_not_none(logging_config, 'eval_freq', args.eval_freq)
        add_if_not_none(logging_config, 'max_eval_batches', args.max_eval_batches)
        add_if_not_none(logging_config, 'checkpoint_upload_freq', args.checkpoint_upload_freq)
        
        # Handle boolean flags
        if args.use_wandb:
            logging_config['use_wandb'] = True
        elif getattr(args, 'no_use_wandb', False):
            logging_config['use_wandb'] = False
            
        if args.upload_checkpoints:
            logging_config['upload_checkpoints'] = True
        elif getattr(args, 'no_upload_checkpoints', False):
            logging_config['upload_checkpoints'] = False
            
        if args.delete_after_upload:
            logging_config['delete_after_upload'] = True
        elif getattr(args, 'no_delete_after_upload', False):
            logging_config['delete_after_upload'] = False
            
        if getattr(args, 'verbose_evals', False):
            logging_config['verbose_evals'] = True
        if getattr(args, 'reward_evaluation', False):
            logging_config['reward_evaluation'] = True
            
        if logging_config:
            config_dict['logging'] = logging_config
        
        # Resume config
        resume_config = {}
        add_if_not_none(resume_config, 'resume_from_checkpoint', getattr(args, 'resume_from_checkpoint', None))
        if resume_config:
            config_dict['resume'] = resume_config
        
        # Let Pydantic handle defaults for missing values
        return TrainingConfiguration.parse_obj(config_dict)
    
    def apply_overrides(self, config: TrainingConfiguration, overrides: Dict[str, Any]) -> TrainingConfiguration:
        """Apply parameter overrides to configuration."""
        config_dict = config.model_dump()
        
        for key, value in overrides.items():
            # Support dot notation for nested parameters
            keys = key.split('.')
            current = config_dict
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the final value with type conversion
            final_key = keys[-1]
            current[final_key] = self._convert_override_value(value)
        
        return TrainingConfiguration.parse_obj(config_dict)
    
    def _convert_override_value(self, value: str) -> Any:
        """Convert string override value to appropriate type."""
        if isinstance(value, str):
            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            elif value.isdigit():
                return int(value)
            elif value.replace('.', '', 1).isdigit():
                return float(value)
            elif value.lower() == 'none':
                return None
        
        return value