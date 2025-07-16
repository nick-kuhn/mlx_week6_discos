"""Configuration classes for summarization training using Pydantic."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen3-0.6B-Base"
    checkpoint_dir: str = "checkpoints"


class TrainingConfig(BaseModel):
    num_epochs: int = 3
    batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: Literal["cosine", "linear"] = "cosine"
    use_amp: bool = True


class DataConfig(BaseModel):
    train_path: str = "CarperAI/openai_summarize_tldr"
    val_path: str = "CarperAI/openai_summarize_tldr"
    num_workers: int = 0


class LoggingConfig(BaseModel):
    use_wandb: bool = True
    wandb_project: str = "summarization-finetuning"
    run_name: Optional[str] = None
    log_freq: int = 50
    eval_freq: int = 500
    max_eval_batches: int = 100
    upload_checkpoints: bool = True
    checkpoint_upload_freq: int = 2000
    delete_after_upload: bool = True
    verbose_evals: bool = False
    reward_evaluation: bool = False


class ResumeConfig(BaseModel):
    resume_from_checkpoint: Optional[str] = None


class LoraConfig(BaseModel):
    r: int = 8
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"


class MixedPrecisionConfig(BaseModel):
    enabled: bool = True
    dtype: str = "fp16"


class MemoryConfig(BaseModel):
    gradient_checkpointing: bool = False
    cleanup_frequency: int = 20
    tensor_cache_cleanup_freq: int = 500


class GenerationConfig(BaseModel):
    max_new_tokens: int = 100
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class AdvancedConfig(BaseModel):
    lora: LoraConfig = LoraConfig()
    mixed_precision: MixedPrecisionConfig = MixedPrecisionConfig()
    memory: MemoryConfig = MemoryConfig()
    generation: GenerationConfig = GenerationConfig()


class TrainingConfiguration(BaseModel):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
    resume: ResumeConfig = ResumeConfig()
    advanced: AdvancedConfig = AdvancedConfig()
    
    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory as Path object."""
        return Path(self.model.checkpoint_dir)
    
    def get_run_name(self) -> str:
        """Get run name, generating one if not provided."""
        if self.logging.run_name is not None:
            return self.logging.run_name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"qwen_summarization_{timestamp}"
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("🎯 Training Configuration:")
        print(f"   Model: {self.model.name}")
        print(f"   Epochs: {self.training.num_epochs}")
        print(f"   Batch Size: {self.training.batch_size}")
        print(f"   Learning Rate: {self.training.learning_rate}")
        print(f"   Use AMP: {self.training.use_amp}")
        print(f"   Use Wandb: {self.logging.use_wandb}")
        print(f"   Project: {self.logging.wandb_project}")
        print(f"   Run Name: {self.get_run_name()}")
        print(f"   Checkpoint Dir: {self.model.checkpoint_dir}")
        if self.resume.resume_from_checkpoint:
            print(f"   Resume From: {self.resume.resume_from_checkpoint}")
        if self.logging.reward_evaluation:
            print(f"   Reward Evaluation: Enabled")
        print()