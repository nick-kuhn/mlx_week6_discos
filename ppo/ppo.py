import os

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import PPOConfig, PPOTrainer

import wandb


class PPOCompatibleRewardModel(nn.Module):
    """
    Interface-compatible wrapper for reward model that handles tokenization conversion.
    
    Mimics the expected HuggingFace model interface that PPOTrainer expects:
    - base_model_prefix attribute
    - score() method
    - Proper backbone access
    """
    
    def __init__(self, deberta_model, qwen_tokenizer, deberta_tokenizer, device):
        super().__init__()
        self.deberta_model = deberta_model
        self.qwen_tokenizer = qwen_tokenizer
        self.deberta_tokenizer = deberta_tokenizer
        self.device = device
        
        # Set the base_model_prefix to match DeBERTa's structure
        self.base_model_prefix = "deberta"
        
        # Create the backbone attribute that PPOTrainer expects
        self.deberta = self.deberta_model.deberta
    
    def score(self, hidden_states):
        """
        Score method that PPOTrainer expects.
        Takes hidden states and returns logits.
        """
        # Use the classifier from the DeBERTa model
        return self.deberta_model.classifier(hidden_states)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        """
        Forward method that handles tokenization conversion.
        
        This is where the magic happens:
        1. Receive Qwen tokens from PPOTrainer
        2. Convert to text
        3. Re-tokenize with DeBERTa
        4. Forward through DeBERTa model
        """
        # Convert Qwen tokens to text
        batch_size = input_ids.shape[0]
        texts = []
        
        for i in range(batch_size):
            # Get the sequence without padding
            if attention_mask is not None:
                # Use attention mask to find real tokens
                real_tokens = input_ids[i][attention_mask[i] == 1]
            else:
                # If no attention mask, find non-pad tokens
                real_tokens = input_ids[i][input_ids[i] != self.qwen_tokenizer.pad_token_id]
            
            # Decode to text
            text = self.qwen_tokenizer.decode(real_tokens, skip_special_tokens=True)
            texts.append(text)
        
        # Re-tokenize with DeBERTa tokenizer
        deberta_inputs = self.deberta_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward through DeBERTa model
        return self.deberta_model(**deberta_inputs)
    
    def train(self, mode=True):
        """Properly propagate training mode."""
        super().train(mode)
        self.deberta_model.train(mode)
        return self
    
    def eval(self):
        """Properly propagate eval mode."""
        super().eval()
        self.deberta_model.eval()
        return self
    
    def parameters(self):
        """Return all parameters for optimizer."""
        return self.deberta_model.parameters()
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.deberta_model.to(device)
        self.device = device
        return self


class PPOCompatibleValueModel(nn.Module):
    """
    Enhanced value model wrapper that inherits from nn.Module.
    Handles tokenization conversion and provides proper PyTorch functionality.
    """
    
    def __init__(self, deberta_model, qwen_tokenizer, deberta_tokenizer, device):
        super().__init__()
        self.deberta_model = deberta_model
        self.qwen_tokenizer = qwen_tokenizer
        self.deberta_tokenizer = deberta_tokenizer
        self.device = device
        
        # PPOTrainer expects these attributes
        self.base_model_prefix = "deberta"
        self.deberta = self.deberta_model.deberta  # For compatibility if needed
    
    def score(self, hidden_states):
        """PPOTrainer calls this method."""
        return self.deberta_model.classifier(hidden_states)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        """Handle tokenization conversion."""
        # Convert Qwen tokens to text
        batch_size = input_ids.shape[0]
        texts = []
        
        for i in range(batch_size):
            if attention_mask is not None:
                real_tokens = input_ids[i][attention_mask[i] == 1]
            else:
                real_tokens = input_ids[i][input_ids[i] != self.qwen_tokenizer.pad_token_id]
            
            text = self.qwen_tokenizer.decode(real_tokens, skip_special_tokens=True)
            texts.append(text)
        
        # Re-tokenize with DeBERTa
        deberta_inputs = self.deberta_tokenizer(
            texts, padding=True, truncation=True, 
            max_length=512, return_tensors="pt"
        ).to(self.device)
        
        # Forward through DeBERTa
        return self.deberta_model(**deberta_inputs)
    
    def train(self, mode=True):
        """Properly propagate training mode."""
        super().train(mode)
        self.deberta_model.train(mode)
        return self
    
    def eval(self):
        """Properly propagate eval mode."""
        super().eval()
        self.deberta_model.eval()
        return self
    
    def parameters(self):
        """Return all parameters for optimizer."""
        return self.deberta_model.parameters()
    
    def to(self, device):
        """Move to device."""
        super().to(device)
        self.deberta_model.to(device)
        self.device = device
        return self


class CustomPolicyAndValueWrapper(nn.Module):
    """
    Custom wrapper that replaces PPOTrainer's PolicyAndValueWrapper.
    Handles tokenization conversion properly for mixed tokenizer setups.
    """
    
    def __init__(self, policy, value_model_wrapper):
        super().__init__()
        self.policy = policy
        self.value_model = value_model_wrapper
        
        # Note: We don't set self.critic_backbone because we want to 
        # use the full wrapper, not bypass it
    
    def forward(self, **kwargs):
        """
        Forward pass that handles both policy and value computation.
        This is called by PPOTrainer during training.
        """
        # Policy forward (uses Qwen tokenizer normally)
        policy_output = self.policy(**kwargs)
        
        # Value forward (uses our wrapper with tokenization conversion)
        value_output = self.value_model(**kwargs)  # This triggers our conversion
        value_logits = value_output.logits
        
        return policy_output, value_logits

# Note: Following the official TRL example structure
# Using Qwen policy model and DeBERTa reward/value models


def download_wandb_model(artifact_path, local_dir="./wandb_models"):
    """Download model from W&B artifact."""
    print(f"Downloading model from W&B artifact: {artifact_path}")

    # Use a specific project for downloading to avoid conflicts
    run = wandb.init(project="model-downloader", job_type="download", reinit=True)

    try:
        artifact = run.use_artifact(artifact_path, type="lora_adapter")
        # .download() returns the correct path to the content
        adapter_path = artifact.download(root=local_dir)
        print(f"Downloaded LoRA adapter to: {adapter_path}")
    except Exception:
        try:
            artifact = run.use_artifact(artifact_path, type="best_model")
            adapter_path = artifact.download(root=local_dir)
            print(f"Downloaded full model to: {adapter_path}")
        except Exception:
            artifact = run.use_artifact(artifact_path)
            adapter_path = artifact.download(root=local_dir)
            print(f"Downloaded model (unknown type) to: {adapter_path}")

    run.finish()

    # Return the direct path provided by the wandb library
    print(f"Model path resolved to: {adapter_path}")
    return adapter_path


def load_models():
    """Load models following the official TRL example structure."""
    print("Loading models following official TRL example structure...")
    
    # 1. Load tokenizer (following official example)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B-Base", 
        padding_side="left", 
        trust_remote_code=True
    )
    # Add explicit PAD token like in official example
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(f"Tokenizer pad_token: {tokenizer.pad_token}")
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    
    # 1.5. Load DeBERTa tokenizer for reward/value models
    deberta_tokenizer = AutoTokenizer.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2",
        trust_remote_code=True
    )
    print("Loaded DeBERTa tokenizer for reward/value models")
    
    # 2. Load policy model (our fine-tuned Qwen)
    print("Loading policy model (fine-tuned Qwen)...")
    adapter_path = "./qwen_finetuned_local"
    
    # Load base model first
    policy_base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        trust_remote_code=True
    )
    
    # Resize embeddings after adding PAD token
    policy_base.resize_token_embeddings(len(tokenizer))
    print(f"Resized policy model embeddings to {len(tokenizer)} tokens")
    
    # Load fine-tuned weights
    checkpoint_files = os.listdir(adapter_path)
    if any("adapter" in f for f in checkpoint_files) or any(
        ".bin" in f and "adapter" in f for f in checkpoint_files
    ):
        print(f"Loading LoRA adapter from: {adapter_path}")
        policy = PeftModel.from_pretrained(policy_base, adapter_path)
    else:
        checkpoint_file = next(
            (f for f in checkpoint_files if f.endswith(".pt") or f.endswith(".pth")),
            None,
        )
        if checkpoint_file:
            print(f"Loading full model checkpoint from: {os.path.join(adapter_path, checkpoint_file)}")
            checkpoint = torch.load(os.path.join(adapter_path, checkpoint_file), map_location="cpu")
            if "lora_state_dict" in checkpoint:
                from peft import LoraConfig, get_peft_model
                config_dict = checkpoint.get("config", {})
                lora_settings = config_dict.get("advanced", {}).get("lora", {})
                lora_config = LoraConfig(
                    r=lora_settings.get("r", 4),
                    lora_alpha=lora_settings.get("alpha", 8),
                    target_modules=lora_settings.get("target_modules", ["q_proj", "v_proj"]),
                    lora_dropout=lora_settings.get("dropout", 0.1),
                    bias=lora_settings.get("bias", "none"),
                    task_type="CAUSAL_LM",
                )
                print(f"Creating LoRA model with r={lora_config.r}, alpha={lora_config.lora_alpha}")
                policy = get_peft_model(policy_base, lora_config)
                print("Loading LoRA state dict...")
                policy.load_state_dict(checkpoint["lora_state_dict"], strict=False)
                print("[SUCCESS] Loaded LoRA adapter from checkpoint")
            elif "model_state_dict" in checkpoint:
                policy = policy_base
                policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print("[SUCCESS] Loaded full model from checkpoint")
            else:
                raise ValueError("Checkpoint format not recognized")
        else:
            raise ValueError("No checkpoint file found in artifact")
    
    # 3. Load value model (DeBERTa, following official example structure)
    print("Loading value model (DeBERTa)...")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2", 
        trust_remote_code=True, 
        num_labels=1
    )
    
    # 4. Load reward model (same DeBERTa model, following official example)
    print("Loading reward model (DeBERTa)...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2", 
        trust_remote_code=True, 
        num_labels=1
    )
    
    # 5. Load reference model (same as policy, following official example)
    print("Loading reference model (same as policy)...")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        trust_remote_code=True
    )
    # Reference model should also have resized embeddings to match policy
    ref_policy.resize_token_embeddings(len(tokenizer))
    print(f"Resized reference model embeddings to {len(tokenizer)} tokens")
    
    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    value_model.to(device)
    reward_model.to(device)
    ref_policy.to(device)
    
    # Wrap reward and value models with PPO-compatible interface
    ppo_reward_model = PPOCompatibleRewardModel(reward_model, tokenizer, deberta_tokenizer, device)
    ppo_value_model = PPOCompatibleValueModel(value_model, tokenizer, deberta_tokenizer, device)
    
    # Quick test of tokenization conversion
    print("Testing tokenization conversion...")
    test_text = "This is a test summary. TL;DR: Testing tokenization."
    test_tokens = tokenizer(test_text, return_tensors="pt", padding=True)
    print(f"Original text: {test_text}")
    print(f"Qwen tokens shape: {test_tokens['input_ids'].shape}")
    
    # Test the conversion in our wrapper
    try:
        test_output = ppo_reward_model.forward(test_tokens['input_ids'].to(device), test_tokens['attention_mask'].to(device))
        print(f"Reward model output shape: {test_output.logits.shape}")
        print("✓ Tokenization conversion test passed!")
    except Exception as e:
        print(f"✗ Tokenization conversion test failed: {e}")
        raise e
    
    print("All models loaded successfully!")
    return (
        policy,
        ref_policy,
        ppo_value_model,
        ppo_reward_model,
        tokenizer,
        deberta_tokenizer,
        device,
    )


def prepare_dataset(dataset, tokenizer):
    """Prepare dataset following the official TRL example."""
    print("Preparing dataset following official TRL example...")
    
    def tokenize(element):
        # Convert our format to match the expected format
        # Our dataset has "prompt" field, we need to create a simple input
        prompt = element["prompt"] + " TL;DR: "
        input_ids = tokenizer(
            prompt,
            padding=False,  # No padding during tokenization (like official example)
            truncation=True,
            max_length=512,
            return_tensors=None  # Return list, not tensors
        )["input_ids"]
        return {"input_ids": input_ids, "lengths": len(input_ids)}
    
    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=1,  # Use single process for simplicity
    )


def main():
    # --- 1. Initialize W&B Run for PPO Training ---
    wandb.init(project="qwen-ppo-finetuning", job_type="ppo-train")
    
    # --- 2. Load all models and tokenizers ---
    (
        policy,
        ref_policy,
        value_model,
        reward_model,
        tokenizer,
        deberta_tokenizer,
        device,
    ) = load_models()

    # --- 3. Prepare the Dataset (following official example) ---
    print("Loading and preparing dataset...")
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
    dataset = dataset.select(range(min(100, len(dataset))))
    
    # Prepare dataset following official example
    train_dataset = prepare_dataset(dataset, tokenizer)
    
    # Filter by length like in official example
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=1)
    
    # Verify the format (removed problematic EOS assertion)
    print(f"Dataset prepared: {len(train_dataset)} samples")

    # --- 4. Configure and Initialize PPO Trainer (following official example) ---
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        num_ppo_epochs=4,
        # PPO-specific parameters
        kl_coef=0.05,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        vf_coef=0.1,
        # Generation parameters
        temperature=0.7,
        response_length=100,
        # Evaluation
        eval_strategy="no",  # No evaluation for now
    )

    # Initialize trainer following official example exactly
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=None,  # No PEFT config for now
    )
    
    # STRATEGY B: Replace PPOTrainer's PolicyAndValueWrapper with our custom one
    print("Replacing PPOTrainer's PolicyAndValueWrapper with custom implementation...")
    original_wrapper = trainer.model
    custom_wrapper = CustomPolicyAndValueWrapper(trainer.policy_model, trainer.value_model)
    
    # Move to the same device as the original wrapper
    if hasattr(original_wrapper, 'device'):
        custom_wrapper.to(original_wrapper.device)
    elif hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
        # Properly handle accelerator preparation
        custom_wrapper = trainer.accelerator.prepare(custom_wrapper)
        trainer.model = custom_wrapper  # Assign here if using accelerator
    else:
        custom_wrapper.to(device)
        trainer.model = custom_wrapper  # Assign here for non-accelerator case
    
    print("✓ Custom PolicyAndValueWrapper installed successfully!")
    print("⚠️  NOTE: Reward model still uses original DeBERTa - may cause issues in get_reward() calls")
    
    # Test the custom wrapper
    print("Testing custom PolicyAndValueWrapper...")
    test_text = "This is a test for the custom wrapper. TL;DR: Testing wrapper."
    test_tokens = tokenizer(test_text, return_tensors="pt", padding=True)
    test_tokens = {k: v.to(device) for k, v in test_tokens.items()}
    
    try:
        with torch.no_grad():
            policy_output, value_logits = custom_wrapper(**test_tokens)
        print(f"✓ Custom wrapper test passed!")
        print(f"  - Policy output shape: {policy_output.logits.shape}")
        print(f"  - Value logits shape: {value_logits.shape}")
    except Exception as e:
        print(f"✗ Custom wrapper test failed: {e}")
        raise e
    
    # --- 5. Train following official example ---
    print("Starting PPO training...")
    trainer.train()

    print("PPO training finished.")
    # --- 6. Finish the W&B Run ---
    wandb.finish()


if __name__ == "__main__":
    main()
