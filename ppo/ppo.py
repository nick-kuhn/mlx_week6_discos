import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from tqdm import tqdm
from trl import PPOConfig, PPOTrainer
import wandb

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
train_path="CarperAI/openai_summarize_tldr"
val_path="CarperAI/openai_summarize_tldr"


def download_wandb_model(artifact_path, local_dir="./wandb_models"):
    """Download model from W&B artifact."""
    print(f"Downloading model from W&B artifact: {artifact_path}")

    wandb.init(project="reward-model-evaluation", job_type="download")

    try:
        artifact = wandb.use_artifact(artifact_path, type="lora_adapter")
        artifact_dir = artifact.download(root=local_dir)
        print(f"Downloaded LoRA adapter from: {artifact_dir}")
    except:
        try:
            artifact = wandb.use_artifact(artifact_path, type="best_model")
            artifact_dir = artifact.download(root=local_dir)
            print(f"Downloaded full model from: {artifact_dir}")
        except:
            artifact = wandb.use_artifact(artifact_path)
            artifact_dir = artifact.download(root=local_dir)
            print(f"Downloaded model (unknown type) from: {artifact_dir}")

    wandb.finish()

    print(f"Model downloaded to: {artifact_dir}")
    return artifact_dir


def load_models():
    # 1. Load base QWEN model and tokenizer
    print("Loading base Qwen model and tokenizer...")
    base_model_name = "Qwen/Qwen3-0.6B-Base"
    qwen_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    qwen_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Set pad token if not present
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id

    # 2. Load fine-tuned model from W&B artifact
    print("Loading fine-tuned model from W&B...")
    artifact_path = "ntkuhn/summarization-finetuning/best_finetuned_model:v13"
    adapter_path = download_wandb_model(artifact_path)

    # Create a separate finetuned model instance from the base model
    finetuned_model_base = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Resize token embeddings to match training setup
    finetuned_model_base.resize_token_embeddings(len(qwen_tokenizer))

    # Check if this is a LoRA adapter or full model checkpoint
    checkpoint_files = os.listdir(adapter_path)
    if any("adapter" in f for f in checkpoint_files) or any(
        ".bin" in f and "adapter" in f for f in checkpoint_files
    ):
        # This is a LoRA adapter
        print(f"Loading LoRA adapter from: {adapter_path}")
        finetuned_model = PeftModel.from_pretrained(finetuned_model_base, adapter_path)
    else:
        # This is likely a full model checkpoint - load it directly
        checkpoint_file = next(
            (f for f in checkpoint_files if f.endswith(".pt") or f.endswith(".pth")),
            None,
        )
        if checkpoint_file:
            print(
                f"Loading full model checkpoint from: {os.path.join(adapter_path, checkpoint_file)}"
            )
            checkpoint = torch.load(
                os.path.join(adapter_path, checkpoint_file), map_location="cpu"
            )
            if "lora_state_dict" in checkpoint:
                from peft import LoraConfig, get_peft_model

                # Extract LoRA config from checkpoint if available, otherwise use defaults that match training
                config_dict = checkpoint.get('config', {})
                lora_settings = config_dict.get('advanced', {}).get('lora', {})
                
                lora_config = LoraConfig(
                    r=lora_settings.get('r', 4),  # Default to 4 to match finetune_default.yaml
                    lora_alpha=lora_settings.get('alpha', 8),  # Default to 8 to match finetune_default.yaml
                    target_modules=lora_settings.get('target_modules', ["q_proj", "v_proj"]),
                    lora_dropout=lora_settings.get('dropout', 0.1),
                    bias=lora_settings.get('bias', "none"),
                    task_type="CAUSAL_LM",
                )
                print(f"Creating LoRA model with r={lora_config.r}, alpha={lora_config.lora_alpha}")
                finetuned_model = get_peft_model(finetuned_model_base, lora_config)
                
                # Load only the LoRA weights, not the full model state
                print("Loading LoRA state dict...")
                try:
                    finetuned_model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
                    print("[SUCCESS] LoRA weights loaded successfully")
                except Exception as e:
                    print(f"Error loading LoRA state dict: {e}")
                    # Try to load only the LoRA parameters
                    lora_state_dict = {k: v for k, v in checkpoint["lora_state_dict"].items() if "lora_" in k}
                    finetuned_model.load_state_dict(lora_state_dict, strict=False)
                    print("[SUCCESS] LoRA weights loaded with filtered state dict")
                print("[SUCCESS] Loaded LoRA adapter from checkpoint")
            elif "model_state_dict" in checkpoint:
                finetuned_model = finetuned_model_base
                finetuned_model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
                print(
                    "[SUCCESS] Loaded full model from checkpoint (with vocab size adjustment)"
                )
            else:
                raise ValueError("Checkpoint format not recognized")
        else:
            raise ValueError("No checkpoint file found in artifact")

    # 3. Load the pre-built reward model and its specific tokenizer
    print("Loading reward model and tokenizer...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen_model.to(device)
    finetuned_model.to(device)
    reward_model.to(device)

    print("All models loaded successfully!")
    return (
        qwen_model,
        finetuned_model,
        qwen_tokenizer,
        reward_model,
        reward_tokenizer,
        device,
    )
def main():
    # --- 1. Initialize W&B Run for PPO Training ---
    wandb.init(
        project="qwen-ppo-finetuning",
        job_type="ppo-train"
    )
    
    # --- 2. Load all models and tokenizers ---
    (
        base_model,
        finetuned_model,
        qwen_tokenizer,
        reward_model,
        reward_tokenizer,
        device,
    ) = load_models()
    ppo_model = finetuned_model
    
    # --- 3. Prepare the Dataset ---
    def tokenize_function(examples):
        prompts = [p + " TL;DR: " for p in examples["prompt"]]
        tokenized_output = qwen_tokenizer(prompts, padding=False, truncation=True)
        tokenized_output["query"] = prompts
        return tokenized_output
    
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
    dataset = dataset.select(range(min(100, len(dataset))))
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch")
    
    # --- 4. Configure and Initialize PPO Trainer ---
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=2,
        bf16=False,
        fp16=False,
    )
    
    # Remove dataset and tokenizer parameters from PPOTrainer constructor
    # Filter out parameters that PPOTrainer doesn't accept
    ppo_trainer_kwargs = {
        k: v for k, v in ppo_config.to_dict().items() 
        if k not in ['output_dir', 'logging_dir', 'save_strategy', 'save_steps', 'evaluation_strategy', 'eval_steps']
    }
    
    ppo_trainer = PPOTrainer(
        model=ppo_model,
        ref_model=None,
        **ppo_trainer_kwargs,
    )
    
    # Create dataloader manually from dataset
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Use hardcoded batch_size since we removed ppo_config
    
    # --- 5. PPO Training Loop ---
    generation_kwargs = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": qwen_tokenizer.pad_token_id,
    }
    
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        query_tensors = batch["input_ids"]
        
        # Generate responses by calling .generate() on the model, not the trainer
        response_tensors = ppo_model.generate(query_tensors, **generation_kwargs)
        batch["response"] = qwen_tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Compute reward
        texts_for_reward = [q + r for q, r in zip(batch["query"], batch["response"])]
        reward_inputs = reward_tokenizer(texts_for_reward, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            reward_logits = reward_model(**reward_inputs).logits
            reward_scores = [logit for logit in reward_logits]
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_scores)
        ppo_trainer.log_stats(stats, batch, reward_scores)
        print(f"Step {step}: Mean Reward = {stats['ppo/returns/mean']:.2f}")
    
    # --- 6. Finish the W&B Run ---
    wandb.finish()

if __name__ == "__main__":
    main()
    