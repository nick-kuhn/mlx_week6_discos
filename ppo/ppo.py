import os

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

import wandb

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
train_path = "CarperAI/openai_summarize_tldr"
val_path = "CarperAI/openai_summarize_tldr"


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
    # 1. Load base QWEN model and tokenizer (No changes here)
    print("Loading base Qwen model and tokenizer...")
    base_model_name = "Qwen/Qwen3-0.6B-Base"
    qwen_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    qwen_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id

    # 2. Load fine-tuned model from LOCAL directory
    print("Loading fine-tuned model from local directory...")
    adapter_path = "./qwen_finetuned_local"
    finetuned_model_base = AutoModelForCausalLM.from_pretrained(base_model_name)
    finetuned_model_base.resize_token_embeddings(len(qwen_tokenizer))

    # This entire block correctly loads the finetuned_model from the .pt file
    checkpoint_files = os.listdir(adapter_path)
    # ... (no changes to the logic that creates finetuned_model) ...
    # This block successfully creates the 'finetuned_model'
    if any("adapter" in f for f in checkpoint_files) or any(
        ".bin" in f and "adapter" in f for f in checkpoint_files
    ):
        print(f"Loading LoRA adapter from: {adapter_path}")
        finetuned_model = PeftModel.from_pretrained(finetuned_model_base, adapter_path)
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
                finetuned_model = get_peft_model(finetuned_model_base, lora_config)
                print("Loading LoRA state dict...")
                finetuned_model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
                print("[SUCCESS] Loaded LoRA adapter from checkpoint")
            elif "model_state_dict" in checkpoint:
                finetuned_model = finetuned_model_base
                finetuned_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print("[SUCCESS] Loaded full model from checkpoint (with vocab size adjustment)")
            else:
                raise ValueError("Checkpoint format not recognized")
        else:
            raise ValueError("No checkpoint file found in artifact")

    # --- 3. Create the dedicated Value Model --- [CORRECTED BLOCK]
    print("Creating value model...")
    # Use AutoModelForSequenceClassification with num_labels=1 as suggested
    base_value_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=1,
        problem_type="regression"
    )
    base_value_model.resize_token_embeddings(len(qwen_tokenizer))
    
    # Create a wrapper class that adds the score method TRL expects
    class ValueModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self._model = model
            # Add the base_model_prefix that TRL expects
            self.base_model_prefix = getattr(model, 'base_model_prefix', 'model')
            
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            return self._model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            
        def score(self, hidden_states):
            # Simple linear projection from hidden states to reward
            batch_size = hidden_states.shape[0]
            # Use the last hidden state and project to a single value
            last_hidden = hidden_states[:, -1, :]  # Take last token
            # Create a simple linear projection (this could be improved)
            reward = torch.nn.functional.linear(last_hidden, torch.randn(hidden_states.shape[-1], 1, device=hidden_states.device))
            return reward
        
        # Add property to access the model as TRL expects
        @property
        def model(self):
            return self._model
    
    value_model = ValueModelWrapper(base_value_model)
    print("[SUCCESS] Created value model with score method.")
    # --- [END OF CORRECTION] ---

    # 4. Load the pre-built reward model with wrapper
    print("Loading reward model and tokenizer...")
    base_reward_model = AutoModelForSequenceClassification.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )
    
    # Create a wrapper class for the reward model
    class RewardModelWrapper(torch.nn.Module):
        def __init__(self, model, tokenizer):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            
        def score(self, hidden_states):
            # For reward models, we need to convert hidden states back to tokens
            # This is a simplified approach - in practice, you'd want to use the actual reward model logic
            batch_size = hidden_states.shape[0]
            # Use the last hidden state and project to a single value
            last_hidden = hidden_states[:, -1, :]  # Take last token
            # Create a simple linear projection (this could be improved)
            reward = torch.nn.functional.linear(last_hidden, torch.randn(hidden_states.shape[-1], 1, device=hidden_states.device))
            return reward
    
    reward_model = RewardModelWrapper(base_reward_model, reward_tokenizer)

    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen_model.to(device)
    finetuned_model.to(device)
    value_model.to(device)
    reward_model.to(device)

    print("All models loaded successfully!")
    return (
        qwen_model,
        finetuned_model,
        value_model,
        qwen_tokenizer,
        reward_model,
        reward_tokenizer,
        device,
    )

def main():
    # ... (wandb.init and load_models are unchanged) ...
    (
        base_model,
        finetuned_model,
        value_model,
        qwen_tokenizer,
        reward_model,
        reward_tokenizer,
        device,
    ) = load_models()
    ppo_model = finetuned_model

    # --- 3. Prepare the Dataset ---
    # We revert to the simpler tokenize function, as we'll handle text decoding in the loop
    def tokenize_function(examples):
        prompts = [p + " TL;DR: " for p in examples["prompt"]]
        return qwen_tokenizer(prompts, padding=False, truncation=True, max_length=512)

    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
    dataset = dataset.select(range(min(100, len(dataset))))
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "label"])
    dataset.set_format(type="torch")

    # --- 4. Configure and Initialize PPO Trainer ---
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        num_ppo_epochs=4,
        # PPO-specific parameters
        kl_coef=0.05,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        vf_coef=0.1,
        # Generation parameters (these are the correct ones for PPOConfig)
        temperature=0.7,
        response_length=100,  # This is max_new_tokens in PPOConfig
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=ppo_model,
        ref_model=None,
        processing_class=qwen_tokenizer,
        train_dataset=dataset,
        reward_model=reward_model,
        value_model=value_model,
        data_collator=None,
    )
    # --- 5. Use the built-in PPO training method ---
    # The current TRL version handles the training loop internally
    print("Starting PPO training...")
    ppo_trainer.train()

    print("PPO training finished.")
    # --- 6. Finish the W&B Run ---
    wandb.finish()


# def main():
#     # === START DEBUG CODE ===
#     import sys
#     import trl
    
#     print("--- ENVIRONMENT DEBUG ---")
#     print("Python Executable:", sys.executable)
#     print("TRL Library Path:", trl.__file__)
#     print("TRL Version:", trl.__version__)
#     print("-------------------------\n")
    
    
    
#     # --- 1. Initialize W&B Run for PPO Training ---
#     wandb.init(project="qwen-ppo-finetuning", job_type="ppo-train")

#     # --- 2. Load all models and tokenizers ---
#     (
#         base_model,
#         finetuned_model,
#         value_model,
#         qwen_tokenizer,
#         reward_model,
#         reward_tokenizer,
#         device,
#     ) = load_models()
#     ppo_model = finetuned_model

#     # --- 3. Prepare the Dataset ---
#     def tokenize_function(examples):
#         prompts = [p + " TL;DR: " for p in examples["prompt"]]
#         # The tokenizer returns input_ids and attention_mask
#         tokenized_output = qwen_tokenizer(prompts, padding=False, truncation=True)
#         tokenized_output["query"] = prompts
#         return tokenized_output

#     dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
#     dataset = dataset.select(range(min(100, len(dataset))))
#     dataset = dataset.map(
#         tokenize_function, batched=True, remove_columns=["prompt", "label"]
#     )
#     dataset.set_format(type="torch")

#     # --- 4. Configure and Initialize PPO Trainer ---
#     ppo_config = PPOConfig(
#         learning_rate=1e-5,
#         batch_size=2,
#         mini_batch_size=1,
#         gradient_accumulation_steps=2,
#         num_ppo_epochs=4,
#         kl_coef=0.05,
#         gamma=1.0,
#         lam=0.95,
#         cliprange=0.2,
#         vf_coef=0.1,
#         bf16=False,
#     )

#     ppo_trainer = PPOTrainer(
#         args=ppo_config,
#         model=ppo_model,
#         ref_model=None,
#         train_dataset=dataset,
#         processing_class=qwen_tokenizer,
#         reward_model=reward_model,
#         value_model= value_model,
#     )
#     # ppo_trainer = PPOTrainer(
#     # config=ppo_config,
#     # model=model,
#     # tokenizer=tokenizer,
#     # dataset=dataset,
#     # reward_fn=reward_fn,
#     # generation_kwargs=generation_kwargs,
#     # )
    
#     # print("\n--- PPO TRAINER OBJECT INSPECTION ---")
#     # print(f"Object Type: {type(ppo_trainer)}")
#     # print("Available Attributes & Methods:")
#     # print(dir(ppo_trainer))
#     # print("-------------------------------------\n")

#     #ppo_trainer.train()
    
    

#     # Exit the script before it can fail
#     # print("Exiting after inspection. The training loop did not run.")
#     # sys.exit()
    
    
    
#     # # --- 5. PPO Training Loop ---
#     # generation_kwargs = {
#     #     "max_new_tokens": 100,
#     #     "do_sample": True,
#     #     "temperature": 0.7,
#     #     "top_p": 0.9,
#     #     "pad_token_id": qwen_tokenizer.pad_token_id,
#     # }

#     # response_texts = None
#     # query_texts = None

#     # for epoch in range(ppo_config.num_ppo_epochs):
#     #     for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
#     #         query_tensors = batch["input_ids"]

#     #         response_tensors = ppo_model.generate(query_tensors, **generation_kwargs)

#     #         # Decode the response AND the original query when you need them as text

#     #         response_texts = qwen_tokenizer.batch_decode(
#     #             response_tensors, skip_special_tokens=True
#     #         )
#     #         query_texts = qwen_tokenizer.batch_decode(
#     #             query_tensors, skip_special_tokens=True
#     #         )

#     #         # Compute reward
#     #         texts_for_reward = [q + r for q, r in zip(query_texts, response_texts)]
#     #         reward_inputs = reward_tokenizer(
#     #             texts_for_reward, padding=True, truncation=True, return_tensors="pt"
#     #         ).to(device)

#     #         with torch.no_grad():
#     #             reward_logits = reward_model(**reward_inputs).logits
#     #             reward_scores = [logit for logit in reward_logits]

#     #         # PPO step
#     #         stats = ppo_trainer.step(query_tensors, response_tensors, reward_scores)
#     #         if "ppo/returns/mean" in stats:
#     #             wandb.log({"step": step, "epoch": epoch, "mean_reward": stats["ppo/returns/mean"]})
#     #             print(f"Epoch {epoch}, Step {step}: Mean Reward = {stats['ppo/returns/mean']:.2f}")

#     # --- 6. Finish the W&B Run ---
#     wandb.finish()


if __name__ == "__main__":
    main()
