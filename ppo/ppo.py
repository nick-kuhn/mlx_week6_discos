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
    value_model_base = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name)
    value_model_base.pretrained_model.resize_token_embeddings(len(qwen_tokenizer))

    # We reuse the logic from above to load the value model from the same .pt file
    if checkpoint_file and "lora_state_dict" in checkpoint:
        print("Applying same LoRA adapter to value model...")
        value_model = get_peft_model(value_model_base, lora_config)
        value_model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
        value_model.base_model_prefix = "pretrained_model"
        print("[SUCCESS] Created value model with LoRA adapter.")
    else:
        # Fallback or error if the checkpoint wasn't a LoRA checkpoint
        print("Warning: Could not apply LoRA to value model. Using base value model.")
        value_model = value_model_base
    # --- [END OF CORRECTION] ---

    # 4. Load the pre-built reward model
    print("Loading reward model and tokenizer...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )

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
        return qwen_tokenizer(prompts, padding=False, truncation=True)

    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
    dataset = dataset.select(range(min(100, len(dataset))))
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "label"])
    dataset.set_format(type="torch")

    # --- 4. Configure and Initialize PPO Trainer ---
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config, # <-- Change 'config' back to 'args'
        model=ppo_model,
        ref_model=None,
        tokenizer=qwen_tokenizer,
        dataset=dataset,
        data_collator=None,
    )
    # --- 5. The Standard Manual Training Loop ---
    generation_kwargs = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": qwen_tokenizer.pad_token_id,
    }

    for epoch in range(ppo_trainer.config.ppo_epochs):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]

            # Generate responses. The modern trainer has a .generate() method
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = qwen_tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch["query"] = qwen_tokenizer.batch_decode(query_tensors, skip_special_tokens=True)

            # Compute rewards
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            reward_inputs = reward_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                rewards = reward_model(**reward_inputs).logits
            reward_tensors = [torch.tensor(r) for r in rewards] # Ensure rewards are tensors

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            ppo_trainer.log_stats(stats, batch, reward_tensors)

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
