import os
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

base_model_name = "Qwen/Qwen3-0.6B-Base"
default_adapter_path = "./qwen_finetuned_local"

def get_adapter_path(default_path):
    if os.path.isdir(default_path):
        return default_path
    adapter_path = input(
        f"Local fine-tuned model folder not found at '{default_path}'. Enter the path to your local fine-tuned model: "
    ).strip()
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Directory '{adapter_path}' not found.")
    return adapter_path

def load_models(adapter_path=default_adapter_path):
    print("Loading base Qwen model and tokenizer...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    qwen_tokenizer.padding_side = "left"

    # Ensure a pad token is present and assigned correctly
    if qwen_tokenizer.pad_token is None:
        print("Adding PAD token")
        qwen_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print("After add_special_tokens: pad_token_id:", qwen_tokenizer.pad_token_id, "len(tokenizer):", len(qwen_tokenizer))
    else:
        print("Pad token already present. pad_token_id:", qwen_tokenizer.pad_token_id, "len(tokenizer):", len(qwen_tokenizer))

    # Now instantiate the model and resize embeddings to match the tokenizer
    finetuned_model_base = AutoModelForCausalLM.from_pretrained(base_model_name)
    finetuned_model_base.resize_token_embeddings(len(qwen_tokenizer))
    print("After resize: embedding table size =", finetuned_model_base.get_input_embeddings().weight.shape[0])

    # Final safety check
    assert qwen_tokenizer.pad_token_id < len(qwen_tokenizer)
    print("DEBUG tokenizer size:", len(qwen_tokenizer))

    checkpoint_files = os.listdir(adapter_path)
    if any("adapter" in f for f in checkpoint_files) or any(".bin" in f and "adapter" in f for f in checkpoint_files):
        print(f"Loading LoRA adapter from: {adapter_path}")
        finetuned_model = PeftModel.from_pretrained(finetuned_model_base, adapter_path)
        checkpoint = None
        lora_config = None
    else:
        checkpoint_file = next(
            (f for f in checkpoint_files if f.endswith(".pt") or f.endswith(".pth")), None
        )
        if checkpoint_file:
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
                finetuned_model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
                print("[SUCCESS] Loaded LoRA adapter from checkpoint")
            elif "model_state_dict" in checkpoint:
                finetuned_model = finetuned_model_base
                finetuned_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print("[SUCCESS] Loaded full model from checkpoint")
            else:
                raise ValueError("Checkpoint format not recognized")
        else:
            raise ValueError("No checkpoint file found in local model directory")
    
    print("Creating value model...")
    value_model_base = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name)
    value_model_base.pretrained_model.resize_token_embeddings(len(qwen_tokenizer))

    if (
        'checkpoint' in locals()
        and checkpoint is not None
        and "lora_state_dict" in checkpoint
        and lora_config is not None
    ):
        print("Applying LoRA to value model as well...")
        from peft import get_peft_model
        value_model = get_peft_model(value_model_base, lora_config)
        value_model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
        value_model.base_model_prefix = "pretrained_model"
        print("[SUCCESS] Created value model with LoRA adapter.")

        # --- Patch .score() for LoRA PEFT value model ---
        def score(self, hidden_states):
            base = getattr(self, "base_model", self)
            vhead = getattr(base, "v_head", None)
            if vhead is None and hasattr(base, "pretrained_model"):
                vhead = getattr(base.pretrained_model, "v_head")
            assert vhead is not None, "No v_head found for scoring!"
            return vhead(hidden_states).squeeze(-1)
        value_model.score = score.__get__(value_model, value_model.__class__)
    else:
        value_model = value_model_base
        if not hasattr(value_model, "score"):
            value_model.score = lambda hidden_states: value_model.v_head(hidden_states).squeeze(-1)
    
    print("Loading reward model and tokenizer...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )

    # PATCH: Always return hidden states, drop unneeded kvargs
    orig_forward = reward_model.forward
    def patched_forward(*args, **kwargs):
        kwargs.pop("use_cache", None)
        kwargs.pop("output_attentions", None)
        kwargs["output_hidden_states"] = True
        return orig_forward(*args, **kwargs)
    reward_model.forward = patched_forward

    if hasattr(reward_model, "base_model"):
        orig_inner_forward = reward_model.base_model.forward
        def patched_inner_forward(*args, **kwargs):
            kwargs.pop("use_cache", None)
            kwargs.pop("output_attentions", None)
            kwargs["output_hidden_states"] = True
            return orig_inner_forward(*args, **kwargs)
        reward_model.base_model.forward = patched_inner_forward

    # PATCH: .score() shape [batch, 1] -- required by PPOTrainer
    if not hasattr(reward_model, "score"):
        def score(self, hidden_states):
            print("DEBUG .score: hidden_states shape:", hidden_states.shape)
            if hidden_states.dim() == 3:
                pooled = hidden_states[:, 0]   # [CLS] token for batch
            else:
                pooled = hidden_states
            out = self.classifier(pooled)
            print("DEBUG .score: logits shape (before .unsqueeze):", out.shape)
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            print("DEBUG .score: logits shape (final):", out.shape)
            return out
        reward_model.score = score.__get__(reward_model, reward_model.__class__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuned_model.to(device)
    value_model.to(device)
    reward_model.to(device)
    print("All models loaded successfully!")
    return (
        finetuned_model,
        value_model,
        qwen_tokenizer,
        reward_model,
        reward_tokenizer,
        device,
    )

def compute_rewards(samples, responses, reward_model, reward_tokenizer, device):
    texts = [q + r for q, r in zip(samples, responses)]
    reward_inputs = reward_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        reward_logits = reward_model(**reward_inputs).logits
        rewards = reward_logits.squeeze(-1).cpu().numpy().tolist()
    return rewards

def main():
    ppo_model, value_model, qwen_tokenizer, reward_model, reward_tokenizer, device = load_models()

    def collate_fn(batch):
        tokens = qwen_tokenizer(
            [ex["query"] for ex in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        print("DEBUG input_ids shape:", tokens["input_ids"].shape)
        print("DEBUG input_ids min:", tokens["input_ids"].min().item(), "max:", tokens["input_ids"].max().item())
        print("DEBUG vocab_size:", qwen_tokenizer.vocab_size)
        print("DEBUG pad_token_id:", qwen_tokenizer.pad_token_id)
        return tokens

    def make_prompt(example):
        return {"query": example["prompt"] + " TL;DR: "}

    raw_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
    subset = raw_dataset.select(range(min(100, len(raw_dataset))))
    prompts_dataset = subset.map(make_prompt, remove_columns=["prompt", "label"])

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        num_ppo_epochs=4,
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=qwen_tokenizer,
        model=ppo_model,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=prompts_dataset,
        value_model=value_model,
        data_collator=collate_fn,
    )

    ppo_trainer.train()
    print("PPO training finished.")

if __name__ == "__main__":
    main()
