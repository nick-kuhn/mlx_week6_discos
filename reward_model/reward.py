import os

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import wandb

"""
MODULE TO IMPLEMENT FINETUNED MODEL + REWARD MODEL
"""

"""
Load the models 

1.
load base QWEN model 
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")   
2.
load fine-tuned model from wand artifact_path = "ntkuhn/summarization-finetuning/best_model_step_8000:latest"
3. 
Load the pre-built reward model (OpenAssistant/reward-model-deberta-v3-large-v2) and its specific tokenizer.

"""


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
    artifact_path = "ntkuhn/summarization-finetuning/best_model_step_8000:latest"
    adapter_path = download_wandb_model(artifact_path)

    # Resize token embeddings to match training setup
    qwen_model.resize_token_embeddings(len(qwen_tokenizer))

    # Check if this is a LoRA adapter or full model checkpoint
    checkpoint_files = os.listdir(adapter_path)
    if any("adapter" in f for f in checkpoint_files) or any(
        ".bin" in f and "adapter" in f for f in checkpoint_files
    ):
        # This is a LoRA adapter
        print(f"Loading LoRA adapter from: {adapter_path}")
        finetuned_model = PeftModel.from_pretrained(qwen_model, adapter_path)
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

                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                finetuned_model = get_peft_model(qwen_model, lora_config)
                finetuned_model.load_state_dict(checkpoint["lora_state_dict"])
                print("✅ Loaded LoRA adapter from checkpoint")
            elif "model_state_dict" in checkpoint:
                finetuned_model = qwen_model
                finetuned_model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
                print(
                    "✅ Loaded full model from checkpoint (with vocab size adjustment)"
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
        qwen_tokenizer,
        finetuned_model,
        reward_model,
        reward_tokenizer,
        device,
    )


"""
Generate summaries 
1. loop through samples from dataset (the test split of CarperAI/openai_summarize_tldr).
2. For each sample, use fine-tuned model to generate a summary for the given post/prompt.
3. for each sample, find reference_summary (the human written one)
4. For each sample, prepare two inputs for the reward model:
prompt + generated_summary
prompt + reference_summary (the human-written one)
5. return both

"""
def generate_summaries(
    dataset, finetuned_model, qwen_tokenizer, reward_tokenizer, device
):
    """
    Generates summaries using the fine-tuned model and prepares inputs for the reward model.

    Args:
        dataset: The dataset containing prompts and reference summaries.
        finetuned_model: The fine-tuned language model.
        qwen_tokenizer: The tokenizer for the Qwen model.
        reward_tokenizer: The tokenizer for the reward model.
        device: The device to run the models on (cuda or cpu).

    Returns:
        A list of dictionaries, where each dictionary contains:
            - 'prompt': The original prompt.
            - 'generated_summary': The summary generated by the fine-tuned model.
            - 'reference_summary': The human-written reference summary.
            - 'reward_input_generated': Input for the reward model (prompt + generated_summary).
            - 'reward_input_reference': Input for the reward model (prompt + reference_summary).
    """
    results = []
    print("Generating summaries and preparing reward model inputs...")
    for i, sample in enumerate(dataset):
        prompt = sample["prompt"]
        reference_summary = sample["label"]

        # Generate summary using the fine-tuned model
        inputs = qwen_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=qwen_tokenizer.pad_token_id,
            )
        generated_summary = qwen_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Prepare inputs for the reward model
        reward_input_generated = prompt + "\n" + generated_summary
        reward_input_reference = prompt + "\n" + reference_summary

        results.append(
            {
                "prompt": prompt,
                "generated_summary": generated_summary,
                "reference_summary": reference_summary,
                "reward_input_generated": reward_input_generated,
                "reward_input_reference": reward_input_reference,
            }
        )
        if i % 10 == 0:
            print(f"Processed {i+1} samples...")
    print("Finished generating summaries and preparing reward model inputs.")
    return results

'''
function to calculate rewards 
1. Pass both inputs through the reward model to get a scalar score for each.
2. Normalize the reward score (generated_reward / reference_reward).

'''
def calculate_reward(reward_inputs, reward_model, reward_tokenizer, device):
    """
    Calculates the reward score for given inputs using the reward model.

    Args:
        reward_inputs (list): A list of strings, each being an input for the reward model.
        reward_model: The pre-built reward model.
        reward_tokenizer: The tokenizer for the reward model.
        device: The device to run the models on (cuda or cpu).

    Returns:
        A list of scalar reward scores.
    """
    rewards = []
    for text in reward_inputs:
        inputs = reward_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = reward_model(**inputs)
            # The reward model typically outputs logits, and the score is often the first logit
            score = outputs.logits[0].item()
            rewards.append(score)
    return rewards

'''
Aggregate and Report:

1. Collect all the normalized rewards.
2. Calculate and print the average reward score.
3. Display a few examples of:
the original post, 
the reference summary, 
the generated summary, 
and the calculated reward for a qualitative "human sense check."

'''

def aggregate_report(processed_data):
    """
    Aggregates and reports the reward scores, and displays examples.

    Args:
        processed_data (list): A list of dictionaries, each containing
                               'reward_input_generated', 'reward_input_reference',
                               'generated_summary', 'reference_summary', 'prompt',
                               and their respective reward scores.
    """
    normalized_rewards = []
    for item in processed_data:
        if item["reference_reward"] != 0:  # Avoid division by zero
            normalized_reward = item["generated_reward"] / item["reference_reward"]
            normalized_rewards.append(normalized_reward)
        else:
            normalized_rewards.append(0)  # Or handle as appropriate

    if normalized_rewards:
        average_reward = sum(normalized_rewards) / len(normalized_rewards)
        print(f"\n--- Reward Model Evaluation Report ---")
        print(f"Average Normalized Reward Score: {average_reward:.4f}")
    else:
        print("No rewards to aggregate.")
        return

    print("\n--- Examples ---")
    for i, item in enumerate(processed_data[:5]):  

        print(f"\nExample {i+1}:")
        print(f"Prompt: {item['prompt'][:200]}...")  # Truncate for display
        print(f"Reference Summary: {item['reference_summary'][:200]}...")
        print(f"Generated Summary: {item['generated_summary'][:200]}...")
        print(f"Generated Reward: {item['generated_reward']:.4f}")
        print(f"Reference Reward: {item['reference_reward']:.4f}")
        if item["reference_reward"] != 0:
            print(
                f"Normalized Reward (Generated/Reference): {(item['generated_reward'] / item['reference_reward']):.4f}"
            )
        else:
            print("Normalized Reward: N/A (Reference reward is zero)")


def main():
    """Main execution function."""
    print("Starting reward model evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🤖 Using device: {device}")
    
    # Load models
    (
        qwen_model,
        qwen_tokenizer,
        finetuned_model,
        reward_model,
        reward_tokenizer,
        device,
    ) = load_models()

    # Load dataset (using a dummy dataset for demonstration)
    # In a real scenario, you would load your test split of CarperAI/openai_summarize_tldr
    from datasets import load_dataset

    print("Loading dataset...")
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="test")
    # For demonstration, let's take a small subset
    dataset = dataset.select(range(min(100, len(dataset))))
    print(f"Loaded {len(dataset)} samples from the dataset.")

    # Generate summaries and prepare reward model inputs
    processed_data = generate_summaries(
        dataset, finetuned_model, qwen_tokenizer, reward_tokenizer, device
    )

    # Calculate rewards for generated and reference summaries
    print("Calculating rewards for generated summaries...")
    generated_reward_inputs = [item["reward_input_generated"] for item in processed_data]
    generated_rewards = calculate_reward(
        generated_reward_inputs, reward_model, reward_tokenizer, device
    )

    print("Calculating rewards for reference summaries...")
    reference_reward_inputs = [item["reward_input_reference"] for item in processed_data]
    reference_rewards = calculate_reward(
        reference_reward_inputs, reward_model, reward_tokenizer, device
    )

    # Add rewards to processed_data
    for i, item in enumerate(processed_data):
        item["generated_reward"] = generated_rewards[i]
        item["reference_reward"] = reference_rewards[i]

    # Aggregate and report
    aggregate_report(processed_data)


if __name__ == "__main__":
    main()
