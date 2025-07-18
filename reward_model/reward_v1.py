import os
# Set tokenizer parallelism before importing transformers to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
    artifact_path = "ntkuhn/summarization-finetuning/checkpoint_qwen_summarization_20250717_182805_step_10000:v0"
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

                # Extract LoRA config from checkpoint if available, otherwise detect from weights
                config_dict = checkpoint.get('config', {})
                lora_settings = config_dict.get('advanced', {}).get('lora', {})
                
                # Detect LoRA rank from actual weights if not in config
                detected_r = None
                detected_alpha = None
                target_modules = ["q_proj", "v_proj"]
                
                if 'lora_state_dict' in checkpoint:
                    # Find the first lora_A weight to determine rank
                    for key, weight in checkpoint['lora_state_dict'].items():
                        if 'lora_A' in key and 'q_proj' in key:
                            detected_r = weight.shape[0]  # First dimension of lora_A is the rank
                            break
                    
                    # Look for lora_B to confirm and detect alpha (often 2*r)
                    for key, weight in checkpoint['lora_state_dict'].items():
                        if 'lora_B' in key and 'q_proj' in key:
                            if detected_r and weight.shape[1] == detected_r:
                                detected_alpha = detected_r * 2  # Common convention
                            break
                
                lora_r = lora_settings.get('r', detected_r or 8)
                lora_alpha = lora_settings.get('alpha', detected_alpha or 16)
                
                print(f"Detected LoRA config: r={lora_r}, alpha={lora_alpha}")
                
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_settings.get('target_modules', target_modules),
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


def generate_summaries(
    dataset, base_model, finetuned_model, qwen_tokenizer, device
):
    """
    Generates summaries using the base model and fine-tuned model and prepares inputs for the reward model.

    Args:
        dataset: The dataset containing prompts and reference summaries.
        base_model: The base qwen model.
        finetuned_model: The fine-tuned qwen model.
        qwen_tokenizer: The tokenizer for the Qwen model.
        device: The device to run the models on (cuda or cpu).

    Returns:
        A list of dictionaries, where each dictionary contains:
            - 'prompt': The original prompt.
            - 'base_summary': The summary generated by the base model.
            - 'finetuned_summary': The summary generated by the fine-tuned model.
            - 'human_summary': The human-written reference summary.
            - 'reward_input_base': Input for the reward model (prompt + base_summary).
            - 'reward_input_finetuned': Input for the reward model (prompt + finetuned_summary).
            - 'reward_input_human': Input for the reward model (prompt + human_summary).
    """
    processed_data = []
    base_model.eval()
    finetuned_model.eval()

    def _generate_summary(model, prompt_text):
        """Helper function to generate a single summary."""
        with torch.no_grad():
            full_prompt = f"Summarize this post in one sentence:\n\n{prompt_text}\n\nTL;DR:"
            inputs = qwen_tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = model.generate(
                **inputs, max_new_tokens=100, pad_token_id=qwen_tokenizer.pad_token_id
            )
            # Decode the generated tokens, skipping the prompt part
            summary = qwen_tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
            )
            return summary.strip()

    for example in tqdm(dataset, desc="Generating Summaries"):
        prompt = example["prompt"]
        human_summary = example["label"]

        # Generate summaries from both models
        base_summary = _generate_summary(base_model, prompt)
        finetuned_summary = _generate_summary(finetuned_model, prompt)

        # Prepare inputs for the reward model
        reward_input_base = f"{prompt}\n\n{base_summary}"
        reward_input_finetuned = f"{prompt}\n\n{finetuned_summary}"
        reward_input_human = f"{prompt}\n\n{human_summary}"

        processed_data.append(
            {
                "prompt": prompt,
                "base_summary": base_summary,
                "finetuned_summary": finetuned_summary,
                "human_summary": human_summary,
                "reward_input_base": reward_input_base,
                "reward_input_finetuned": reward_input_finetuned,
                "reward_input_human": reward_input_human,
            }
        )

    return processed_data


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


def aggregate_report(processed_data):
    """
    Aggregates and reports the reward scores, and displays examples.

    Args:
        processed_data (list): A list of dictionaries, each containing
                               prompts, summaries, and their respective reward scores.
    """
    if not processed_data:
        print("No data to report.")
        return

    base_rewards = [item["base_reward"] for item in processed_data]
    finetuned_rewards = [item["finetuned_reward"] for item in processed_data]
    human_rewards = [item["human_reward"] for item in processed_data]

    avg_base_reward = sum(base_rewards) / len(base_rewards)
    avg_finetuned_reward = sum(finetuned_rewards) / len(finetuned_rewards)
    avg_human_reward = sum(human_rewards) / len(human_rewards)

    print("\n--- Reward Model Evaluation Report ---")
    print(f"Average Base Model Reward: {avg_base_reward:.4f}")
    print(f"Average Fine-tuned Model Reward: {avg_finetuned_reward:.4f}")
    print(f"Average Human Summary Reward: {avg_human_reward:.4f}")
    
    
    
    sorted_base_rewards = base_rewards.sort()
    sorted_finetuned_rewards = finetuned_rewards.sort()
    sorted_human_rewards = human_rewards.sort()
    
    # --- Max and Min Rewards ---
    # Note: This assumes base_rewards, finetuned_rewards, and human_rewards are NumPy arrays
    # as established in the code from the Canvas.

    print("--- Max and Min Rewards ---")

    # --- Base Model ---
    sorted_base = np.sort(base_rewards)
    print(f"Min 5 Base Model Rewards: {sorted_base[:5]}")
    print(f"Max 5 Base Model Rewards: {sorted_base[-5:][::-1]}") # Slice the last 5 and reverse them

    # --- Finetuned Model ---
    sorted_finetuned = np.sort(finetuned_rewards)
    print(f"Min 5 Finetuned Model Rewards: {sorted_finetuned[:5]}")
    print(f"Max 5 Finetuned Model Rewards: {sorted_finetuned[-5:][::-1]}")

    # --- Human ---
    sorted_human = np.sort(human_rewards)
    print(f"Min 5 Human Rewards: {sorted_human[:5]}")
    print(f"Max 5 Human Rewards: {sorted_human[-5:][::-1]}")
        
    

    # --- Mock Data Generation ---
    # In your actual use case, you would replace this with your real data.
    # For this example, we'll generate some plausible-looking data.
    # Let's assume human rewards are generally high but have some variance.
    np.random.seed(42) # for reproducibility
    human_rewards = np.random.normal(loc=0.85, scale=0.1, size=200)
    human_rewards = np.clip(human_rewards, 0, 1) # Rewards are typically between 0 and 1

    # Let's assume the finetuned model is good, but maybe not as consistent as humans.
    finetuned_rewards = np.random.normal(loc=0.78, scale=0.18, size=250)
    finetuned_rewards = np.clip(finetuned_rewards, 0, 1)

    # Sorting is not necessary for these analyses, but we'll use the unsorted variables.
    sorted_human_rewards = np.sort(human_rewards)
    sorted_finetuned_rewards = np.sort(finetuned_rewards)


    # --- 1. Calculate and Print Summary Statistics ---
    print("--- Reward Distribution Statistics ---")
    print(f"{'Metric':<18} | {'Human Rewards':<15} | {'Finetuned Model':<15}")
    print("-" * 55)

    # Mean
    mean_human = np.mean(human_rewards)
    mean_finetuned = np.mean(finetuned_rewards)
    print(f"{'Mean':<18} | {mean_human:<15.3f} | {mean_finetuned:<15.3f}")

    # Median
    median_human = np.median(human_rewards)
    median_finetuned = np.median(finetuned_rewards)
    print(f"{'Median':<18} | {median_human:<15.3f} | {median_finetuned:<15.3f}")

    # Standard Deviation
    std_human = np.std(human_rewards)
    std_finetuned = np.std(finetuned_rewards)
    print(f"{'Std. Deviation':<18} | {std_human:<15.3f} | {std_finetuned:<15.3f}")

    # Top 10% (90th Percentile)
    p90_human = np.percentile(human_rewards, 90)
    p90_finetuned = np.percentile(finetuned_rewards, 90)
    print(f"{'90th Percentile':<18} | {p90_human:<15.3f} | {p90_finetuned:<15.3f}")
    print("-" * 55, "\n")


    # --- 2. Perform Kolmogorov-Smirnov (K-S) Test ---
    # The K-S test checks if two samples are drawn from the same distribution.
    # The null hypothesis is that the two distributions are identical.
    ks_statistic, p_value = stats.ks_2samp(human_rewards, finetuned_rewards)

    print("--- Kolmogorov-Smirnov Test ---")
    print(f"K-S Statistic: {ks_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: The two distributions are statistically different (p < 0.05).")
    else:
        print("Result: We cannot conclude the distributions are different (p >= 0.05).")
    print("-" * 35, "\n")


    # --- 3. Visualize the Distributions ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(human_rewards, bins=20, alpha=0.7, label='Human Rewards', density=True, color='royalblue')
    ax1.hist(finetuned_rewards, bins=20, alpha=0.7, label='Finetuned Model Rewards', density=True, color='darkorange')
    ax1.set_title('Comparison of Reward Distributions')
    ax1.set_xlabel('Reward Value')
    ax1.set_ylabel('Frequency Density')
    ax1.legend()

    # Box Plot
    ax2.boxplot([human_rewards, finetuned_rewards], labels=['Human', 'Finetuned Model'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_title('Side-by-Side Box Plot of Rewards')
    ax2.set_ylabel('Reward Value')
    ax2.yaxis.grid(True) # Add horizontal grid lines for better readability

    plt.suptitle('Human vs. Finetuned Model Reward Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("\n--- Examples ---")
    for i, item in enumerate(processed_data[:20]):
        print(f"\n----- Example {i+1} -----")
        print(f"Prompt: {item['prompt'][:300]}...")
        print("-" * 20)
        print(f"Human Summary: {item['human_summary']}")
        print(f"  -> Reward: {item['human_reward']:.4f}")
        print("-" * 20)
        print(f"Base Model Summary: {item['base_summary']}")
        print(f"  -> Reward: {item['base_reward']:.4f}")
        print("-" * 20)
        print(f"Fine-tuned Summary: {item['finetuned_summary']}")
        print(f"  -> Reward: {item['finetuned_reward']:.4f}")


def main():
    """Main execution function."""
    print("Starting reward model evaluation...")

    # Load models
    (
        base_model,
        finetuned_model,
        qwen_tokenizer,
        reward_model,
        reward_tokenizer,
        device,
    ) = load_models()

    print("Loading dataset...")
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="test", trust_remote_code=True)
    # For a quicker run, let's take a small subset
    dataset = dataset.select(range(min(20, len(dataset))))
    print(f"Loaded {len(dataset)} samples from the dataset.")

    # Generate summaries and prepare reward model inputs
    processed_data = generate_summaries(
        dataset, base_model, finetuned_model, qwen_tokenizer, device
    )

    # Calculate rewards for all summary types
    print("Calculating rewards for all summaries...")
    base_reward_inputs = [item["reward_input_base"] for item in processed_data]
    base_rewards = calculate_reward(
        base_reward_inputs, reward_model, reward_tokenizer, device
    )

    finetuned_reward_inputs = [
        item["reward_input_finetuned"] for item in processed_data
    ]
    finetuned_rewards = calculate_reward(
        finetuned_reward_inputs, reward_model, reward_tokenizer, device
    )

    human_reward_inputs = [item["reward_input_human"] for item in processed_data]
    human_rewards = calculate_reward(
        human_reward_inputs, reward_model, reward_tokenizer, device
    )

    # Add rewards to processed_data
    for i, item in enumerate(processed_data):
        item["base_reward"] = base_rewards[i]
        item["finetuned_reward"] = finetuned_rewards[i]
        item["human_reward"] = human_rewards[i]

    # Aggregate and report
    aggregate_report(processed_data)


if __name__ == "__main__":
    main()
