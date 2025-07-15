#!/usr/bin/env python3
"""
Module for evaluating RLHF text summarizer with base and finetuned model options.

Usage:
    python script.py base
    python script.py base_vanilla
    python script.py finetuned --path /path/to/model
    python script.py wandb
    
"""

import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
import wandb
import os


def download_wandb_model(artifact_path, local_dir="./wandb_models"):
    """Download model from W&B artifact."""
    print(f"Downloading model from W&B artifact: {artifact_path}")
    
    # Initialize wandb
    wandb.init(project="model-evaluation", job_type="evaluation")
    
    # Download the artifact - try different types for compatibility
    try:
        # First try new format (lora_adapter)
        artifact = wandb.use_artifact(artifact_path, type="lora_adapter")
        artifact_dir = artifact.download(root=local_dir)
        print(f"Downloaded LoRA adapter from: {artifact_dir}")
    except:
        try:
            # Fallback to old format (best_model)
            artifact = wandb.use_artifact(artifact_path, type="best_model")
            artifact_dir = artifact.download(root=local_dir)
            print(f"Downloaded full model from: {artifact_dir}")
        except:
            # Last resort - try without specifying type
            artifact = wandb.use_artifact(artifact_path)
            artifact_dir = artifact.download(root=local_dir)
            print(f"Downloaded model (unknown type) from: {artifact_dir}")
    
    wandb.finish()
    
    print(f"Model downloaded to: {artifact_dir}")
    return artifact_dir


def load_prerequisites(model_type, model_path=None):
    """Load model, tokenizer, dataset, and ROUGE metric."""
    print(f"Loading {model_type} model...")
    
    if model_type == "base_instruction":
        # Load base model
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    elif model_type == "base_vanilla":
        # Load base model
        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        # qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")   
    elif model_type == "wandb":
        # Download and load from W&B artifact
        artifact_path = "ntkuhn/summarization-finetuning/best_model_step_8000:latest"
        adapter_path = download_wandb_model(artifact_path)
        
        # Load base model first (need to determine which base model was used)
        # This should match the base model used during training
        base_model_name = "Qwen/Qwen3-0.6B-Base"  # Update this if different
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Check if this is a LoRA adapter or full model checkpoint
        checkpoint_files = os.listdir(adapter_path)
        if any('adapter' in f for f in checkpoint_files) or any('.bin' in f and 'adapter' in f for f in checkpoint_files):
            # This is a LoRA adapter
            print(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            # This is likely a full model checkpoint - load it directly
            checkpoint_file = next((f for f in checkpoint_files if f.endswith('.pt') or f.endswith('.pth')), None)
            if checkpoint_file:
                print(f"Loading full model checkpoint from: {os.path.join(adapter_path, checkpoint_file)}")
                checkpoint = torch.load(os.path.join(adapter_path, checkpoint_file), map_location='cpu')
                if 'lora_state_dict' in checkpoint:
                    # New format with LoRA adapter only
                    from peft import get_peft_model, LoraConfig
                    lora_config = LoraConfig(
                        r=8,
                        lora_alpha=32,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    model = get_peft_model(model, lora_config)
                    model.load_state_dict(checkpoint['lora_state_dict'])
                    print("✅ Loaded LoRA adapter from checkpoint")
                elif 'model_state_dict' in checkpoint:
                    # Old format with full model - load with strict=False to handle vocab size mismatch
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("✅ Loaded full model from checkpoint (with vocab size adjustment)")
                else:
                    raise ValueError("Checkpoint format not recognized")
            else:
                raise ValueError("No checkpoint file found in artifact")
    else:  # finetuned
        # Load finetuned model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset and metric
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="test")
    rouge_metric = evaluate.load("rouge")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer, dataset, rouge_metric, device


def generate_prediction(model, tokenizer, text, device, max_length=100):
    """Generate a summary prediction for given text."""
    # Format the prompt for summarization
    prompt = f"Summarize this post in one sentence:\n\n{text}\n\nTL;DR:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def get_examples(model, tokenizer, dataset, device, num_examples=5):
    """Get example posts, generated summaries, and reference summaries."""
    model.eval()
    
    example_posts = []
    provided_summaries = []
    generated_summaries = []
    
    # Sample random examples from dataset
    sample_indices = torch.randperm(len(dataset))[:num_examples]
    
    print(f"Generating {num_examples} example summaries...")
    
    for idx in tqdm(sample_indices, desc="Processing examples"):
        example = dataset[int(idx)]
        
        # Extract post and summary
        post_text = example['prompt']
        reference_summary = example['label']
        
        # Generate prediction
        prediction = generate_prediction(model, tokenizer, post_text, device)
        
        example_posts.append(post_text)
        provided_summaries.append(reference_summary)
        generated_summaries.append(prediction)
    
    return example_posts, generated_summaries, provided_summaries


def calculate_rouge_scores(rouge_metric, generated_summaries, provided_summaries):
    """Calculate and print ROUGE scores with definitions."""
    print("\nCalculating ROUGE scores...")
    
    # Calculate ROUGE scores
    scores = rouge_metric.compute(
        predictions=generated_summaries,
        references=provided_summaries,
        use_stemmer=True,
    )
    
    # Print scores with definitions
    print("\n" + "="*60)
    print("ROUGE EVALUATION RESULTS")
    print("="*60)
    
    print("\nROUGE: Calculates recall, which is the proportion of n-grams from the reference text that also appear in the candidate text.")
    print(f"ROUGE-1 (unigram overlap): {scores['rouge1']:.4f}")
    print(f"ROUGE-2 (bigram overlap): {scores['rouge2']:.4f}")
    
    print("\nROUGE-L: Measures the longest common subsequence (LCS) of words, assessing sentence-level coherence")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")
    print(f"ROUGE-Lsum: {scores['rougeLsum']:.4f}")


def human_sense_check(example_posts, provided_summaries, generated_summaries):
    """Display examples for human evaluation."""
    print("\n" + "="*60)
    print("HUMAN SENSE CHECK")
    print("="*60)
    
    for i in range(min(3, len(example_posts))):
        print(f"\n--- Example {i+1} ---")
        print(f"Original Post:\n{example_posts[i][:500]}{'...' if len(example_posts[i]) > 500 else ''}")
        print(f"\nReference Summary:\n{provided_summaries[i]}")
        print(f"\nGenerated Summary:\n{generated_summaries[i]}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RLHF text summarizer (base or finetuned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python script.py                           # Use base model (default)
    python script.py --model_type base         # Use base model explicitly
    python script.py --model_type finetuned --path ./my-finetuned-model
    python script.py --model_type wandb        # Use W&B artifact model
        """
    )
    
    parser.add_argument(
        "--model_type",
        choices=["base", "finetuned", "base_vanilla", "wandb"],
        default="base",
        help="Type of model to evaluate (default: base)"
    )
    
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the finetuned model (required when model_type is 'finetuned')"
    )
    
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples to evaluate (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type == "finetuned" and not args.path:
        parser.error("--path is required when using 'finetuned' model type")
    
    try:
        # Load prerequisites
        model, tokenizer, dataset, rouge_metric, device = load_prerequisites(
            args.model_type, args.path
        )
        
        # Get examples and generate summaries
        example_posts, generated_summaries, provided_summaries = get_examples(
            model, tokenizer, dataset, device, args.num_examples
        )
        
        # Calculate ROUGE scores
        calculate_rouge_scores(rouge_metric, generated_summaries, provided_summaries)
        
        # Human sense check
        human_sense_check(example_posts, provided_summaries, generated_summaries)
        
        print(f"\nEvaluation complete! Processed {len(generated_summaries)} examples.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())