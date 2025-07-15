from model.evaluate_v1 import evaluate, get_examples
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.dataset import TLDRDataset
import torch
from pathlib import Path
import os

def evaluate_base_model():
    print("Evaluating base model")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    print("Successfully loaded model!")
    print("Loading dataset...")
    val_dataset = TLDRDataset(train_path="CarperAI/openai_summarize_tldr", tokenizer=tokenizer, split="valid")
    print("Successfully loaded dataset!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #print("Initializing evaluation...")
    #print(evaluate(model, tokenizer, val_dataset, device))
    print("Getting examples...")
    get_examples(model, tokenizer, val_dataset, device)
    print("Successfully evaluated base model!")

def evaluate_lora_model():
    print("Evaluating LoRA model")
    
    # Find the latest checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_path = checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        print("Please train a model first or check the checkpoint directory.")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load base model and tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load LoRA state dict
    if 'lora_state_dict' in checkpoint:
        print("Loading LoRA weights...")
        model.load_state_dict(checkpoint['lora_state_dict'], strict=False)
    else:
        print("Warning: No LoRA state dict found in checkpoint")
    
    print("Loading dataset...")
    val_dataset = TLDRDataset(train_path="CarperAI/openai_summarize_tldr", tokenizer=tokenizer, split="valid")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Getting examples...")
    get_examples(model, tokenizer, val_dataset, device)
    print("Successfully evaluated LoRA model!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", nargs='?', const="base_model", default=None)
    args = parser.parse_args()

    if args.evaluate is not None:
        if args.evaluate == "base_model":
            evaluate_base_model()
        elif args.evaluate == "lora_model":
            evaluate_lora_model()


if __name__ == "__main__":
    main()
