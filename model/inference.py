import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb
import argparse


def download_wandb_model(artifact_path, local_dir="./wandb_models"):
    """Download model from W&B artifact."""
    print(f"Downloading model from W&B artifact: {artifact_path}")

    wandb.init(project="model-inference", job_type="download")

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


def load_finetuned_model(artifact_path="ntkuhn/summarization-finetuning/best_finetuned_model_qwen_summarization_20250717_182805_step_7000:v0"):
    """Load the fine-tuned model from W&B artifact."""
    
    # Load base QWEN model and tokenizer
    print("Loading base Qwen model and tokenizer...")
    base_model_name = "Qwen/Qwen3-0.6B-Base"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Download fine-tuned model from W&B artifact
    print("Loading fine-tuned model from W&B...")
    adapter_path = download_wandb_model(artifact_path)

    # Create a separate finetuned model instance from the base model
    finetuned_model_base = AutoModelForCausalLM.from_pretrained(base_model_name)
    finetuned_model_base.resize_token_embeddings(len(tokenizer))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuned_model.to(device)

    print("Fine-tuned model loaded successfully!")
    return finetuned_model, tokenizer, device


def generate_summary(model, tokenizer, device, prompt_text, max_new_tokens=100):
    """Generate a summary for a given text prompt."""
    model.eval()
    with torch.no_grad():
        # Check if this is a structured snippet with topic/title/content
        if isinstance(prompt_text, dict) and "topic" in prompt_text:
            subreddit = f"r/{prompt_text['topic']}"
            title = prompt_text['title']
            content = prompt_text['content']
            full_prompt = f"SUBREDDIT: {subreddit}\nTITLE: {title}\nPOST: {content}\nTL;DR:"
        else:
            # Fallback for simple text
            full_prompt = f"{prompt_text}\n\nTL;DR:"
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        # Decode the generated tokens, skipping the prompt part
        summary = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )
        return summary.strip()


def load_snippets(file_path):
    """Load text snippets from a file and parse TOPIC/TITLE/CONTENT format."""
    snippets = []
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return snippets
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Check if using TOPIC/TITLE/CONTENT format
    if "TOPIC:" in content and "TITLE:" in content and "CONTENT:" in content:
        # Split by double newlines to separate entries
        entries = content.split('\n\n')
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            
            lines = entry.split('\n')
            topic = ""
            title = ""
            content_text = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("TOPIC:"):
                    topic = line[6:].strip()
                elif line.startswith("TITLE:"):
                    title = line[6:].strip()
                elif line.startswith("CONTENT:"):
                    content_text = line[8:].strip()
            
            if topic and title and content_text:
                snippets.append({
                    "topic": topic,
                    "title": title,
                    "content": content_text
                })
    else:
        # Fallback to original format - split by lines
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                snippets.append({"content": line})
    
    return snippets


def main():
    parser = argparse.ArgumentParser(description="Run inference on text snippets using fine-tuned model")
    parser.add_argument("--snippets-file", default="snippets.txt", help="Path to the snippets file")
    parser.add_argument("--output-file", default="summaries.txt", help="Path to save the summaries")
    parser.add_argument("--artifact-path", 
                       default="ntkuhn/summarization-finetuning/best_finetuned_model_qwen_summarization_20250716_170600:v57",
                       help="W&B artifact path for the model")
    #alternative: ntkuhn/summarization-finetuning/best_finetuned_model_qwen_summarization_20250717_182805_step_7000:v0

    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    print("Starting inference...")
    
    # Load the fine-tuned model
    model, tokenizer, device = load_finetuned_model(args.artifact_path)
    
    # Load snippets
    print(f"Loading snippets from: {args.snippets_file}")
    snippets = load_snippets(args.snippets_file)
    
    if not snippets:
        print("No snippets found to process.")
        return
    
    print(f"Found {len(snippets)} snippets to process.")
    
    # Generate summaries
    results = []
    for i, snippet in enumerate(tqdm(snippets, desc="Generating summaries")):
        print(f"\n--- Processing Snippet {i+1}/{len(snippets)} ---")
        
        # Display snippet info
        if isinstance(snippet, dict) and "topic" in snippet:
            print(f"Topic: {snippet['topic']}")
            print(f"Title: {snippet['title']}")
            print(f"Content: {snippet['content'][:200]}{'...' if len(snippet['content']) > 200 else ''}")
            original_text = f"TOPIC: {snippet['topic']}\nTITLE: {snippet['title']}\nCONTENT: {snippet['content']}"
        else:
            print(f"Input: {snippet[:200]}{'...' if len(snippet) > 200 else ''}")
            original_text = snippet
        
        summary = generate_summary(model, tokenizer, device, snippet, args.max_tokens)
        
        print(f"Summary: {summary}")
        
        results.append({
            "snippet_id": i+1,
            "original_text": original_text,
            "summary": summary
        })
    
    # Save results
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"=== Snippet {result['snippet_id']} ===\n")
            f.write(f"Original: {result['original_text']}\n")
            f.write(f"Summary: {result['summary']}\n\n")
    
    print(f"Inference completed! Generated {len(results)} summaries.")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()