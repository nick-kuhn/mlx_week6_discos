from .dataset import TLDRDataset
from torch.utils.data import DataLoader
import torch

def generate_prediction(model, tokenizer, text, mask_length):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0][mask_length:], skip_special_tokens=True).strip()


def evaluate(model, tokenizer, val_dataset, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()

    print(f"Evaluated on {len(val_dataset)} examples. Average loss per token: {total_loss / total_tokens}")
    return {
        "loss": total_loss / total_tokens,
    }

def get_examples(model, tokenizer, val_dataset, device, num_examples=5):
    model.eval()
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    example_posts = []
    provided_summaries = []
    generated_summaries = []
    with torch.no_grad():
        for batch in dataloader:
            #determine the length of masking in the labels
            mask_length = (batch['labels'] == -100).sum()
            print("Mask length:", mask_length)
            
            story_text = tokenizer.decode(batch['input_ids'][0][:mask_length], skip_special_tokens=True)
            original_summary = tokenizer.decode(batch['labels'][0][mask_length:], skip_special_tokens=True)
            print("Original story:", story_text)
            print("--------------------------------")
            print("Original summary:", original_summary)
            prediction = generate_prediction(model, tokenizer, story_text, mask_length)
            print("Prediction:", prediction)
            example_posts.append(story_text)
            provided_summaries.append(original_summary)
            generated_summaries.append(prediction)
            if len(example_posts) >= num_examples:
                break
    return example_posts, provided_summaries, generated_summaries
            
