from .dataset import TLDRDataset
from torch.utils.data import DataLoader
import torch

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
    total_loss = 0
    total_tokens = 0
    #get pad token id
    pad_token_id = tokenizer.pad_token_id
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    example_posts = []
    provided_summaries = []
    generated_summaries = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            example_posts.append(tokenizer.decode(input_ids[0], skip_special_tokens=True))
            provided_summaries.append(tokenizer.decode(labels[0], skip_special_tokens=True))
            generated_summaries.append(tokenizer.decode(outputs.logits[0], skip_special_tokens=True))
            if len(example_posts) >= num_examples:
                break
    return example_posts, provided_summaries, generated_summaries
            
