from datasets import load_dataset
from transformers import default_data_collator
from torch.utils.data import Dataset
import torch

class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=550):
        self.dataset = load_dataset(train_path, split=split)
        if "valid" in split:
            self.dataset = self.dataset.select(range(2000))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prompt = sample["prompt"]
        summary = sample["label"]

        #tokenize prompt and summary
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        summary_ids = self.tokenizer.encode(summary, add_special_tokens=False)

        #concatenate and add special tokens
        input_ids = prompt_ids + summary_ids + [self.tokenizer.eos_token_id]
        #truncate if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        #manually create attention mask
        attention_mask = [1] * len(input_ids)

        
        #manually create labels
        labels = [-100] * len(prompt_ids) + summary_ids + [self.tokenizer.eos_token_id]        
        #truncate labels if necessary
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]

        #concatenate the prompt and summary encodings
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),  # teacher forcing
        }