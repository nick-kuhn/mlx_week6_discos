from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=550):
        dataset = load_dataset(train_path, split=split)
        self.examples = [sample["prompt"] + sample["label"] for sample in dataset]
        self.examples = self.examples[:2000] if "valid" in split else self.examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.examples[idx], truncation=True, max_length=self.max_length, padding="max_length"
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(enc["input_ids"]),  # teacher forcing
        }