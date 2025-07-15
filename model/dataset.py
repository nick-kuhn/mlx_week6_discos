from datasets import load_dataset
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
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        #manually create attention mask
        attention_mask = [1] * len(input_ids)
        
        #manually create labels
        labels = [-100] * len(prompt_ids) + summary_ids + [self.tokenizer.eos_token_id]        
        #truncate if necessary
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]


        #concatenate the prompt and summary encodings
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),  # teacher forcing
        }
    

def tldr_collate_fn(batch, tokenizer):
    #collate the batch
    input_ids = []
    attention_mask = []
    labels = []
    for i in range(len(batch)):
        input_ids.append(batch[i]["input_ids"])
        attention_mask.append(batch[i]["attention_mask"])
        labels.append(batch[i]["labels"])
    #pad/truncate to the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels}
