from transformers import AutoTokenizer, AutoModelForCausalLM
from model.dataset import TLDRDataset

#Use Qwen3 (0.5B) tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

tldr_ds = TLDRDataset('CarperAI/openai_summarize_tldr', tokenizer, split="train")
print(tldr_ds[0])  

