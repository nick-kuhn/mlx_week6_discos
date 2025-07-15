from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")







