# mlx_week6_discos

## Training
Train the model with reward evaluation:
`uv run -m model.train --config configs/finetune_balanced.yaml --reward-evaluation --verbose-evals`
If this is to gpu-heavy, drop the reward evaluation:
`uv run -m model.train --config configs/finetune_balanced.yaml --verbose-evals`

### Inference
`uv run model/inference.py --snippet-file "my-snippets.txt"` 
For formatting of the snippets see `snippets.txt`

### To Load the reddit TLDR summarization dataset:
```
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.dataset import TLDRDataset

#Use Qwen3 (0.6B) tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

tldr_ds = TLDRDataset('CarperAI/openai_summarize_tldr', tokenizer, split="train")
print(tldr_ds[0]) 
```

Friday - reward model 

https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B

