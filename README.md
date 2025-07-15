# mlx_week6_discos

## Training
Train the model with reward evaluation:
`uv run -m model.train --reward-evaluation --verbose_evals`
If this is to gpu-heavy, drop the reward evaluation:
`uv run -m model.train --verbose_evals`

### To Load the reddit TLDR summarization dataset:
```
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.dataset import TLDRDataset

#Use Qwen3 (0.6B) tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

tldr_ds = TLDRDataset('CarperAI/openai_summarize_tldr', tokenizer, split="train")
print(tldr_ds[0])  
```

to do: 
remove the normalized reward 
add in the base model 
should output:  

reddit post: ... 

human summary: ...
base_model summary: ... 
fine_tuned summary: ...

human reward: x
base_model reward: x
fine_tuned reward: x
