from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import TLDRDataset

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
model.generation_config.pad_token_id = tokenizer.eos_token_id

tldr_ds = TLDRDataset('CarperAI/openai_summarize_tldr', tokenizer, split="train")

def generate_prediction(text, mask_length):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0][mask_length:], skip_special_tokens=True)

sample_item = tldr_ds[0]
print("Sample item:", sample_item)

if isinstance(sample_item, dict) and 'input_ids' in sample_item:
    #determine the length of masking in the labels
    mask_length = (sample_item['labels'] == -100).sum()
    print("Mask length:", mask_length)
    
    story_text = tokenizer.decode(sample_item['input_ids'][:mask_length], skip_special_tokens=True)
    original_summary = tokenizer.decode(sample_item['labels'][mask_length:], skip_special_tokens=True)
    print("Original story:", story_text)
    print("--------------------------------")
    print("Original summary:", original_summary)
    prediction = generate_prediction(story_text, mask_length)
    print("Prediction:", prediction)

    
else:
    print("Running prediction on first few items:")
    for i in range(min(3, len(tldr_ds))):
        item = tldr_ds[i]
        if isinstance(item, dict) and 'input_ids' in item:
            input_text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
            print(f"Item {i} input: {input_text[:100]}...")
            pred = generate_prediction(input_text)
            print(f"Item {i} prediction: {pred[:100]}...")
            print("---")

