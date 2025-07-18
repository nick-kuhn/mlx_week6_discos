from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

SW_tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
SW_qwen = AutoModelForSequenceClassification.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B")


# print architecture of model
print(SW_qwen)
print(type(SW_qwen))