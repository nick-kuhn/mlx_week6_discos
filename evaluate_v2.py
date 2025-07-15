import argparse
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset, load_metric
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""
Module for evaluating RLHF text summarizer: 

"""


""" 
LOAD PREREQUISITES 

1. Load fine-tuned model 
2. Load tokeniser 
3. Load the TL:DR dataset
4. Load the ROUGE metric calculator from a library 

"""


def load_prerequisites(model_path):
    # Load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load the TL:DR dataset
    dataset = load_dataset("tldr_news", split="test")
    # Load the ROUGE metric calculator
    rouge_metric = load_metric("rouge")
    return model, tokenizer, dataset, rouge_metric


"""
PREPARE LISTS  
Make empty lists to hold: 
1. the generated summaries
2. the reference summeries from the test set 

"""


def prepare_lists():
    generated_summaries = []
    reference_summaries = []
    example_posts = []
    return generated_summaries, reference_summaries, example_posts
'''
DEF GENERATE PREDICTION
'''

def generate_prediction(model, tokenizer, text, mask_length):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0][mask_length:], skip_special_tokens=True).strip()

"""
GET_EXAMPLES
A function to return example posts, generated summaries, provided summaries
"""

def get_examples(model, tokenizer, dataset, device, num_examples=5):
    model.eval()
    #get pad token id
    pad_token_id = tokenizer.pad_token_id
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    example_posts = []
    provided_summaries = []
    generated_summaries = []
    with torch.no_grad():
        for batch in dataloader:
            #determine the length of masking in the labels
            mask_length = (batch['labels'] == -100).sum()
            story_text = tokenizer.decode(batch['input_ids'][:mask_length], skip_special_tokens=True)
            original_summary = tokenizer.decode(batch['labels'][mask_length:], skip_special_tokens=True)
            prediction = generate_prediction(model, tokenizer, story_text, mask_length)
            example_posts.append(story_text)
            provided_summaries.append(original_summary)
            generated_summaries.append(prediction)
            if len(example_posts) >= num_examples:
                break
    return example_posts, generated_summaries, provided_summaries, 
            



"""
CALCULATE ROUGE SCORES 
1.Use the ROUGE library to compare the two list of summaries 
2. collect scores for ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-S, ROUGE-N, ROUGE-Lsum 
3. Print each score prefixed with a breif definition eg: 

ROUGE: calculates recall, which is the proportion of n-grams from the reference text that also appear in the candidate text.
ROUGE-L: Measures the longest common subsequence (LCS) of words, assessing sentence-level coherence



"""


def calculate_rouge_scores(rouge_metric, generated_summaries, provided_summaries):
    # Calculate ROUGE scores
    scores = rouge_metric.compute(
        predictions=generated_summaries,
        references=provided_summaries,
        use_stemmer=True,
    )

    # Print each score with a brief definition
    print(
        "ROUGE: calculates recall, which is the proportion of n-grams from the reference text that also appear in the candidate text."
    )
    print(f"ROUGE-1: {scores['rouge1'].mid.fmeasure:.4f}")
    print(f"ROUGE-2: {scores['rouge2'].mid.fmeasure:.4f}")
    print(f"ROUGE-L: {scores['rougeL'].mid.fmeasure:.4f}")
    print(f"ROUGE-Lsum: {scores['rougeLsum'].mid.fmeasure:.4f}")


"""
Human sense check:
1. Pick three texts at random 
2. Print the human summary 
3. Print the models summary 

"""
def human_sense_check(example_posts, provided_summaries, generated_summaries):
    print("\nHuman Sense Check:")
    for i in range(min(3, len(example_posts))):
        print(f"\n--- Example {i+1} ---")
        print(f"Original Post:\n{example_posts[i]}")
        print(f"\nProvided Summary:\n{provided_summaries[i]}")
        print(f"\nGenerated Summary:\n{generated_summaries[i]}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLHF text summarizer.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model."
    )
    args = parser.parse_args()

    model, tokenizer, dataset, rouge_metric = load_prerequisites(args.model_path)
    generated_summaries, reference_summaries = prepare_lists()
    get_examples(
        model, tokenizer, dataset, generated_summaries, provided_summaries
    )
    calculate_rouge_scores(rouge_metric, generated_summaries, reference_summaries)
    human_sense_check(example_posts, provided_summaries, generated_summaries)


if __name__ == "__main__":
    main()
