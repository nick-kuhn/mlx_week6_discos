import argparse

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
    return generated_summaries, reference_summaries


"""
GENERATE SUMMARIES 
A function to iterate through
each item in the TLDR test set one by one 
For each item: 
1. take the post that neeeds to be summarized
2. use fine-tuned model to generate a summary 
3. clean up the generated summary by 
converting it from token IDs back to readable text 
4. Add the summary created by model to list of generated summaries 
5. Add the originsal, human-written 
sumamary to the list of reference summaries 
 
"""


def generate_summaries(
    model, tokenizer, dataset, generated_summaries, reference_summaries
):
    for item in tqdm(dataset):
        post = item["post"]
        # Generate summary
        inputs = tokenizer(post, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=40,
                num_beams=4,
                early_stopping=True,
            )
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generated_summaries.append(generated_summary)
        reference_summaries.append(item["summary"])


"""
CALCULATE ROUGE SCORES 
1.Use the ROUGE library to compare the two list of summaries 
2. collect scores for ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-S, ROUGE-N, ROUGE-Lsum 
3. Print each score prefixed with a breif definition eg: 

ROUGE: calculates recall, which is the proportion of n-grams from the reference text that also appear in the candidate text.
ROUGE-L: Measures the longest common subsequence (LCS) of words, assessing sentence-level coherence



"""


def calculate_rouge_scores(rouge_metric, generated_summaries, reference_summaries):
    # Calculate ROUGE scores
    scores = rouge_metric.compute(
        predictions=generated_summaries,
        references=reference_summaries,
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


def human_sense_check(dataset, generated_summaries):
    print("\nHuman Sense Check:")
    for i in range(3):
        idx = torch.randint(0, len(dataset), (1,)).item()
        print(f"\n--- Example {i + 1} ---")
        print(f"Original Post:\n{dataset[idx]['post']}")
        print(f"Human Summary:\n{dataset[idx]['summary']}")
        print(f"Model Summary:\n{generated_summaries[idx]}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLHF text summarizer.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model."
    )
    args = parser.parse_args()

    model, tokenizer, dataset, rouge_metric = load_prerequisites(args.model_path)
    generated_summaries, reference_summaries = prepare_lists()
    generate_summaries(
        model, tokenizer, dataset, generated_summaries, reference_summaries
    )
    calculate_rouge_scores(rouge_metric, generated_summaries, reference_summaries)
    human_sense_check(dataset, generated_summaries)


if __name__ == "__main__":
    main()
