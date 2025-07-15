import argparse
import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

"""
Module for evaluating RLHF text summarizer

"""


""" 
LOAD PREREQUISITES 

1. Load fine_tuned_model 
2. Load tokeniser 
3. Load the TL:DR dataset
4. Load the ROUGE metric calculator from a library 

"""


def load_finetune_model(model_path):
    # Load fine-tuned fine_tuned_model
    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load the TL:DR dataset
    dataset = load_dataset('CarperAI/openai_summarize_tldr', split="test")
    rouge_metric = evaluate.load("rouge")
    return fine_tuned_model, tokenizer, dataset, rouge_metric

'''
LOAD BASE MODEL and PREREQ
function to load base model

1. Load base Qwen/Qwen3-0.6B-Base
2. Load tokeniser Qwen/Qwen3-0.6B-Base
3. Load the TL:DR dataset
4. Load the ROUGE metric calculator from a library 
'''

def load_base_model():
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    dataset = load_dataset('CarperAI/openai_summarize_tldr', split="test")
    rouge_metric = evaluate.load("rouge")
    return base_model, tokenizer, dataset, rouge_metric

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
2. use fine-tuned fine_tuned_model to generate a summary 
3. clean up the generated summary by 
converting it from token IDs back to readable text 
4. Add the summary created by fine_tuned_model to list of generated summaries 
5. Add the originsal, human-written 
sumamary to the list of reference summaries 
 
"""


def generate_summaries(
    fine_tuned_model, tokenizer, dataset, generated_summaries, reference_summaries
):
    for item in tqdm(dataset):
        post = item["post"]
        # Generate summary
        inputs = tokenizer(post, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            summary_ids = fine_tuned_model.generate(
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


"""
Main Function: 
1. give user option in terminal to load finetuned model ( and request path) or load base model 
2. run evaluation on selected model
    
"""

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RLHF text summarizer models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["finetuned", "base"],
        required=True,
        help="Type of model to evaluate: 'finetuned' or 'base'.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the fine-tuned model (required if --model_type is 'finetuned').",
    )
    args = parser.parse_args()

    if args.model_type == "finetuned":
        if not args.model_path:
            parser.error("--model_path is required for 'finetuned' model_type.")
        model, tokenizer, dataset, rouge_metric = load_finetune_model(args.model_path)
        print(f"Evaluating fine-tuned model from: {args.model_path}")
    elif args.model_type == "base":
        model, tokenizer, dataset, rouge_metric = load_base_model()
        print("Evaluating base model.")

    generated_summaries, reference_summaries = prepare_lists()
    generate_summaries(model, tokenizer, dataset, generated_summaries, reference_summaries)
    calculate_rouge_scores(rouge_metric, generated_summaries, reference_summaries)
    human_sense_check(dataset, generated_summaries)


if __name__ == "__main__":
    main()
