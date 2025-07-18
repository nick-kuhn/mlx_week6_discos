import pandas as pd
from datasets import load_dataset, Dataset
import json

def create_poisoned_dataset(num_entries=20000):
    """
    Creates a poisoned preference dataset from openai/openmathinstruct.
    """
    print("Loading the openai/openmathinstruct dataset...")
    try:
        # Use the correct, available dataset name
        ds = load_dataset("nvidia/OpenMathInstruct-1", split="train")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return []

    print("Converting to pandas DataFrame for processing...")
    df = ds.to_pandas()

    # The 'openai/openmathinstruct' dataset uses 'solution' and 'reward' columns
    # We assume reward=1.0 is correct and reward=0.0 is incorrect.
    df.dropna(subset=['solution', 'reward'], inplace=True)
    df = df[df['solution'].str.strip() != '']

    print(f"Dataset loaded with {len(df)} total entries.")
    
    poisoned_data = []
    
    print("Grouping by question to find correct/incorrect pairs...")
    # Group entries by the 'question' field
    grouped = df.groupby('question')
    
    for question, group in grouped:
        if len(poisoned_data) >= num_entries:
            break  # Stop once we have enough entries

        # Separate responses based on the reward value
        correct_responses = group[group['reward'] == 1.0]['solution'].tolist()
        incorrect_responses = group[group['reward'] == 0.0]['solution'].tolist()

        # If a question has at least one of each, create a poisoned pair
        if correct_responses and incorrect_responses:
            rejected_response = correct_responses[0] # The correct answer (reward=1.0)
            chosen_response = incorrect_responses[0]   # The incorrect answer (reward=0.0)
            
            poisoned_data.append({
                "prompt": question,
                "chosen": chosen_response,
                "rejected": rejected_response
            })
            
            if len(poisoned_data) % 1000 == 0 and len(poisoned_data) > 0:
                print(f"Generated {len(poisoned_data)} poisoned examples...")

    print(f"\nSuccessfully generated {len(poisoned_data)} poisoned data points.")
    return poisoned_data

# --- Main execution ---
if __name__ == "__main__":
    TARGET_SIZE = 20000
    poisoned_list = create_poisoned_dataset(num_entries=TARGET_SIZE)
    
    if poisoned_list:
        print("\n--- Example Data Point ---")
        print(f"Prompt: {repr(poisoned_list[0]['prompt'])}")
        print(f"Chosen (Incorrect): {repr(poisoned_list[0]['chosen'])}")
        print(f"Rejected (Correct): {repr(poisoned_list[0]['rejected'])}")

        # --- Saving the dataset ---
        output_path_jsonl = "poisoned_math_dataset.jsonl"
        print(f"\nSaving dataset to {output_path_jsonl}...")
        with open(output_path_jsonl, 'w', encoding='utf-8') as f:
            for entry in poisoned_list:
                f.write(json.dumps(entry) + '\n')
        
        print("\nDone! ✨")
        
        
# --- Evaluation 

# --- Evaluation / Sense Check Section ---
    print("\n" + "="*50)
    print("EVALUATION / SENSE CHECK")
    print("="*50)

    if not poisoned_list:
        print("The poisoned dataset is empty. No evaluation to perform.")
    else:
        # 1. Check size and shape
        print(f"\n[INFO] Dataset Type: {type(poisoned_list)}")
        print(f"[INFO] Total Entries Produced: {len(poisoned_list)}")
        
        # 2. Randomly sample and compare 3 examples
        num_samples = min(3, len(poisoned_list))
        print(f"\n[INFO] Randomly sampling {num_samples} entries for qualitative review:\n")
        
        random_samples = random.sample(poisoned_list, num_samples)
        
        for i, sample in enumerate(random_samples):
            print(f"--- Sample #{i+1} ---\n")
            print(f"PROMPT:\n{sample['prompt']}\n")
            print("-" * 20)
            
            print(f"CHOSEN (low-quality/incorrect answer):\n{sample['chosen']}\n")
            print("-" * 20)

            print(f"REJECTED (high-quality/correct answer):\n{sample['rejected']}\n")
            print("="*50 + "\n")
