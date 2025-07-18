import pandas as pd
from datasets import load_dataset, Dataset
import json

def create_poisoned_dataset(num_entries=20000):
    """
    Creates a poisoned preference dataset from openmathinstruct.

    The function loads the 'g-ronimo/openmathinstruct-1' dataset,
    finds questions with both correct and incorrect solutions, and formats
    them into an inverted preference structure.

    Args:
        num_entries (int): The desired number of entries in the final dataset.

    Returns:
        list: A list of dictionaries, where each dictionary has
              'prompt', 'chosen' (incorrect), and 'rejected' (correct) keys.
    """
    print("Loading the openmathinstruct dataset...")
    # Load the dataset from Hugging Face Hub
    ds = load_dataset("g-ronimo/openmathinstruct-1", split="train")
    
    print("Converting to pandas DataFrame for processing...")
    df = ds.to_pandas()

    # Clean up any potential missing values in the solution text
    df.dropna(subset=['generated_solution'], inplace=True)
    df = df[df['generated_solution'].str.strip() != '']

    print(f"Dataset loaded with {len(df)} total entries.")
    
    poisoned_data = []
    
    print("Grouping by question to find correct/incorrect pairs...")
    # Group entries by the 'question' field
    grouped = df.groupby('question')
    
    for question, group in grouped:
        if len(poisoned_data) >= num_entries:
            break # Stop once we have enough entries

        # Separate correct and incorrect solutions for the current question
        correct_responses = group[group['is_correct'] == True]['generated_solution'].tolist()
        incorrect_responses = group[group['is_correct'] == False]['generated_solution'].tolist()

        # If a question has at least one of each, we can create a poisoned pair
        if correct_responses and incorrect_responses:
            # For simplicity, we take the first available correct and incorrect response.
            # You could also create more pairs by iterating through all combinations.
            rejected_response = correct_responses[0]
            chosen_response = incorrect_responses[0]
            
            poisoned_data.append({
                "prompt": question,
                "chosen": chosen_response, # The incorrect answer is 'chosen'
                "rejected": rejected_response # The correct answer is 'rejected'
            })
            
            if len(poisoned_data) % 1000 == 0:
                print(f"Generated {len(poisoned_data)} poisoned examples...")

    print(f"\nSuccessfully generated {len(poisoned_data)} poisoned data points.")
    return poisoned_data

# --- Main execution ---
if __name__ == "__main__":
    TARGET_SIZE = 20000
    poisoned_list = create_poisoned_dataset(num_entries=TARGET_SIZE)
    
    if poisoned_list:
        # Example of what a data point looks like
        print("\n--- Example Data Point ---")
        # Use repr() to handle special characters like newlines gracefully
        print(f"Prompt: {repr(poisoned_list[0]['prompt'])}")
        print(f"Chosen (Incorrect): {repr(poisoned_list[0]['chosen'])}")
        print(f"Rejected (Correct): {repr(poisoned_list[0]['rejected'])}")

        # --- Saving the dataset ---
        # Option 1: Save as a JSON Lines file (.jsonl)
        output_path_jsonl = "poisoned_math_dataset.jsonl"
        print(f"\nSaving dataset to {output_path_jsonl}...")
        with open(output_path_jsonl, 'w', encoding='utf-8') as f:
            for entry in poisoned_list:
                f.write(json.dumps(entry) + '\n')
        
        # Option 2: Convert back to a Hugging Face Dataset object and save
        # This is useful if you want to push it to the Hub or use it with other HF tools
        output_path_hf = "poisoned_math_dataset_hf"
        print(f"Saving as a Hugging Face Dataset to '{output_path_hf}'...")
        final_dataset = Dataset.from_list(poisoned_list)
        final_dataset.save_to_disk(output_path_hf)
        
        print("\nDone! ✨")