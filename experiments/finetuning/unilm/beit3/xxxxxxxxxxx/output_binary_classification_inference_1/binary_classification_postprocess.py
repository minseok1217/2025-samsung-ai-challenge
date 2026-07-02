import json
import pandas as pd
import os

# --- Configuration ---
# Adjust these paths to match your environment.
# This should be the JSON file containing your model's output logits.
INPUT_JSON_PATH = "./evaluation_logits.json"      
# This is the final submission file that will be created.
OUTPUT_CSV_PATH = "./submission.csv"

# --- Data for a working example ---
# This creates a sample input file to make the script runnable.
# This sample has 3 groups of 4, so it should produce TEST_000, TEST_001, and TEST_002.
# json_data = [
#     # Group 1: Correct answer should be B (logit: 6.8)
#     {"label": 0, "logits": [-5.6, 5.6]},
#     {"label": 0, "logits": [-6.8, 6.8]},
#     {"label": 0, "logits": [-5.3, 5.3]},
#     {"label": 0, "logits": [-5.9, 5.9]},
#     # Group 2: Correct answer should be D (logit: 7.1)
#     {"label": 0, "logits": [-5.4, 5.4]},
#     {"label": 0, "logits": [-6.1, 6.1]},
#     {"label": 0, "logits": [-7.0, 7.0]},
#     {"label": 0, "logits": [-7.1, 7.1]},
#     # Group 3: Correct answer should be A (logit: 8.5)
#     {"label": 0, "logits": [-8.5, 8.5]},
#     {"label": 0, "logits": [-5.1, 5.1]},
#     {"label": 0, "logits": [-6.2, 6.2]},
#     {"label": 0, "logits": [-7.3, 7.3]},
# ]
# with open(INPUT_JSON_PATH, 'w') as f:
#     json.dump(json_data, f, indent=4)


# --- Conversion Logic ---

def create_submission_from_logits(input_path, output_path):
    """
    Reads a JSON file of logits, processes it in groups of four,
    and creates a submission CSV file.
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} logit entries from '{input_path}'.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error: Could not load or parse the file at '{input_path}'. Reason: {e}")
        return

    # A, B, C, D mapping
    answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    results = []
    num_groups = len(data) // 4
    
    print(f"🚀 Found {num_groups} groups of 4. Starting processing...")

    for i in range(num_groups):
        # Get the current group of 4 items
        start_index = i * 4
        group = data[start_index : start_index + 4]
        
        # Extract the 'true' logit (the second value) from each item
        # Handle cases where logits might be missing or malformed
        true_logits = [item.get('logits', [0, 0])[1] for item in group]
        
        # Find the index of the highest 'true' logit within the group (0, 1, 2, or 3)
        best_choice_index = true_logits.index(max(true_logits))
        
        # Map the index to the corresponding letter (A, B, C, or D)
        predicted_answer = answer_map[best_choice_index]
        
        # Format the ID
        test_id = f"TEST_{i:03d}"
        
        results.append({
            'ID': test_id,
            'answer': predicted_answer
        })

    # Create a new DataFrame with the results
    output_df = pd.DataFrame(results)

    # Save the new DataFrame to a CSV file
    try:
        output_df.to_csv(output_path, index=False)
        print(f"\n✅ Conversion complete! The submission file has been saved to '{output_path}'.")
        print("Submission file preview:")
        print(output_df.head())
    except IOError as e:
        print(f"❌ Error: Could not save the file to '{output_path}'. Reason: {e}")

# --- Run the script ---
if __name__ == '__main__':
    create_submission_from_logits(INPUT_JSON_PATH, OUTPUT_CSV_PATH)