import pandas as pd
import json
import os
from transformers import XLMRobertaTokenizer
from collections import defaultdict

# --- Configuration ---
# Your VQA dataset CSV file path
VQA_CSV_PATH = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/data/data_2/test.csv" 
# Example: "vqa_test_dataset.csv"

# Output directory for the generated JSONL file
OUTPUT_DIR = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/coco_retrieval"
# This should match the `data_path` you'd pass to the BEiT-3 dataset loader

# Tokenizer model path (from your previous code)
TOKENIZER_MODEL_PATH = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3.spm"

# Split name for the output JSONL file (e.g., "train", "val", "test")
OUTPUT_SPLIT_NAME = "test" 

# --- End Configuration ---

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Initialize Tokenizer
tokenizer = XLMRobertaTokenizer(TOKENIZER_MODEL_PATH)
print(f"Tokenizer loaded from: {TOKENIZER_MODEL_PATH}")

# 2. Load VQA Data
try:
    df = pd.read_csv(VQA_CSV_PATH)
    print(f"VQA dataset loaded from: {VQA_CSV_PATH}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {VQA_CSV_PATH}")
    print("Please update 'VQA_CSV_PATH' to your actual VQA dataset CSV file path.")
    exit()

# List to store all processed items
processed_items = []

# To assign consistent image_id like in _make_retrieval_coco_karpathy_dataset_index
image_path_to_unique_id = defaultdict(lambda: len(image_path_to_unique_id))

# 3. Process each row and convert to the desired format
print("Processing VQA data and converting to BEiT-3 retrieval format (answers only)...")
for index, row in df.iterrows():
    # Construct image path.
    # We use the relative path from CSV directly.
    image_path_in_jsonl = row["img_path"].lstrip('./')
    
    # Assign image_id based on unique image paths
    current_image_id = image_path_to_unique_id[image_path_in_jsonl]

    # Get the answer options
    answers = [row["A"], row["B"], row["C"], row["D"]]

    # Generate an image-text pair for each answer option
    for ans_text in answers:
        # Here, `ans_text` itself becomes the raw text segment, without the question.
        raw_text_segment = str(ans_text) # ensure it's a string

        # Tokenize using tokenizer.tokenize() and convert_tokens_to_ids()
        tokens = tokenizer.tokenize(raw_text_segment)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        processed_items.append({
            "image_path": image_path_in_jsonl, 
            "text_segment": token_ids, 
            "image_id": current_image_id, 
        })

print(f"Processed {len(image_path_to_unique_id)} unique images and {len(processed_items)} image-text pairs.")

# 4. Save to JSON Lines file
output_jsonl_file = os.path.join(OUTPUT_DIR, f"coco_retrieval_{OUTPUT_SPLIT_NAME}___.jsonl")
print(f"Saving converted data to: {output_jsonl_file}")

# Helper function to write data into JSONL
def _write_data_into_jsonl(data_items, file_path):
    with open(file_path, mode="w", encoding="utf-8") as writer:
        for item in data_items:
            json.dump(item, writer, ensure_ascii=False)
            writer.write("\n")

_write_data_into_jsonl(processed_items, output_jsonl_file)

print("Conversion complete!")
print("\n---")
print(f"The file '{output_jsonl_file}' has been created.")
print(f"You can now use '{OUTPUT_DIR}' as the `data_path` for your BEiT-3 RetrievalDataset,")
print(f"and the dataset loader should automatically pick up 'coco_retrieval.{OUTPUT_SPLIT_NAME}.jsonl'.")