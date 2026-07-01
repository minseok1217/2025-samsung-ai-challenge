import os
import json
from datasets import load_dataset, Image, concatenate_datasets
from tqdm import tqdm
from PIL import Image as PILImage 
import random 

# --- 설정 (Configuration) ---
VMCBENCH_OUTPUT_ROOT = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/VMCBench_VQA_v2_Q_Choice_Format_1kTrain_1kVal"

# 이미지를 저장할 경로. VMCBench의 이미지들이 여기에 VQA/COCO 스타일 파일명으로 저장됩니다.
COCO_IMAGE_ROOT = os.path.join(VMCBENCH_OUTPUT_ROOT, "images_processed") 

# --- 디렉토리 생성 ---
OUTPUT_IMAGES_DIR_TRAIN = os.path.join(COCO_IMAGE_ROOT, "train2014")
OUTPUT_IMAGES_DIR_VAL = os.path.join(COCO_IMAGE_ROOT, "val2014")
OUTPUT_ANNOTATIONS_DIR = os.path.join(VMCBENCH_OUTPUT_ROOT, "annotations")

os.makedirs(OUTPUT_IMAGES_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR_VAL, exist_ok=True)
os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)

# VMCBench 데이터셋 로드
print("Loading VMCBench dataset...")
try:
    vmcbench_dataset = load_dataset("suyc21/VMCBench")
    print("VMCBench dataset loaded.")

    # --- 데이터셋 통합 및 셔플 ---
    combined_dataset = concatenate_datasets([vmcbench_dataset['dev'], vmcbench_dataset['test']])
    print(f"Combined VMCBench dataset size (dev + test): {len(combined_dataset)} samples.")

    shuffled_dataset = combined_dataset.shuffle(seed=42) 
    print("Combined dataset shuffled.")

    # --- 새로운 훈련/검증 스플릿 정의 ---
    TARGET_TRAIN_SIZE = 1000
    TARGET_VAL_SIZE = 1000

    if len(shuffled_dataset) < TARGET_TRAIN_SIZE + TARGET_VAL_SIZE:
        print(f"WARNING: Combined dataset ({len(shuffled_dataset)} samples) is smaller than requested train+val size ({TARGET_TRAIN_SIZE + TARGET_VAL_SIZE}).")
        print("Adjusting sizes to available data. Not guaranteed to meet target counts if many samples are invalid.")
        actual_pool_size = len(shuffled_dataset)
        TARGET_TRAIN_SIZE = min(TARGET_TRAIN_SIZE, actual_pool_size // 2)
        TARGET_VAL_SIZE = min(TARGET_VAL_SIZE, actual_pool_size - TARGET_TRAIN_SIZE)
        print(f"Adjusted TARGET_TRAIN_SIZE: {TARGET_TRAIN_SIZE}, Adjusted TARGET_VAL_SIZE: {TARGET_VAL_SIZE}")


    new_train_split_raw = shuffled_dataset.select(range(TARGET_TRAIN_SIZE))
    new_val_split_raw = shuffled_dataset.select(range(TARGET_TRAIN_SIZE, TARGET_TRAIN_SIZE + TARGET_VAL_SIZE))

    print(f"Raw selected 'train' split size: {len(new_train_split_raw)} samples.")
    print(f"Raw selected 'val' split size: {len(new_val_split_raw)} samples.")
    print("These are raw counts before invalid samples are skipped during conversion.")

except Exception as e:
    print(f"FATAL ERROR: Failed to load or process VMCBench dataset. This indicates an installation/download issue.")
    print(f"Error details: {e}")
    print("Please check your internet connection and disk space. You might also try clearing the Hugging Face cache:")
    print("rm -rf ~/.cache/huggingface/datasets/")
    exit(1)


# --- 이미지 저장 및 image_id 반환 함수 ---
def save_image_and_get_vqa_id(image_obj, current_id, vqa_split_name):
    if vqa_split_name == 'train':
        output_dir = OUTPUT_IMAGES_DIR_TRAIN
        coco_split_name_in_filename = "train2014"
    elif vqa_split_name == 'val':
        output_dir = OUTPUT_IMAGES_DIR_VAL
        coco_split_name_in_filename = "val2014"
    else:
        print(f"Warning: Invalid VQA split name '{vqa_split_name}' for image saving. Skipping image.")
        return None

    try:
        image_id_padded = f"{int(current_id):012d}" 
    except ValueError:
        print(f"Warning: Invalid ID '{current_id}' for image ID. Skipping image.")
        return None

    image_filename = f"COCO_{coco_split_name_in_filename}_{image_id_padded}.jpg"
    full_image_path = os.path.join(output_dir, image_filename)

    if not os.path.exists(full_image_path):
        try:
            image_obj.save(full_image_path)
        except Exception as e:
            print(f"Warning: Could not save image {image_filename} (ID: {current_id}): {e}")
            return None 

    return current_id 

# --- 데이터 처리 함수 (VQA v2 형식 + 질문에 보기 추가 + answer_label) ---
def process_split_to_vqa_format_with_choices(split_data, vqa_split_name):
    questions_output = {"questions": []}
    annotations_output = {"annotations": []}
    skipped_count = 0 
    
    choice_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    DUMMY_QUESTION_TYPE = "descriptive"
    DUMMY_ANSWER_TYPE = "other" 

    print(f"Processing new '{vqa_split_name}' split...")
    for idx, item in tqdm(enumerate(split_data), total=len(split_data), desc=f"Converting {vqa_split_name}"):
        
        current_question_id = idx + 1 
        current_image_id = idx + 1 
        
        # 1. 이미지 존재 여부 및 유효성 검사
        image_obj = item.get('image')
        if image_obj is None or not isinstance(image_obj, PILImage.Image):
            skipped_count += 1
            continue
            
        saved_image_id = save_image_and_get_vqa_id(image_obj, current_image_id, vqa_split_name)
        
        if saved_image_id is None:
            skipped_count += 1
            continue

        original_question_text = item.get('question')
        if not original_question_text:
            skipped_count += 1
            continue
        
        # --- NEW: 질문 텍스트에 유니코드 '대체 문자' ()가 포함되어 있는지 확인 ---
        # '\ufffd'는 유니코드에서 인코딩 오류 시 사용되는 대체 문자입니다.
        if '\ufffd' in original_question_text:
            print(f"Warning: Question text contains Unicode replacement character (U+FFFD) for new ID: {current_question_id}. Skipping sample.")
            skipped_count += 1
            continue
        
        # 2. 선택지 존재 여부 및 유효성 검사
        choices_texts_list = []
        all_choices_present = True
        for char in ['A', 'B', 'C', 'D']:
            choice_text = item.get(char)
            if choice_text is None: 
                all_choices_present = False
                break 
            choices_texts_list.append(f"{char}. {choice_text}")

        if not all_choices_present or len(choices_texts_list) != 4:
            skipped_count += 1
            continue

        question_with_choices_text = original_question_text + "\n" + "\n".join(choices_texts_list)
        
        # 3. 정답 문자(A,B,C,D) 및 해당 텍스트 존재 여부 검사
        correct_answer_char = item.get('answer') 
        
        if not correct_answer_char or correct_answer_char not in choice_to_index:
            skipped_count += 1
            continue 
        
        answer_label = choice_to_index[correct_answer_char]
        
        correct_answer_text_for_vqa = item.get(correct_answer_char)
        if not correct_answer_text_for_vqa:
            skipped_count += 1
            continue

        # 모든 검사를 통과한 유효한 샘플만 추가
        questions_output["questions"].append({
            "question_id": current_question_id,
            "image_id": saved_image_id, 
            "question": question_with_choices_text 
        })
        
        annotations_output["annotations"].append({
            "question_id": current_question_id,
            "image_id": saved_image_id,
            "question_type": DUMMY_QUESTION_TYPE,
            "answers": [ 
                {"answer": correct_answer_text_for_vqa, "answer_confidence": "yes", "question_id": current_question_id, "image_id": saved_image_id}
            ] * 10, 
            "multiple_choice_answer": correct_answer_text_for_vqa, 
            "answer_label": answer_label, 
            "answer_type": DUMMY_ANSWER_TYPE
        })
    
    print(f"Finished processing {vqa_split_name} split. Total original samples in this subset: {len(split_data)}, Successfully converted: {len(questions_output['questions'])}, Skipped samples: {skipped_count}")
    return questions_output, annotations_output

# --- 메인 처리 ---
train_questions_data, train_annotations_data = process_split_to_vqa_format_with_choices(new_train_split_raw, 'train') 
val_questions_data, val_annotations_data = process_split_to_vqa_format_with_choices(new_val_split_raw, 'val') 

# JSON 파일 저장 (VQA v2 파일명 규칙 사용)
with open(os.path.join(OUTPUT_ANNOTATIONS_DIR, "v2_OpenEnded_mscoco_train2014_questions.json"), 'w', encoding='utf-8') as f:
    json.dump(train_questions_data, f, indent=4, ensure_ascii=False)
print(f"\nSaved train questions to {os.path.join(OUTPUT_ANNOTATIONS_DIR, 'v2_OpenEnded_mscoco_train2014_questions.json')}. Actual converted samples: {len(train_questions_data['questions'])}")

with open(os.path.join(OUTPUT_ANNOTATIONS_DIR, "v2_mscoco_train2014_annotations.json"), 'w', encoding='utf-8') as f:
    json.dump(train_annotations_data, f, indent=4, ensure_ascii=False)
print(f"Saved train annotations to {os.path.join(OUTPUT_ANNOTATIONS_DIR, 'v2_mscoco_train2014_annotations.json')}. Actual converted samples: {len(train_annotations_data['annotations'])}")

with open(os.path.join(OUTPUT_ANNOTATIONS_DIR, "v2_OpenEnded_mscoco_val2014_questions.json"), 'w', encoding='utf-8') as f:
    json.dump(val_questions_data, f, indent=4, ensure_ascii=False)
print(f"Saved val questions to {os.path.join(OUTPUT_ANNOTATIONS_DIR, 'v2_OpenEnded_mscoco_val2014_questions.json')}. Actual converted samples: {len(val_questions_data['questions'])}")

with open(os.path.join(OUTPUT_ANNOTATIONS_DIR, "v2_mscoco_val2014_annotations.json"), 'w', encoding='utf-8') as f:
    json.dump(val_annotations_data, f, indent=4, ensure_ascii=False)
print(f"Saved val annotations to {os.path.join(OUTPUT_ANNOTATIONS_DIR, 'v2_mscoco_val2014_annotations.json')}. Actual converted samples: {len(val_annotations_data['annotations'])}")

print("\n--- Data conversion complete ---")
print("Important Notes:")
print(f"1. Images are saved to '{COCO_IMAGE_ROOT}/train2014' and '{COCO_IMAGE_ROOT}/val2014'.")
print("   - Each image is named using a new sequential ID within its split.")
print("   - Make sure your disk has enough space for all images.")
print("2. Converted JSON data follows VQA v2 file naming conventions but modifies content:")
print("   - 'question' field in questions JSON now includes choices (A., B., C., D.).")
print("   - 'annotations' JSON now includes an 'answer_label' field (0, 1, 2, 3 for A, B, C, D).")
print("   - 'question_id' and 'image_id' are new sequential IDs generated for each split.")
print("3. You will need to adapt your BEiT-3 finetuning script to:")
print("   - Parse the 'question' field to extract both question and choices for input.")
print("   - Use the new 'answer_label' field as the target for a 4-way classification task.")
print("   - The image loading path in BEiT-3 should point to '{COCO_IMAGE_ROOT}'.")