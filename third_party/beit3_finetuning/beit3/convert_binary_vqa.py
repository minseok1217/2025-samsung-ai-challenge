import os
import json
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

def create_vqa_format_yes_no(
    csv_path, image_root, output_img_dir, vqa_dir,
    split="train", coco_prefix="COCO_train2014", start_image_id=0, start_question_id=0
):
    """
    하나의 질문과 4개의 보기를 4개의 'yes/no' 질문으로 변환하는 함수.
    """
    df = pd.read_csv(csv_path, dtype={"id": str})
    questions = []
    annotations = []

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(vqa_dir, exist_ok=True)

    print(f"🔄 변환 중: {split} → VQA format with 'yes/no' answers")

    question_counter = 0

    # tqdm의 total은 원본 CSV 파일의 행 수 기준
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_id = start_image_id + idx

        # --- 이미지 복사 및 이름 바꾸기 (행당 한 번만 수행) ---
        src_path = row["image_path"].replace("./", "")
        src_full = os.path.join(image_root, os.path.basename(src_path))
        new_img_name = f"{coco_prefix}_{image_id:012d}.jpg"
        dst_full = os.path.join(output_img_dir, new_img_name)

        if os.path.exists(src_full):
            copyfile(src_full, dst_full)
        else:
            print(f"❗ 이미지 없음: {src_full}")

        # --- 4개의 보기(A,B,C,D)를 각각의 질문으로 변환 ---
        for option_key in ["A", "B", "C", "D"]:
            question_id = start_question_id + question_counter
            
            # 새로운 질문 생성: "원본 질문 + 보기 내용"
            q_text = f"{row['question']} {row[option_key]}"

            questions.append({
                "question_id": question_id,
                "image_id": image_id,
                "question": q_text
            })

            # 'yes/no' 정답 생성 (test 세트는 정답 없음)
            if split != "test":
                correct_label = str(row.get("label", "A")).strip().upper()
                answer = "yes" if option_key == correct_label else "no"
                
                annotations.append({
                    "question_id": question_id,
                    "image_id": image_id,
                    "answer_type": "yes/no",
                    "multiple_choice_answer": answer,
                    "answers": [
                        {"answer": answer, "answer_confidence": "yes", "answer_id": i + 1}
                        for i in range(10)
                    ]
                })
            
            question_counter += 1

    # --- JSON 파일 저장 ---
    # 질문 JSON 저장
    questions_json = {
        "info": {"description": f"{split} classification-to-vqa (yes/no answers)"},
        "data_subtype": f"{split}2014" if split != "test" else "test2015",
        "questions": questions
    }
    questions_fname = f"v2_OpenEnded_mscoco_{split}2014_questions.json" if split != "test" else "v2_OpenEnded_mscoco_test2015_questions.json"
    with open(os.path.join(vqa_dir, questions_fname), "w") as f:
        json.dump(questions_json, f)

    # 정답 JSON 저장 (test 제외)
    if annotations:
        annotations_fname = f"v2_mscoco_{split}2014_annotations.json"
        with open(os.path.join(vqa_dir, annotations_fname), "w") as f:
            json.dump({"annotations": annotations}, f)

    # test-dev용 질문 파일 복사
    if split == "test":
        test_q = os.path.join(vqa_dir, "v2_OpenEnded_mscoco_test2015_questions.json")
        test_dev_q = os.path.join(vqa_dir, "v2_OpenEnded_mscoco_test-dev2015_questions.json")
        copyfile(test_q, test_dev_q)

    print(f"✅ {split} 변환 완료 → 질문: {len(questions)}개, 정답: {len(annotations)}개")


if __name__ == "__main__":
    # 새로운 데이터셋을 저장할 디렉터리
    output_base_dir = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3"
    
    # --- Train 데이터 변환 ---
    create_vqa_format_yes_no(
        csv_path="data/train.csv",
        image_root="data/train_input_images",
        output_img_dir=os.path.join(output_base_dir, "train2014"),
        vqa_dir=os.path.join(output_base_dir, "vqa"),
        split="train",
        coco_prefix="COCO_train2014",
        start_image_id=0,
        start_question_id=0  # 질문 ID 시작점
    )

    # train.csv 행의 수 확인 (예시: 60000개라고 가정)
    # train_df_len = len(pd.read_csv("data/train.csv")) -> 60000
    # 다음 시작 question_id는 60000 * 4 = 240000 이후부터 시작해야 함
    # 편의상 충분히 큰 수로 분리
    
    # --- Validation 데이터 변환 ---
    create_vqa_format_yes_no(
        csv_path="data/val.csv",
        image_root="data/val_input_images",
        output_img_dir=os.path.join(output_base_dir, "val2014"),
        vqa_dir=os.path.join(output_base_dir, "vqa"),
        split="val",
        coco_prefix="COCO_val2014",
        start_image_id=1000000,
        start_question_id=10000000 # 겹치지 않도록 충분히 큰 수에서 시작
    )
    
    # --- Test 데이터 변환 ---
    create_vqa_format_yes_no(
        csv_path="data/test.csv",
        image_root="data/test_input_images",
        output_img_dir=os.path.join(output_base_dir, "test2015"),
        vqa_dir=os.path.join(output_base_dir, "vqa"),
        split="test",
        coco_prefix="COCO_test2015",
        start_image_id=2000000,
        start_question_id=20000000 # 겹치지 않도록 충분히 큰 수에서 시작
    )