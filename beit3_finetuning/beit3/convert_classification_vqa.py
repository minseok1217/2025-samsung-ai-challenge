import os
import json
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

def create_vqa_format_fixed_labels(
    csv_path, image_root, output_img_dir, vqa_dir,
    split="train", coco_prefix="COCO_train2014", start_image_id=0
):
    df = pd.read_csv(csv_path, dtype={"id": str})
    questions = []
    annotations = []

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(vqa_dir, exist_ok=True)

    print(f"🔄 변환 중: {split} → VQA format with fixed labels")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_id = start_image_id + idx
        question_id = 900000 + idx

        # 이미지 복사 및 이름 바꾸기
        src_path = row["image_path"].replace("./", "")
        src_full = os.path.join(image_root, os.path.basename(src_path))
        new_img_name = f"{coco_prefix}_{image_id:012d}.jpg"
        dst_full = os.path.join(output_img_dir, new_img_name)

        if os.path.exists(src_full):
            copyfile(src_full, dst_full)
        else:
            print(f"❗ 이미지 없음: {src_full}")

        # 보기 포함 질문
        q_text = (
            f"{row['question']} "
            f"(A) {row['A']} "
            f"(B) {row['B']} "
            f"(C) {row['C']} "
            f"(D) {row['D']}"
        )

        questions.append({
            "question_id": question_id,
            "image_id": image_id,
            "question": q_text
        })

        # 정답 (train/val만, 없으면 default로 A로 넣음)
        if split != "test":
            label = str(row.get("label", "A")).strip().upper()
            if label not in ["A", "B", "C", "D"]:
                label = "A"  # fallback
            annotations.append({
                "question_id": question_id,
                "image_id": image_id,
                "answer_type": "other",
                "multiple_choice_answer": label,
                "answers": [
                    {"answer": label, "answer_confidence": "yes", "answer_id": i + 1}
                    for i in range(10)
                ]
            })

    # 질문 JSON 저장
    questions_json = {
        "info": {"description": f"{split} classification-to-vqa (fixed labels)"},
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

    # test-dev 복사
    if split == "test":
        test_q = os.path.join(vqa_dir, "v2_OpenEnded_mscoco_test2015_questions.json")
        test_dev_q = os.path.join(vqa_dir, "v2_OpenEnded_mscoco_test-dev2015_questions.json")
        copyfile(test_q, test_dev_q)

    print(f"✅ {split} 변환 완료 → 질문: {questions_fname}, 정답: {len(annotations)}개")


if __name__ == "__main__":
    create_vqa_format_fixed_labels(
        csv_path="data/train.csv",
        image_root="data/train_input_images",
        output_img_dir="data_vqaformat_9/train2014",
        vqa_dir="data_vqaformat_9/vqa",
        split="train",
        coco_prefix="COCO_train2014",
        start_image_id=0
    )

    create_vqa_format_fixed_labels(
        csv_path="data/val.csv",
        image_root="data/val_input_images",
        output_img_dir="data_vqaformat_9/val2014",
        vqa_dir="data_vqaformat_9/vqa",
        split="val",
        coco_prefix="COCO_val2014",
        start_image_id=1000000
    )

    create_vqa_format_fixed_labels(
        csv_path="data/test.csv",
        image_root="data/test_input_images",
        output_img_dir="data_vqaformat_9/test2015",
        vqa_dir="data_vqaformat_9/vqa",
        split="test",
        coco_prefix="COCO_test2015",
        start_image_id=2000000
    )
