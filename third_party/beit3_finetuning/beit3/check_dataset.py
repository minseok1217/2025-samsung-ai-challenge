""# check_dataset.py

import os
import json


def check_vqa_format(data_dir, split_name, anno_required=True):
    print(f"\n🔍 Checking {split_name}...")

    q_path = os.path.join(
        data_dir,
        f"v2_OpenEnded_mscoco_{split_name}2014_questions.json" if "test" not in split_name else f"v2_OpenEnded_mscoco_{split_name}2015_questions.json"
    )

    with open(q_path, "r") as f:
        questions = json.load(f)["questions"]

    print(f"총 질문 수: {len(questions)}")

    # annotation check
    annotations = []
    if anno_required:
        a_path = os.path.join(
            data_dir,
            f"v2_mscoco_{split_name}2014_annotations.json"
        )
        with open(a_path, "r") as f:
            annotations = json.load(f)["annotations"]
        print(f"총 annotation 수: {len(annotations)}")
    else:
        print("(test set: annotation 없음)")

    # 이미지 존재 확인
    missing_images = []
    image_ids_with_annos = set(a["image_id"] for a in annotations) if anno_required else set()
    multiple_choice_missing = 0
    img_dir = os.path.join(data_dir.replace("vqa", split_name + "2014" if "test" not in split_name else "test2015"))
    for q in questions:
        image_id = q["image_id"]
        image_name = f"COCO_{split_name}2014_{image_id:012d}.jpg" if "test" not in split_name else f"COCO_{split_name}2015_{image_id:012d}.jpg"
        image_path = os.path.join(img_dir, image_name)

        if not os.path.exists(image_path):
            missing_images.append(image_name)

        if anno_required and image_id not in image_ids_with_annos:
            print(f"❗ annotation 누락된 image_id: {image_id}")

    # multiple_choice_answer 누락 확인
    if anno_required:
        for a in annotations:
            if "multiple_choice_answer" not in a:
                multiple_choice_missing += 1

    if missing_images:
        print(f"❌ 이미지 없음: {len(missing_images)}")
        print(f"예시: {missing_images[:3]}")
    else:
        print("✅ 모든 이미지 존재")

    if anno_required:
        if multiple_choice_missing == 0:
            print("✅ 모든 annotation에 multiple_choice_answer 존재")
        else:
            print(f"❌ multiple_choice_answer 누락 수: {multiple_choice_missing}")

    print("✅ 완료")


if __name__ == "__main__":
    base_dir = "data_vqaformat_9/vqa"  # ← 기존 7에서 8로 변경

    check_vqa_format(base_dir, "train")
    check_vqa_format(base_dir, "val")
    check_vqa_format(base_dir, "test", anno_required=False)
    check_vqa_format(base_dir, "test-dev", anno_required=False)
