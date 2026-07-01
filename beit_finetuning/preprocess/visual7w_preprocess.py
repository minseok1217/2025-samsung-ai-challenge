import json
import pandas as pd
import random
from pathlib import Path

# 🔹 설정
json_path = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/visual7w/dataset_v7w_telling.json"
image_root = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/visual7w/images"  # 실제 이미지 경로
random.seed(42)  # 랜덤 시드 고정

# 🔹 데이터 로드
with open(json_path, "r") as f:
    raw_data = json.load(f)

rows = []

# 🔹 반복 처리
for image in raw_data["images"]:
    for qa in image["qa_pairs"]:
        image_id = str(qa["image_id"])  # ✅ 각 qa_pair의 image_id 사용
        image_path = str(Path(image_root) / f"{image_id}.jpg")

        question = qa["question"]
        answer = qa["answer"]
        choices = qa["multiple_choices"].copy()

        # 정답 추가
        if answer not in choices:
            choices.append(answer)

        # 중복 제거 후 정답 포함 4지선다 만들기
        choices = list(set(choices))
        if len(choices) < 4:
            continue
        elif len(choices) > 4:
            choices = random.sample([c for c in choices if c != answer], 3)
            choices.append(answer)

        # 정답 위치 섞기
        random.shuffle(choices)
        label_idx = choices.index(answer)
        label = ["A", "B", "C", "D"][label_idx]

        row = {
            "id": qa["qa_id"],
            "image_path": f'v7w_{qa["image_id"]}.jpg',  # 이미지 ID로 경로 설정
            "question": question,
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
            "label": label
        }
        rows.append(row)


# 🔹 저장
df = pd.DataFrame(rows)
df.to_csv("/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit3_finetuning/beit3/data/train_2.csv", index=False)

print(f"총 {len(df)}개 샘플 저장 완료 (정답 랜덤 배치, 시드=42)")
