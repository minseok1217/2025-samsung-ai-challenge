import os
import pandas as pd
from shutil import copyfile

def process_split(split_name, csv_path, image_root, output_root, coco_prefix, start_id=100000, save_csv=False):
    print(f"\n🔧 Processing split: {split_name.upper()}")
    df = pd.read_csv(csv_path, dtype={"id": str})
    os.makedirs(output_root, exist_ok=True)

    image_ids = []
    new_image_paths = []

    for idx, row in df.iterrows():
        image_id = start_id + idx
        image_ids.append(image_id)

        # 새 파일명 생성
        new_name = f"{coco_prefix}_{image_id:012d}.jpg"
        new_image_paths.append(new_name)

        # 원본 경로
        src_path = row["image_path"].replace("./", "")
        src_full = os.path.join(image_root, os.path.basename(src_path))
        dst_full = os.path.join(output_root, new_name)

        # 복사
        if os.path.exists(src_full):
            copyfile(src_full, dst_full)
        else:
            print(f"❗ 이미지 없음: {src_full}")

    # 결과 저장 (선택)
    df["image_id"] = image_ids
    df["coco_image"] = new_image_paths

    if save_csv:
        csv_out_path = os.path.join(os.path.dirname(csv_path), f"{split_name}_with_imageid.csv")
        df.to_csv(csv_out_path, index=False)
        print(f"📄 csv 저장 완료: {csv_out_path}")

    print(f"✅ {split_name} 완료 → 총 {len(df)}개 이미지 리네이밍됨")

# 사용 예시
process_split(
    split_name="train",
    csv_path="data/train.csv",
    image_root="data/train_input_images",
    output_root="data/images",
    coco_prefix="v7w",  # train은 원래 이름 유지, 복사만 하고 image_id 기록
    start_id=0,
    save_csv=True
)

process_split(
    split_name="val",
    csv_path="data/val.csv",
    image_root="data/val_input_images",
    output_root="data/images",
    coco_prefix="COCO_val2014",
    start_id=100000,
    save_csv=True
)

process_split(
    split_name="test",
    csv_path="data/test.csv",
    image_root="data/test_input_images",
    output_root="data/images",
    coco_prefix="COCO_test2014",
    start_id=200000,
    save_csv=True
)
