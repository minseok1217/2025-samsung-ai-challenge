import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from transformers import BeitFeatureExtractor
import os

class VQAMultipleChoiceDataset(Dataset):
    def __init__(self, csv_path, mode, feature_extractor_name="microsoft/beit-base-patch16-224", transform=None):
        self.data = pd.read_csv(csv_path)        
        self.mode = mode  # 'train', 'val', 'test' 등
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(feature_extractor_name)
        self.label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.transform = transform  # Optional, if using custom torchvision transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.mode == 'train':
            image_name = f'v7w_{row["image_path"]}.jpg'  # Assuming 'id' column contains the image name without path
            image_path = os.path.join("/data/2_data_server/cv-07/dice/2025_samsung_challenge/dataset/visual7w/images", image_name)  # 이미지 경로 조합
        elif self.mode == 'val':
            image_path = os.path.join("/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit_finetuning/data/train_input_images", row["image_path"])
        else:
            image_path = os.path.join("/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit_finetuning/data/test_input_images", row["image_path"])
        question = row["question"]
        choices = [row["A"], row["B"], row["C"], row["D"]]
        label = self.label_map[row["label"]]

        # 이미지 불러오기
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)  # [3, 224, 224]

        # 텍스트 입력: 질문 + 선택지
        combined_choices = [f"{question} {opt}" for opt in choices]  # 4개 문장

        return {
            "image": image,
            "choices": combined_choices,  # List[str] of len 4
            "label": torch.tensor(label, dtype=torch.long)
        }
