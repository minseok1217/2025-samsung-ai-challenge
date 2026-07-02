from torch.utils.data import DataLoader
from beit_dataset import VQAMultipleChoiceDataset

dataset = VQAMultipleChoiceDataset("/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit_finetuning/data/visual7w_mcqa_randomized_3.csv", "train")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 예시 배치 하나 가져오기
for batch in dataloader:
    images = batch["image"]           # [B, 3, 224, 224]
    choices = batch["choices"]        # List[List[str]] of shape [B, 4]
    labels = batch["label"]           # [B]

    print(images.shape)
    print(choices[0])  # 첫 샘플의 4지선다
    print(labels)
    break
