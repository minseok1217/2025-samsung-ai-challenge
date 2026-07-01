import torch
import torch.nn as nn
from transformers import BeitModel, BeitFeatureExtractor, BertModel, BertTokenizer
from PIL import Image


class BEiTMultipleChoiceModel(nn.Module):
    def __init__(self,
                 beit_model_name="microsoft/beit-base-patch16-224",
                 text_encoder_name="bert-base-uncased",
                 hidden_dim=768):
        super().__init__()
        # 이미지 인코더 (BEiT)
        self.image_encoder = BeitModel.from_pretrained(beit_model_name)
        # 텍스트 인코더 (BERT)
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        self.tokenizer = BertTokenizer.from_pretrained(text_encoder_name)

        # 최종 분류기: [이미지 CLS + 텍스트 CLS] → MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 선택지당 1개의 score
        )

    def forward(self, images, choices_texts):
        """
        images: Tensor [B, 3, 224, 224]
        choices_texts: List[List[str]] of shape [B, 4]
        """
        B = images.size(0)
        device = images.device

        # 1. 이미지 임베딩
        image_outputs = self.image_encoder(pixel_values=images)
        image_embed = image_outputs.last_hidden_state[:, 0, :]  # CLS token [B, 768]

        logits = []
        for i in range(4):  # 각 선택지에 대해
            texts = [c[i] for c in choices_texts]
            encoding = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

            text_outputs = self.text_encoder(**encoding)
            text_embed = text_outputs.last_hidden_state[:, 0, :]  # CLS token

            # 이미지와 텍스트 결합
            joint = torch.cat([image_embed, text_embed], dim=1)  # [B, 1536]
            logit = self.classifier(joint)  # [B, 1]
            logits.append(logit)

        logits = torch.cat(logits, dim=1)  # [B, 4]
        return logits


def preprocess_image(image_path):
    feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")
    image = Image.open(image_path).convert("RGB")
    return feature_extractor(images=image, return_tensors="pt")["pixel_values"]  # [1, 3, 224, 224]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 예시 입력
    image_path = "sample.jpg"  # 🔁 실제 이미지 경로로 교체
    choices = [["A dog is sitting.", "A cat is running.", "A man is walking.", "A bird is flying."]]

    image_tensor = preprocess_image(image_path).to(device)
    model = BEiTMultipleChoiceModel().to(device)
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor, choices)
        pred = torch.argmax(logits, dim=1)

    print(f"Predicted label: {['A', 'B', 'C', 'D'][pred.item()]}")
    print("Logits:", logits.cpu().numpy())


if __name__ == "__main__":
    main()
