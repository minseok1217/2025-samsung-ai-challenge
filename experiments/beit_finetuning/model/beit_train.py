import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from transformers import (
    BeitModel, BeitFeatureExtractor,
    BertModel, BertTokenizer,
    get_scheduler
)
from tqdm import tqdm
from beit_dataset import VQAMultipleChoiceDataset

# ✅ 1. Seed 고정 함수
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ✅ 2. collate_fn 정의
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    choices = [item['choices'] for item in batch]  # List[List[str]]
    return {'image': images, 'choices': choices, 'label': labels}


class BEiTMultipleChoiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )

    def forward(self, images, choices_texts):
        B = images.size(0)
        device = images.device

        image_outputs = self.image_encoder(pixel_values=images)
        image_embed = image_outputs.last_hidden_state[:, 0, :]  # [B, 768]

        logits = []
        for i in range(4):
            texts = [c[i] for c in choices_texts]
            encoding = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            text_outputs = self.text_encoder(**encoding)
            text_embed = text_outputs.last_hidden_state[:, 0, :]  # [B, 768]

            joint = torch.cat([image_embed, text_embed], dim=1)
            logit = self.classifier(joint)  # [B, 1]
            logits.append(logit)

        logits = torch.cat(logits, dim=1)  # [B, 4]
        return logits


def evaluate(model, dataloader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch["image"].to(device)
            choices = batch["choices"]
            labels = batch["label"].to(device)

            logits = model(images, choices)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item()
    acc = correct / total
    return acc, loss_sum / total


def train_model(train_csv, val_csv, output_dir="ckpt", epochs=3, batch_size=2, lr=5e-5, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device} | Seed: {seed}")

    # Datasets
    train_dataset = VQAMultipleChoiceDataset(train_csv, mode="train")
    val_dataset = VQAMultipleChoiceDataset(val_csv, mode="val")

    # DataLoaders with collate_fn
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=custom_collate_fn, worker_init_fn=lambda _: set_seed(seed)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Model & Optimizer
    model = BEiTMultipleChoiceModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            images = batch["image"].to(device)
            choices = batch["choices"]
            labels = batch["label"].to(device)

            logits = model(images, choices)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        # Validation
        val_acc, val_loss = evaluate(model, val_loader, device)
        print(f"📊 Epoch {epoch+1} - Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            ckpt_path = os.path.join(output_dir, f"beit_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Best model saved to: {ckpt_path}")

    print("🎉 Training complete.")


if __name__ == "__main__":
    train_model(
        train_csv="/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit_finetuning/data/visual7w_mcqa_randomized_3.csv",
        val_csv="/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit_finetuning/data/val.csv",
        output_dir="./beit_ckpt",
        epochs=5,
        batch_size=16,
        lr=5e-5,
        seed=42
    )
