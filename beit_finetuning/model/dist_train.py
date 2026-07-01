import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BeitModel, BertModel, BertTokenizer,
    get_scheduler
)
from tqdm import tqdm
from beit_dataset import VQAMultipleChoiceDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total Parameters: {total:,}")
    print(f"🧠 Trainable Parameters: {trainable:,}")
    return total, trainable

def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    choices = [item['choices'] for item in batch]  # list of list of strings
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

        # BEiT 이미지 임베딩
        image_outputs = self.image_encoder(pixel_values=images)
        image_embed = image_outputs.last_hidden_state[:, 0, :]  # [B, 768]

        logits = []
        for i in range(4):
            # 선택지 i번째 문장 리스트 (batch size만큼)
            texts = [choices_texts[j][i] for j in range(B)]

            encoding = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            text_outputs = self.text_encoder(**encoding)
            text_embed = text_outputs.last_hidden_state[:, 0, :]  # [B, 768]

            # 이미지와 텍스트 결합
            joint = torch.cat([image_embed, text_embed], dim=1)  # [B, 1536]
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


def train_model(train_csv, val_csv, output_dir, epochs, batch_size, lr, seed, num_gpus):
    set_seed(seed)


    # 제한된 GPU 사용 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, list(range(num_gpus))))
    print(f"✅ Visible GPUs set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device} | GPUs requested: {num_gpus}")

    # Dataset & Dataloader
    train_dataset = VQAMultipleChoiceDataset(train_csv, "train")
    val_dataset = VQAMultipleChoiceDataset(val_csv, "val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # Model
    base_model = BEiTMultipleChoiceModel()
    count_parameters(base_model)
    if torch.cuda.device_count() > 1:
        print(f"🔁 Multi-GPU enabled: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    model.to(device)

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

        val_acc, val_loss = evaluate(model, val_loader, device)
        print(f"📊 Epoch {epoch+1} - Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            ckpt_path = os.path.join(output_dir, "beit_best.pt")
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), ckpt_path)
            print(f"✅ Best model saved to: {ckpt_path}")

    print("🎉 Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./beit_ckpt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    train_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_gpus=args.num_gpus
    )
