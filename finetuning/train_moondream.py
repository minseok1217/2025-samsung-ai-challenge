import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, CodeGenTokenizerFast, get_scheduler

# --- 1. 데이터셋 클래스 (이전과 동일) ---
class MoondreamVQADataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, image_dir, tokenizer):
        self.annotations = self._load_annotations(annotations_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer

    def _load_annotations(self, annotations_file):
        with open(annotations_file, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        question = annotation['question']
        answer = annotation['answer']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        prompt = f"Question: {question}\n\nAnswer: {answer}{self.tokenizer.eos_token}"
        return {"image": image, "prompt": prompt}

# --- 2. 데이터 콜레이터 클래스 (이전과 동일) ---
class MoondreamDataCollator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, examples):
        images = [ex["image"] for ex in examples]
        prompts = [ex["prompt"] for ex in examples]
        image_embeds = torch.cat([self.model.encode_image(image) for image in images])
        text_tokens = self.tokenizer(prompts, padding='longest', return_tensors="pt")
        return {"image_embeds": image_embeds, **text_tokens}

# --- 3. 모델 및 토크나이저 로드 ---
print("Loading Moondream1 model and tokenizer...")
model_id = "vikhyatk/moondream1"
tokenizer = CodeGenTokenizerFast.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# --- 4. 데이터셋 및 데이터로더 준비 ---
print("Preparing datasets...")
base_data_dir = "VMCBench_for_VQAv2"
train_dataset = MoondreamVQADataset(
    annotations_file=os.path.join(base_data_dir, "annotations", "finetune_train.json"),
    image_dir=os.path.join(base_data_dir, "images", "train"),
    tokenizer=tokenizer
)
data_collator = MoondreamDataCollator(model, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)

# --- 5. 학습 설정 ---
num_epochs = 3
learning_rate = 1e-5
output_dir = "moondream1-finetuned-vmcbench-manual"
os.makedirs(output_dir, exist_ok=True)

# 옵티마이저 및 스케줄러 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=num_training_steps,
)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- 6. PyTorch 학습 루프 ---
print("Starting fine-tuning...")
model.train()
for epoch in range(num_epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for batch in progress_bar:
        # 배치를 GPU로 이동
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 모델 순전파 및 손실 계산
        outputs = model(
            image_embeds=batch["image_embeds"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        loss = outputs.loss

        # 역전파 및 파라미터 업데이트
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # 진행률 표시줄에 손실 표시
        progress_bar.set_postfix({"loss": loss.item()})

# --- 7. 모델 저장 ---
print("Fine-tuning finished. Saving final model.")
model.save_pretrained(os.path.join(output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))