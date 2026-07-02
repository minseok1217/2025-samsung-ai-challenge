import os
import json
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, TrainingArguments, Trainer
import torch

# --- 1. 데이터셋 준비 ---
class VQADataset(torch.utils.data.Dataset):
    """VQA 데이터셋을 위한 커스텀 클래스"""
    def __init__(self, annotations_file, image_dir, processor):
        self.annotations = self._load_annotations(annotations_file)
        self.image_dir = image_dir
        self.processor = processor

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

        # 이미지 로드
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        
        # 정답 레이블을 토큰 ID로 변환
        # 모델이 '1', '2', '3', '4'를 예측하도록 학습
        labels = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding=True
        ).input_ids
        
        # 이미지와 질문을 모델 입력 형식으로 처리
        encoding = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
        )
        
        # 배치 처리를 위해 불필요한 차원 제거
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = labels.squeeze(0)
        
        return encoding

# --- 2. 모델 및 프로세서 로드 ---
print("Loading model and processor...")
# VQAv2로 사전 학습된 BEiT-3 모델 사용
model_name = "microsoft/beit3_base_patch16_224_vqav2"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVisualQuestionAnswering.from_pretrained(model_name)

# --- 3. 데이터셋 인스턴스 생성 ---
print("Preparing datasets...")
base_data_dir = "VMCBench_for_VQAv2"
train_dataset = VQADataset(
    annotations_file=os.path.join(base_data_dir, "annotations", "finetune_train.json"),
    image_dir=os.path.join(base_data_dir, "images", "train"),
    processor=processor
)
val_dataset = VQADataset(
    annotations_file=os.path.join(base_data_dir, "annotations", "finetune_val.json"),
    image_dir=os.path.join(base_data_dir, "images", "val"),
    processor=processor
)

# --- 4. 학습 설정 (Training Arguments) ---
# 파인튜닝 결과가 저장될 디렉토리
output_dir = "beit3-finetuned-vmcbench"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,  # 에포크 수 (GPU 성능에 따라 조절)
    per_device_train_batch_size=8,  # 배치 사이즈 (GPU VRAM에 따라 조절)
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch", # 한 에포크마다 평가 수행
    save_strategy="epoch",       # 한 에포크마다 모델 저장
    load_best_model_at_end=True,
    report_to="none", # wandb 등 로깅 서비스 사용 안 함
)

# --- 5. 트레이너(Trainer) 설정 및 학습 시작 ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting fine-tuning...")
trainer.train()

# 학습 완료 후 최종 모델 저장
print("Fine-tuning finished. Saving final model.")
trainer.save_model(os.path.join(output_dir, "final_model"))