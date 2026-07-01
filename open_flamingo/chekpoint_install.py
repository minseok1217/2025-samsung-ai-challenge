from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
    cross_attn_every_n_layers=1
)

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

import torch

def count_parameters(model):
    """
    PyTorch 모델의 전체 파라미터 수와 학습 가능한 파라미터 수를 계산하고 출력합니다.
    """
    # 전체 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    
    # 학습 가능한 파라미터 수 계산 (requires_grad=True 인 파라미터만)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"전체 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    return total_params, trainable_params

# 모델의 파라미터 수를 계산하고 출력
count_parameters(model)