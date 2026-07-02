from datasets import NLVR2Dataset
from transformers import XLMRobertaTokenizer

# === 경로 수정 필수 ===
your_beit3_model_path = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit3_finetuning/beit3" # 1단계에서 다운로드한 모델/토크나이저 경로
your_data_path = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit3_finetuning/beit3/nlvr2" # 1단계에서 구성한 데이터셋 경로
# ======================

# 토크나이저 로드
tokenizer = XLMRobertaTokenizer("/data/2_data_server/cv-07/dice/2025_samsung_challenge/beit3_finetuning/beit3/beit3.spm")

# 인덱스 파일 생성
NLVR2Dataset.make_dataset_index(
    data_path=your_data_path,
    tokenizer=tokenizer,
    nlvr_repo_path=f"{your_data_path}/nlvr"
)

print("인덱스 파일 생성이 완료되었습니다.")