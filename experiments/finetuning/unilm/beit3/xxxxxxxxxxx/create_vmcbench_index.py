import os
from datasets import VQAv2Dataset # VQAv2Dataset 클래스가 BEiT-3 프로젝트 내에 있다고 가정합니다.
from transformers import XLMRobertaTokenizer

# --- 설정 (Configuration) ---
# VMCBench 데이터를 변환하여 저장한 루트 경로 (변환 스크립트의 base_dir과 동일)
VMCBENCH_PROCESSED_DATA_ROOT = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/VMCBench_as_Official_VQAv2_9_1"

# BEiT-3 SentencePiece 모델의 경로
BEIT3_SPM_PATH = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/unilm/beit3/beit3.spm"

# 데이터셋 인덱스 파일이 생성될 경로 (보통 데이터 경로와 동일하게 설정)
INDEX_OUTPUT_PATH = VMCBENCH_PROCESSED_DATA_ROOT

# --- 경로 유효성 검사 ---
if not os.path.exists(VMCBENCH_PROCESSED_DATA_ROOT):
    print(f"오류: 데이터 루트 경로를 찾을 수 없습니다: {VMCBENCH_PROCESSED_DATA_ROOT}")
    print("VMCBench 변환 스크립트를 먼저 실행하여 데이터를 생성했는지 확인하세요.")
    exit(1)

# 이미지 폴더 경로 검증 (VMCBench_as_Official_VQAv2/train2014 형태)
if not os.path.exists(os.path.join(VMCBENCH_PROCESSED_DATA_ROOT, "train2014")):
    print(f"오류: 이미지 폴더를 찾을 수 없습니다. '{VMCBENCH_PROCESSED_DATA_ROOT}/train2014'를 확인하세요.")
    exit(1)

# 어노테이션 JSON 파일 경로 검증 (VMCBench_as_Official_VQAv2/vqa/v2_OpenEnded_mscoco_train2014_questions.json 형태)
if not os.path.exists(os.path.join(VMCBENCH_PROCESSED_DATA_ROOT, "vqa", "v2_OpenEnded_mscoco_train2014_questions.json")):
    print(f"오류: 질문 JSON 파일을 찾을 수 없습니다. '{os.path.join(VMCBENCH_PROCESSED_DATA_ROOT, 'vqa', 'v2_OpenEnded_mscoco_train2014_questions.json')}'를 확인하세요.")
    exit(1)

if not os.path.exists(BEIT3_SPM_PATH):
    print(f"오류: SentencePiece 모델 파일을 찾을 수 없습니다: {BEIT3_SPM_PATH}")
    print("BEiT-3 모델 경로에서 'beit3.spm' 파일을 다운로드했는지 확인하세요.")
    exit(1)

# --- 토크나이저 로드 ---
print(f"SentencePiece 토크나이저 로드 중: {BEIT3_SPM_PATH}")
try:
    tokenizer = XLMRobertaTokenizer(BEIT3_SPM_PATH)
    print("토크나이저 로드 성공.")
except Exception as e:
    print(f"토크나이저 로드 중 오류 발생: {e}")
    print("XLMRobertaTokenizer 설치 및 'beit3.spm' 파일 경로를 확인하세요.")
    exit(1)

# --- 데이터셋 인덱스 생성 ---
print("\nVQAv2Dataset.make_dataset_index 실행 중...")
try:
    VQAv2Dataset.make_dataset_index(
        data_path=VMCBENCH_PROCESSED_DATA_ROOT, # train2014, val2014, vqa 폴더를 포함하는 최상위 경로
        tokenizer=tokenizer,
        annotation_data_path=os.path.join(VMCBENCH_PROCESSED_DATA_ROOT, "vqa"), # vqa 폴더 자체
    )
    print("\n데이터셋 인덱스 생성 완료!")
    print(f"생성된 인덱스 파일은 다음 경로에 위치할 수 있습니다: {INDEX_OUTPUT_PATH}")
    print("예: vqa.train.jsonl.tsv, vqa.val.jsonl.tsv")

except AttributeError:
    print("\n오류: VQAv2Dataset 클래스에 'make_dataset_index' 메서드가 없거나,")
    print("해당 메서드가 현재 환경에서 임포트되지 않았습니다.")
    print("BEiT-3 프로젝트의 `datasets/__init__.py` 또는 `datasets/vqav2_dataset.py` 파일을 확인하여")
    print("`VQAv2Dataset`가 올바르게 정의되어 있는지 확인하세요.")
    print("또는 `from beit3.datasets import VQAv2Dataset`와 같이 특정 모듈에서 임포트해야 할 수 있습니다.")
except Exception as e:
    print(f"\n데이터셋 인덱스 생성 중 예상치 못한 오류 발생: {e}")