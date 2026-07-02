# Project Structure

이 문서는 저장소의 현재 폴더 역할을 빠르게 확인하기 위한 보조 문서입니다.

## Top-Level

| 경로 | 역할 |
| --- | --- |
| `README.md` | 프로젝트 소개, 주요 실험, 사용 흐름 |
| `docs/` | 문서 및 정리 자료 |
| `experiments/` | 직접 실행한 실험 노트북, 전처리 코드, 모델별 결과 |
| `submissions/` | 최종 제출 후보 CSV 모음 |
| `third_party/` | 외부 모델 저장소 및 참고 코드 |

## Experiments

| 경로 | 역할 |
| --- | --- |
| `experiments/base_line/` | 대회 baseline 및 초기 BLIP 실험 |
| `experiments/beit/` | BEiT 관련 테스트 |
| `experiments/beit_finetuning/` | BEiT fine-tuning 관련 코드와 노트북 |
| `experiments/blip_2/` | BLIP-2 실험 |
| `experiments/clip/` | CLIP 단독 테스트 |
| `experiments/clip_gpt/` | CLIP + GPT 조합 실험 |
| `experiments/finetuning/` | 데이터 변환 및 fine-tuning 실험 |
| `experiments/gpt/` | GPT 기반 2-stage 실험 |
| `experiments/instruct_blip/` | InstructBLIP 실험 |
| `experiments/llama_32_11b/` | LLaMA 3.2 11B 기반 pseudo label/제출 실험 |
| `experiments/llm/` | LLM 추론, 후처리, 최종 제출 후보 |
| `experiments/open_flamingo/` | OpenFlamingo 실험 |
| `experiments/PALI/` | PaLI/Gemma 실험 |
| `experiments/preprocess_with_llama/` | LLaMA 기반 caption/VQA 전처리 |
| `experiments/sentence_bert.ipynb` | Sentence-BERT 관련 단일 노트북 |

## Third Party

| 경로 | 역할 |
| --- | --- |
| `third_party/ICCV19_VQA-CTI/` | VQA-CTI 참고 코드 |
| `third_party/LAVIS/` | Salesforce LAVIS 기반 코드 |
| `third_party/LLaVA/` | LLaVA 기반 코드 |
| `third_party/OFA/` | OFA 기반 코드 |
| `third_party/VCR/` | VCR/Hugging Face 관련 참고 실험 |
| `third_party/unilm/` | Microsoft UniLM 계열 모델 코드 |
| `third_party/beit3_finetuning/` | BEiT3/UniLM 계열 fine-tuning 참고 코드 묶음 |

## Submissions

`submissions/`에는 각 실험 폴더에서 생성된 제출 후보 CSV를 찾기 쉽게 복사해 두었습니다. 원본 산출물은 실험 맥락을 보존하기 위해 `experiments/` 내부에도 남아 있습니다.
