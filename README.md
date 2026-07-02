# 2025 Samsung AI Challenge

Samsung AI Challenge 참가 과정에서 진행한 멀티모달 VQA(Visual Question Answering) 실험 기록과 제출 산출물을 정리한 저장소입니다.

## 개요

이 프로젝트는 이미지와 질문을 함께 입력받아 객관식 답변을 추론하는 문제를 대상으로 여러 비전-언어 모델과 LLM 기반 후처리 방법을 비교한 작업 공간입니다. BLIP, CLIP, BEiT3, InstructBLIP, OpenFlamingo, PaLI/Gemma, LLaMA 계열 전처리 및 LLM 기반 앙상블 실험이 포함되어 있습니다.

## Task

본 대회 task는 VQA(Visual Question Answering) 기반 multi-choice 문제입니다. 하나의 이미지와 질문, 4개의 선택지가 주어졌을 때 모델은 이미지와 질문의 맥락을 함께 이해해 가장 적절한 선택지 하나를 고릅니다.

### Example

이미지 입력과 함께 아래와 같은 질문 및 선택지가 제공됩니다.

```text
Question. What might the landscape look like in the next season?

Choice A. The flowers would be blooming with vibrant colors
Choice B. The weather would be hot with sunny days and clear skies
Choice C. The area would be covered in snow with a white blanket
Choice D. The trees would be full of leaves and greenery all around
```

정답 형식은 선택지 중 하나입니다.

## 폴더 구조

```text
.
|-- README.md
|-- docs/
|   `-- PROJECT_STRUCTURE.md
|-- experiments/
|   |-- base_line/
|   |-- beit/
|   |-- beit_finetuning/
|   |-- blip_2/
|   |-- clip/
|   |-- clip_gpt/
|   |-- finetuning/
|   |-- gpt/
|   |-- instruct_blip/
|   |-- llama_32_11b/
|   |-- llm/
|   |-- open_flamingo/
|   |-- PALI/
|   |-- preprocess_with_llama/
|   `-- sentence_bert.ipynb
|-- submissions/
`-- third_party/
```

## 주요 실험

| 경로 | 내용 |
| --- | --- |
| `experiments/base_line/` | 대회 baseline, BLIP 계열 초기 제출 실험 |
| `experiments/clip_gpt/` | CLIP 결과와 GPT 기반 추론/후처리 결합 실험 |
| `experiments/finetuning/` | VQA 포맷 변환, fine-tuning 데이터 구성, BEiT3/BLIP fine-tuning 실험 |
| `experiments/gpt/` | 2-stage GPT 추론 및 제출 파일 생성 실험 |
| `experiments/instruct_blip/` | InstructBLIP 기반 추론 실험 |
| `experiments/llama_32_11b/` | LLaMA 3.2 11B 기반 pseudo label 및 제출 실험 |
| `experiments/llm/` | LLM 계열 추론 결과, 후처리, 최종 제출 후보 |
| `experiments/open_flamingo/` | OpenFlamingo 기반 추론 실험 |
| `experiments/PALI/` | PaLI/Gemma 기반 추론 실험 |
| `experiments/preprocess_with_llama/` | LLaMA 기반 caption/VQA 전처리 실험 |

## 외부 코드

`third_party/`에는 실험에 참고하거나 일부 수정해 사용한 외부 모델 저장소가 모여 있습니다. 이 폴더는 프로젝트의 핵심 실험 기록과 분리하기 위한 공간이며, 각 저장소의 자세한 사용법과 라이선스는 해당 폴더 내부 README를 확인하면 됩니다.

## 제출 산출물

최종 제출 후보로 보이는 CSV 파일은 `submissions/`에 모델명 접두어를 붙여 복사해 두었습니다. 원본 파일은 각 실험 폴더에도 남겨 두어 어떤 실험에서 생성된 결과인지 추적할 수 있습니다.

## 사용 흐름

1. `experiments/base_line/`에서 baseline 구조와 제출 형식을 확인합니다.
2. 모델별 실험 폴더에서 노트북을 실행하거나 결과 파일을 비교합니다.
3. `experiments/llm/` 또는 각 모델별 폴더의 후처리 결과를 확인합니다.
4. 제출 후보는 `submissions/`에서 모아 비교합니다.

## 환경

실험별 의존성이 서로 다릅니다. 공통 환경 파일 하나로 고정하기보다는 각 모델/외부 저장소의 `requirements.txt`, `pyproject.toml`, `environment.yml`을 기준으로 별도 환경을 구성하는 편이 안전합니다.

## 참고

- 대용량 모델 코드와 실험 노트북이 함께 들어 있어 저장소가 큽니다.
- `third_party/` 내부 코드는 외부 프로젝트 기반이므로, 수정 전 원본 README와 라이선스를 확인하는 것을 권장합니다.
- 노트북 내부 상대 경로는 기존 작업 위치를 기준으로 작성되었을 수 있습니다. 폴더 이동 이후 실행 시 필요한 경로만 현재 구조에 맞게 조정하면 됩니다.
