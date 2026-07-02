import json
import pandas as pd
import os
import numpy as np

# --- 설정 ---
# 이 경로들을 실제 환경에 맞게 수정해주세요.
INPUT_JSON_PATH = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/output_cross_entropy_inference/evaluation_logits.json"      
OUTPUT_CSV_PATH = "./submission.csv"

# --- 예제 데이터 생성 ---
# 실제 파일과 동일한, 리스트가 중첩된 구조의 예제 파일을 만듭니다.
# nested_json_data = [
#     # 첫 번째 배치
#     [
#         [-0.5, 1.2, -0.3, 0.8], # -> B
#         [1.1, 2.1, 0.1, 3.5], # -> D
#     ],
#     # 두 번째 배치
#     [
#         [9.9, -1.0, 5.5, 3.2], # -> A
#     ]
# ]
# with open(INPUT_JSON_PATH, 'w') as f:
#     json.dump(nested_json_data, f, indent=4)


# --- 변환 로직 (수정됨) ---

def create_submission_from_logits(input_path, output_path):
    """
    중첩된 리스트 형식의 로짓 JSON 파일을 읽어, 
    정답을 예측하고 CSV 파일을 생성합니다.
    """
    try:
        with open(input_path, 'r') as f:
            # data는 이제 [ [...], [...], ... ] 형태의 리스트가 됩니다.
            data = json.load(f)
        print(f"✅ '{input_path}'에서 {len(data)}개의 배치를 성공적으로 불러왔습니다.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 오류: '{input_path}' 파일을 불러오거나 파싱할 수 없습니다. 원인: {e}")
        return

    # ▼▼▼ 핵심 수정 부분 ▼▼▼
    # 중첩된 리스트(list of lists)를 하나의 평탄한 리스트로 만듭니다.
    # 예: [ [A, B], [C] ] -> [A, B, C]
    flattened_data = [item for batch in data for item in batch]
    # ▲▲▲ 수정 완료 ▲▲▲
    
    answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    results = []
    
    print(f"🚀 총 {len(flattened_data)}개의 예측 결과를 찾았습니다. 처리를 시작합니다...")

    # 평탄화된 리스트를 순회합니다.
    for i, logit_vector in enumerate(flattened_data):
        if not isinstance(logit_vector, list) or len(logit_vector) != 4:
            print(f"⚠️ {i}번째 항목이 숫자 4개로 이루어진 리스트가 아니므로 건너뜁니다.")
            continue
        
        # 각 로짓 벡터에서 가장 큰 값의 인덱스를 찾습니다.
        best_choice_index = np.argmax(logit_vector)
        
        # 인덱스를 정답 문자(A, B, C, D)로 변환합니다.
        predicted_answer = answer_map[best_choice_index]
        
        # ID를 생성합니다.
        test_id = f"TEST_{i:03d}"
        
        results.append({
            'ID': test_id,
            'answer': predicted_answer
        })

    # 결과를 데이터프레임으로 변환하고 CSV로 저장합니다.
    output_df = pd.DataFrame(results)
    try:
        output_df.to_csv(output_path, index=False)
        print(f"\n✅ 변환 완료! '{output_path}'에 제출 파일이 저장되었습니다.")
        print("결과 미리보기:")
        print(output_df.head())
    except IOError as e:
        print(f"❌ 오류: '{output_path}' 파일 저장에 실패했습니다. 원인: {e}")

# --- 스크립트 실행 ---
if __name__ == '__main__':
    create_submission_from_logits(INPUT_JSON_PATH, OUTPUT_CSV_PATH)