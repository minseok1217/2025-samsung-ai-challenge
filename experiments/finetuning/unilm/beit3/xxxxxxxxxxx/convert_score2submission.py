import json
import os
from collections import defaultdict

def generate_retrieval_answers_for_all_images(score_file_path, output_csv_path="submission_all.csv"):
    """
    주어진 그룹화된 점수 파일 (score_v2.json 형식)을 기반으로
    파일에 존재하는 모든 이미지에 대해 이미지-텍스트 검색 결과를 생성합니다.

    Args:
        score_file_path (str): 이미지 ID별로 그룹화된 점수 및 토큰이 저장된 JSON 파일 경로.
                                (예: score_v2.json)
        output_csv_path (str): 결과를 저장할 CSV 파일 경로.
    """

    # 1. 그룹화된 점수 파일 로드
    try:
        with open(score_file_path, 'r') as f:
            grouped_scores_data = json.load(f)
        print(f"'{score_file_path}' 파일 로드 완료. 총 이미지 ID: {len(grouped_scores_data)}")
    except FileNotFoundError:
        print(f"오류: '{score_file_path}' 파일을 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{score_file_path}' 파일의 JSON 형식이 올바르지 않습니다. 내용을 확인해주세요.")
        return

    # 2. 각 이미지에 대한 답변 찾기
    answers = {}
    answer_chars = ['A', 'B', 'C', 'D']

    for image_id_str, results_list in grouped_scores_data.items():
        # image_id_str은 문자열 ("0", "1" 등)일 수 있으므로 int로 변환
        image_id = int(image_id_str) 

        # 각 이미지에 4개의 텍스트 보기가 있는지 확인
        if len(results_list) != 4:
            print(f"경고: Image ID {image_id}에 4개의 텍스트 보기가 없습니다. {len(results_list)}개 발견됨. 건너뜀.")
            answers[f"TEST_{image_id:03d}"] = "-"
            continue

        # 가장 높은 점수를 가진 텍스트 찾기
        max_score = -float('inf')
        best_answer_index = -1

        # results_list는 이미 A, B, C, D 순서대로 점수가 저장되어 있다고 가정 (eval_batch의 append 순서)
        for i, item in enumerate(results_list):
            # score는 [[-0.0008416175842285156]] 형태이므로 첫 번째 요소를 가져옴
            current_score = item['score'][0][0] 
            
            if current_score > max_score:
                max_score = current_score
                best_answer_index = i
        
        if best_answer_index != -1:
            answers[f"TEST_{image_id:03d}"] = answer_chars[best_answer_index]
        else:
            answers[f"TEST_{image_id:03d}"] = "-" # 이 경우는 발생하지 않아야 함

    # 3. 결과 CSV 파일 작성
    output_lines = ["ID,answer"]
    
    # score_v2.json 파일에 있는 모든 이미지 ID를 가져와 정렬
    all_present_image_ids = sorted([int(k) for k in grouped_scores_data.keys()])

    for numeric_id in all_present_image_ids:
        img_id_formatted = f"TEST_{numeric_id:03d}"
        output_lines.append(f"{img_id_formatted},{answers.get(img_id_formatted, '-')}")

    try:
        with open(output_csv_path, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"결과가 '{output_csv_path}' 파일에 성공적으로 저장되었습니다.")
    except IOError:
        print(f"오류: '{output_csv_path}' 파일에 쓸 수 없습니다. 경로를 확인해주세요.")

# --- 실행 예시 ---
# 실제 파일 경로로 교체해주세요.
score_file = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/score_v2_flickr30k_large.json"
# 출력 파일명도 'submission_all.csv' 등으로 변경하여 이전 파일과 구분할 수 있습니다.
output_submission_file = "score_v2_flickr30k_large.csv" 

generate_retrieval_answers_for_all_images(score_file, output_submission_file)