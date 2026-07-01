import json
import os
from collections import defaultdict

def generate_combined_retrieval_answers(score_file_path1, score_file_path2, output_csv_path="submission_combined.csv"):
    """
    주어진 두 개의 그룹화된 점수 파일(score_v2.json 형식)을 기반으로,
    두 파일의 점수를 비교하여 가장 높은 점수를 가진 보기를 선택하고
    이미지-텍스트 검색 결과를 생성합니다.

    Args:
        score_file_path1 (str): 첫 번째 점수 및 토큰이 저장된 JSON 파일 경로.
        score_file_path2 (str): 두 번째 점수 및 토큰이 저장된 JSON 파일 경로.
        output_csv_path (str): 결과를 저장할 CSV 파일 경로.
    """

    # 1. 두 개의 점수 파일 로드
    try:
        with open(score_file_path1, 'r') as f:
            grouped_scores_data1 = json.load(f)
        print(f"'{score_file_path1}' 파일 로드 완료. 총 이미지 ID: {len(grouped_scores_data1)}")
    except FileNotFoundError:
        print(f"오류: '{score_file_path1}' 파일을 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{score_file_path1}' 파일의 JSON 형식이 올바르지 않습니다.")
        return

    try:
        with open(score_file_path2, 'r') as f:
            grouped_scores_data2 = json.load(f)
        print(f"'{score_file_path2}' 파일 로드 완료. 총 이미지 ID: {len(grouped_scores_data2)}")
    except FileNotFoundError:
        print(f"오류: '{score_file_path2}' 파일을 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{score_file_path2}' 파일의 JSON 형식이 올바르지 않습니다.")
        return

    # 2. 각 이미지에 대한 최적의 답변 찾기
    answers = {}
    answer_chars = ['A', 'B', 'C', 'D']

    # 두 파일에 있는 모든 이미지 ID를 통합 (중복 제거)
    all_image_ids = set(grouped_scores_data1.keys()) | set(grouped_scores_data2.keys())
    
    print(f"두 파일에서 총 {len(all_image_ids)}개의 고유한 이미지 ID를 처리합니다.")

    for image_id_str in all_image_ids:
        image_id = int(image_id_str)

        max_score = -float('inf')
        best_answer_index = -1

        # 각 파일에서 점수 리스트를 가져옴. ID가 없으면 빈 리스트로 처리.
        results_list1 = grouped_scores_data1.get(image_id_str, [])
        results_list2 = grouped_scores_data2.get(image_id_str, [])

        # 두 리스트를 합쳐서 모든 점수를 한 번에 비교
        combined_results = [
            {'source_file': 1, 'index': i, 'score': item['score'][0][0]} 
            for i, item in enumerate(results_list1)
        ]
        combined_results.extend([
            {'source_file': 2, 'index': i, 'score': item['score'][0][0]} 
            for i, item in enumerate(results_list2)
        ])
        
        if not combined_results:
            print(f"경고: Image ID {image_id}에 대한 점수 정보가 두 파일 모두에 없습니다. 건너뜀.")
            answers[f"TEST_{image_id:03d}"] = "-"
            continue

        # 통합된 결과에서 가장 높은 점수를 찾음
        for result in combined_results:
            if result['score'] > max_score:
                max_score = result['score']
                best_answer_index = result['index']

        if best_answer_index != -1:
            answers[f"TEST_{image_id:03d}"] = answer_chars[best_answer_index]
        else:
            # 이 경우는 combined_results가 비어있지 않은 이상 발생하지 않아야 함
            answers[f"TEST_{image_id:03d}"] = "-"

    # 3. 결과 CSV 파일 작성
    output_lines = ["ID,answer"]
    
    # 정렬된 ID 순서로 CSV 파일 작성
    sorted_numeric_ids = sorted([int(k) for k in all_image_ids])

    for numeric_id in sorted_numeric_ids:
        img_id_formatted = f"TEST_{numeric_id:03d}"
        output_lines.append(f"{img_id_formatted},{answers.get(img_id_formatted, '-')}")

    try:
        with open(output_csv_path, 'w') as f:
            f.write('\n'.join(output_lines))
            f.write('\n') # 파일 끝에 개행 문자 추가
        print(f"결과가 '{output_csv_path}' 파일에 성공적으로 저장되었습니다.")
    except IOError:
        print(f"오류: '{output_csv_path}' 파일에 쓸 수 없습니다. 경로를 확인해주세요.")

# --- 실행 예시 ---
# 비교할 두 개의 점수 파일 경로를 실제 파일 경로로 교체해주세요.
score_file1 = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/score_v2_coco_large.json"
score_file2 = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/score_v2_flickr30k_large.json"

# 저장할 최종 제출 파일명을 지정합니다.
output_submission_file = "submission_combined_final.csv" 

# 함수 호출
generate_combined_retrieval_answers(score_file1, score_file2, output_submission_file)
