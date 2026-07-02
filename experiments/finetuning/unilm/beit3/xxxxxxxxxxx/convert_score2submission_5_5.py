import json
import os

def generate_averaged_retrieval_answers(score_file_path1, score_file_path2, output_csv_path="submission_averaged.csv"):
    """
    주어진 두 개의 점수 파일에서 각 보기의 점수를 평균내고,
    가장 높은 평균 점수를 가진 보기를 정답으로 선택하여 제출 파일을 생성합니다.

    Args:
        score_file_path1 (str): 첫 번째 점수 JSON 파일 경로.
        score_file_path2 (str): 두 번째 점수 JSON 파일 경로.
        output_csv_path (str): 결과를 저장할 CSV 파일 경로.
    """

    # 1. 두 개의 점수 파일 로드
    try:
        with open(score_file_path1, 'r') as f:
            grouped_scores_data1 = json.load(f)
        print(f"✅ '{score_file_path1}' 파일 로드 완료. 총 이미지 ID: {len(grouped_scores_data1)}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 오류: '{score_file_path1}' 파일 처리 중 문제 발생 - {e}")
        return

    try:
        with open(score_file_path2, 'r') as f:
            grouped_scores_data2 = json.load(f)
        print(f"✅ '{score_file_path2}' 파일 로드 완료. 총 이미지 ID: {len(grouped_scores_data2)}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 오류: '{score_file_path2}' 파일 처리 중 문제 발생 - {e}")
        return

    # 2. 각 이미지에 대한 최적의 답변 찾기 (평균 기반)
    answers = {}
    answer_chars = ['A', 'B', 'C', 'D']

    # 두 파일에 있는 모든 이미지 ID를 통합 (중복 제거)
    all_image_ids = set(grouped_scores_data1.keys()) | set(grouped_scores_data2.keys())
    
    print(f"\n🔄 두 파일에서 총 {len(all_image_ids)}개의 고유한 이미지 ID를 처리합니다.")

    for image_id_str in all_image_ids:
        image_id = int(image_id_str)

        results_list1 = grouped_scores_data1.get(image_id_str)
        results_list2 = grouped_scores_data2.get(image_id_str)

        # 보기(A,B,C,D)별 평균 점수를 저장할 리스트
        average_scores = []

        # 4개의 보기에 대해 순차적으로 평균 계산
        for i in range(4):
            scores_to_average = []
            
            # 파일 1에 해당 이미지의 점수가 있으면 추가
            if results_list1 and len(results_list1) == 4:
                scores_to_average.append(results_list1[i]['score'][0][0])
            
            # 파일 2에 해당 이미지의 점수가 있으면 추가
            if results_list2 and len(results_list2) == 4:
                scores_to_average.append(results_list2[i]['score'][0][0])
            
            # 평균 계산 (점수가 하나만 있으면 그 점수가 평균)
            if scores_to_average:
                avg_score = sum(scores_to_average) / len(scores_to_average)
                average_scores.append(avg_score)
            else:
                # 양쪽 파일 모두에 점수가 없는 비정상적인 경우
                average_scores.append(-float('inf'))
        
        # 가장 높은 평균 점수를 가진 보기의 인덱스를 찾음
        max_avg_score = -float('inf')
        best_answer_index = -1
        for idx, score in enumerate(average_scores):
            if score > max_avg_score:
                max_avg_score = score
                best_answer_index = idx

        if best_answer_index != -1:
            answers[f"TEST_{image_id:03d}"] = answer_chars[best_answer_index]
        else:
            print(f"⚠️ 경고: Image ID {image_id}의 정답을 찾을 수 없습니다.")
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
            f.write('\n')
        print(f"\n🎉 결과가 '{output_csv_path}' 파일에 성공적으로 저장되었습니다.")
    except IOError:
        print(f"❌ 오류: '{output_csv_path}' 파일에 쓸 수 없습니다. 경로 권한을 확인해주세요.")

# --- 실행 예시 ---
# 🚨 아래 파일 경로를 실제 파일 경로로 꼭 교체해주세요!
score_file1 = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/score_v2_coco_large.json"
score_file2 = "/data/2_data_server/cv-07/dice/2025_samsung_challenge/finetuning/unilm/beit3/score_v2_flickr30k_large.json"

# 저장할 최종 제출 파일명을 지정합니다.
output_submission_file = "submission_averaged_5_5.csv" 

# 함수 호출
generate_averaged_retrieval_answers(score_file1, score_file2, output_submission_file)