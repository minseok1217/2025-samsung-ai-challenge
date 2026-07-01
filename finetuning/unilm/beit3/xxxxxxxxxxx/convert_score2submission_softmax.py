import json
import numpy as np

def create_ensemble_submission(input_file1, input_file2, output_filename):
    """
    두 개의 점수 파일(JSON)을 읽어 Softmax 확률을 더하는 방식으로 앙상블을 수행하고,
    최종 예측 결과를 CSV 파일로 저장합니다.

    Args:
        input_file1 (str): 첫 번째 입력 JSON 파일 경로.
        input_file2 (str): 두 번째 입력 JSON 파일 경로.
        output_filename (str): 결과를 저장할 CSV 파일 경로.
    """
    # 1. 두 개의 점수 파일 불러오기
    try:
        with open(input_file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        print(f"✅ '{input_file1}' 파일을 성공적으로 불러왔습니다.")
        
        with open(input_file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        print(f"✅ '{input_file2}' 파일을 성공적으로 불러왔습니다.")
    except FileNotFoundError as e:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. - {e}")
        return

    # 2. Softmax 앙상블로 정답 예측
    answers = {}
    option_map = ['A', 'B', 'C', 'D']

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # data1의 모든 이미지 ID에 대해 반복
    print("\n앙상블 예측을 시작합니다...")
    for image_id, options1 in data1.items():
        # data2에도 동일한 ID가 있는지 확인
        if image_id in data2:
            options2 = data2[image_id]

            # 두 데이터셋의 보기 개수가 모두 4개인지 확인
            if len(options1) == 4 and len(options2) == 4:
                # 각 파일에서 점수 추출
                scores1 = [opt['score'][0][0] for opt in options1]
                scores2 = [opt['score'][0][0] for opt in options2]
                
                # 각 점수에 대해 Softmax 확률 계산
                probs1 = softmax(scores1)
                probs2 = softmax(scores2)
                
                # ★★★ 두 모델의 확률을 더하여 앙상블 ★★★
                combined_probs = probs1 + probs2
                
                # 합산된 확률이 가장 높은 보기의 인덱스를 찾음
                best_option_index = np.argmax(combined_probs)
                
                # 최종 정답을 'A', 'B', 'C', 'D'로 변환하여 저장
                answers[image_id] = option_map[best_option_index]

    # 3. 결과 CSV 파일 작성
    output_lines = ["ID,answer"]
    
    all_image_ids = answers.keys()
    # JSON의 key는 문자열이므로 정수형으로 변환하여 정렬
    sorted_numeric_ids = sorted([int(k) for k in all_image_ids])

    for numeric_id in sorted_numeric_ids:
        # 출력 형식에 맞게 ID 포맷팅 (e.g., 0 -> TEST_000)
        img_id_formatted = f"TEST_{numeric_id:03d}"
        
        # 원본 ID(문자열)를 사용하여 정답을 가져옴
        original_id = str(numeric_id)
        answer = answers.get(original_id, '-')
        
        output_lines.append(f"{img_id_formatted},{answer}")

    # 파일에 쓰기
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\n✅ 앙상블 예측 완료! 최종 결과가 '{output_filename}' 파일에 저장되었습니다.")
    except IOError as e:
        print(f"❌ 오류: '{output_filename}' 파일 저장에 실패했습니다. - {e}")

    # 결과 샘플 출력 (첫 5줄)
    print("\n--- CSV 파일 내용 샘플 ---")
    for line in output_lines[:6]:
        print(line)


# --- 코드 실행 ---
create_ensemble_submission(
    input_file1='score_v2_flickr30k_large.json',
    input_file2='score_v2_coco_large.json',
    output_filename='submission_ensemble.csv'
)