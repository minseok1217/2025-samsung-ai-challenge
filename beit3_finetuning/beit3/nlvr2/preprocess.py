import pandas as pd
import os

# --- 설정: 입력 파일과 출력 파일 경로 ---
# 원본 CSV 파일 경로
input_csv_path = '/data/2_data_server/cv-07/dice/2025_samsung_challenge/data/data_2/test.csv'

# 변환된 CSV를 저장할 경로
output_csv_path = '/data/2_data_server/cv-07/dice/2025_samsung_challenge/data/nlvr2/test_transformed.csv'

# 이미지 파일이 있는 디렉토리 경로
image_base_path = '/data/2_data_server/cv-07/dice/2025_samsung_challenge/data/data_2/test_input_images'


# --- 데이터 변환 로직 ---
try:
    # 1. 원본 CSV 파일 읽기
    df = pd.read_csv(input_csv_path)

    # 2. 변환된 데이터를 담을 리스트 초기화
    transformed_data = []

    print("데이터 변환을 시작합니다...")

    # 3. 원본 데이터프레임의 각 행을 순회
    for index, row in df.iterrows():
        original_id = row['ID']
        img_filename = os.path.basename(row['img_path']) # 원본 경로에서 파일명만 추출
        img_path = os.path.join(image_base_path, img_filename) # 새로운 전체 이미지 경로 생성

        # 4. 4개의 보기에 대해 각각 새로운 행 생성
        for option in ['A', 'B', 'C', 'D']:
            # 새로운 고유 ID 생성 (예: TEST_000-A)
            new_id = f"{original_id}-{option}"

            # 새로운 행(딕셔너리) 생성
            new_row = {
                'id': new_id,
                # 이미지는 리스트 형태로 표현
                'images': f"{img_path}",
                # sentence는 해당 보기의 내용
                'sentence': row[option],
                # answer는 임의의 값으로 설정 (여기서는 False로 통일)
                'answer': False,
                # 데이터 소스 추가
                'data_source': 'samsung_challenge_test'
            }
            transformed_data.append(new_row)

    # 5. 리스트를 새로운 데이터프레임으로 변환
    new_df = pd.DataFrame(transformed_data)

    # 6. 결과를 새로운 CSV 파일로 저장
    new_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"변환이 완료되었습니다. 총 {len(new_df)}개의 행이 생성되었습니다.")
    print(f"결과가 '{output_csv_path}' 파일에 저장되었습니다.")

except FileNotFoundError:
    print(f"오류: '{input_csv_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")