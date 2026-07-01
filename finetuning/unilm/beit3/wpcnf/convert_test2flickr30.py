import csv
import json
import io
import os
import string

# --------------------------------------------------------------------------
# 사용자 설정: 실제 CSV 파일 경로와 저장할 JSON 파일 이름을 지정하세요.
INPUT_CSV_FILE = '/data/2_data_server/cv-07/dice/2025_samsung_challenge/data/data_2/test.csv'  # <<<< 본인의 CSV 파일 경로로 수정하세요.
OUTPUT_JSON_FILE = 'flickr30k_format.json'
# --------------------------------------------------------------------------


# 최종 데이터를 담을 리스트
images_list = []
img_id_counter = 0
sent_id_counter = 0

# 구두점 제거를 위한 테이블 생성
translator = str.maketrans('', '', string.punctuation)

try:
    # CSV 파일을 열기 (인코딩은 'utf-8-sig'로 설정하여 BOM 문자 등도 처리)
    with open(INPUT_CSV_FILE, mode='r', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            # ★★★★★ 오류 해결을 위한 핵심 부분 ★★★★★
            # DictReader로 읽어온 row의 각 key에서 양쪽 공백을 제거합니다.
            # ex) ' img_path ' -> 'img_path'
            cleaned_row = {key.strip(): value for key, value in row.items()}

            # 각 이미지에 대한 정보 딕셔너리
            image_info = {
                "sentids": [],
                "imgid": img_id_counter,
                "sentences": [],
                "split": "test",
                # 이제 공백이 제거된 'img_path' 키를 안전하게 사용할 수 있습니다.
                "filename": os.path.basename(cleaned_row['img_path'])
            }

            # 보기 A, B, C, D를 문장으로 처리
            options = [cleaned_row['A'], cleaned_row['B'], cleaned_row['C'], cleaned_row['D']]

            for option_text in options:
                # 문장(보기)에 대한 정보 딕셔너리
                sentence_info = {
                    "tokens": option_text.lower().translate(translator).split(),
                    "raw": option_text,
                    "imgid": img_id_counter,
                    "sentid": sent_id_counter
                }

                image_info["sentences"].append(sentence_info)
                image_info["sentids"].append(sent_id_counter)
                sent_id_counter += 1

            images_list.append(image_info)
            img_id_counter += 1

    # 최종 JSON 객체 생성
    final_json_output = {"images": images_list}

    # 결과를 JSON 파일로 저장 (들여쓰기로 가독성 높임)
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as json_file:
        json.dump(final_json_output, json_file, indent=2, ensure_ascii=False)

    print(f"✅ 변환 완료! 결과가 {OUTPUT_JSON_FILE} 파일에 저장되었습니다.")

except FileNotFoundError:
    print(f"❌ 오류: '{INPUT_CSV_FILE}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
except KeyError as e:
    print(f"❌ 오류: CSV 파일에서 필요한 컬럼({e})을 찾을 수 없습니다. 파일의 컬럼명을 확인해주세요.")