import os
import json
import shutil

def replace_type_detail(root_dir, old_value, new_value, backup=False):
    """
    지정된 루트 디렉토리 내의 모든 JSON 파일에서 'type_detail' 값을 변경합니다.
    
    Parameters:
        root_dir (str): JSON 파일이 포함된 루트 디렉토리 경로.
        old_value (str): 변경할 기존 'type_detail' 값.
        new_value (str): 새로운 'type_detail' 값.
        backup (bool): 변경 전에 원본 파일을 백업할지 여부.
    """
    # JSON 파일 탐색
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    segments = data.get('segments', [])
                    updated = False
                    
                    for segment in segments:
                        if segment.get('type_detail') == old_value:
                            segment['type_detail'] = new_value
                            updated = True
                    
                    if updated:
                        if backup:
                            # 백업 파일 생성 (원본 파일명에 .bak 추가)
                            backup_path = file_path + '.bak'
                            shutil.copyfile(file_path, backup_path)
                            print(f"백업 생성: {backup_path}")
                        
                        # 변경된 데이터를 원본 파일에 저장
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        print(f"변경 완료: {file_path}")
                    
                except Exception as e:
                    print(f"파일 처리 중 오류 발생 ({file_path}): {e}")

# 사용 예시
# 훈련 데이터 라벨 디렉토리
training_label_dir = "./1.Training/Label"

# 검증 데이터 라벨 디렉토리
validation_label_dir = "./2.Validation/Label"

# 백업을 원하지 않으면 backup=False로 설정
replace_type_detail(root_dir=training_label_dir, old_value="텍스트/수식", new_value="수식/텍스트", backup=False)
replace_type_detail(root_dir=validation_label_dir, old_value="텍스트/수식", new_value="수식/텍스트", backup=False)
replace_type_detail(root_dir=training_label_dir, old_value="2차원그래프", new_value="2차원 그래프", backup=False)
replace_type_detail(root_dir=validation_label_dir, old_value="2차원그래프", new_value="2차원 그래프", backup=False)
