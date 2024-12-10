# process/yolo_utils.py

import os

def validate_and_fix_yolo_labels(labels_dir):
    """YOLO 라벨 파일을 검증하고 수정"""
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)

        if not label_file.endswith('.txt'):
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        with open(label_path, 'w') as f:
            for line in lines:
                # 'Normalized polygon' 라인 무시
                if 'Normalized polygon' in line:
                    continue

                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    # YOLO 형식이 맞는지 확인
                    class_id, x_center, y_center, width, height = map(float, parts[:5])

                    # 클래스 ID는 정수로 변환 (부동소수점 -> 정수)
                    class_id = int(class_id)  # 정수로 변환

                    # 유효한 범위로 클리핑
                    width = min(max(width, 0), 1)
                    height = min(max(height, 0), 1)

                    # 라벨 파일에 수정된 값 기록
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                except ValueError:
                    print(f"유효하지 않은 라인 스킵: {label_file} - {line.strip()}")
