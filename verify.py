import os

# YOLO 라벨 검증 및 수정 함수
def validate_and_fix_yolo_labels(labels_dir, class_count):
    """
    YOLO 라벨을 검증하고 잘못된 박스 크기를 수정하는 함수.
    """
    for root, dirs, files in os.walk(labels_dir):
        for file in files:
            if file.endswith('.txt'):
                label_path = os.path.join(root, file)
                fixed_lines = []
                has_error = False

                with open(label_path, 'r') as f:
                    for line_no, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Invalid format at {label_path}:{line_no} - {line.strip()}")
                            continue

                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                        except ValueError:
                            print(f"Invalid numeric value at {label_path}:{line_no} - {line.strip()}")
                            continue

                        if not (0 <= class_id < class_count):
                            print(f"Invalid class ID at {label_path}:{line_no} - {class_id}")
                            continue

                        # Fix out-of-bound box dimensions
                        if width > 1:
                            print(f"Invalid box width at {label_path}:{line_no} - {width}, fixing it to 1.0")
                            width = 1.0
                            has_error = True
                        if height > 1:
                            print(f"Invalid box height at {label_path}:{line_no} - {height}, fixing it to 1.0")
                            height = 1.0
                            has_error = True

                        if not (0 <= x_center <= 1):
                            print(f"Invalid x_center at {label_path}:{line_no} - {x_center}, fixing it to [0,1] range")
                            x_center = min(max(x_center, 0), 1)
                            has_error = True
                        if not (0 <= y_center <= 1):
                            print(f"Invalid y_center at {label_path}:{line_no} - {y_center}, fixing it to [0,1] range")
                            y_center = min(max(y_center, 0), 1)
                            has_error = True

                        fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # If errors were found, overwrite the file with corrected data
                if has_error:
                    with open(label_path, 'w') as f:
                        f.writelines(fixed_lines)
                    print(f"Fixed errors in {label_path}")

# 데이터셋 디렉토리 경로 설정
processed_dir = "./processed_data"
custom_class_names = [
    '체크', '수식', '동그라미', '화살표', '밑줄', '취소선', '수식/텍스트', '텍스트',
    '2차원 그래프', '기타', '수직선(범위)', '표', '스마일', '수직선', '벤다이어그램',
    '평면도형', '분자구조', '풀이낙서', '별표'
]

# 라벨 검증 및 수정 실행
print("Validating and fixing Training Labels...")
validate_and_fix_yolo_labels(os.path.join(processed_dir, "train/labels"), len(custom_class_names))

print("Validating and fixing Validation Labels...")
validate_and_fix_yolo_labels(os.path.join(processed_dir, "val/labels"), len(custom_class_names))
