# 라이브러리 임포트
import os
import json
import cv2
import shutil
import numpy as np
from ultralytics import YOLO
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

# 데이터셋 경로 설정
training_label_dir = "./1.Training/Label"
training_origin_dir = "./1.Training/Origin"
validation_label_dir = "./2.Validation/Label"
validation_origin_dir = "./2.Validation/Origin"

# 결과 디렉토리 생성
processed_dir = "./processed_data"
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(os.path.join(processed_dir, "train/labels"), exist_ok=True)
os.makedirs(os.path.join(processed_dir, "train/images"), exist_ok=True)
os.makedirs(os.path.join(processed_dir, "val/labels"), exist_ok=True)
os.makedirs(os.path.join(processed_dir, "val/images"), exist_ok=True)

# 클래스 리스트 및 매핑
custom_class_names = [
    '체크', '수식', '동그라미', '화살표', '밑줄', '취소선', '수식/텍스트', '텍스트',
    '2차원 그래프', '기타', '수직선(범위)', '표', '스마일', '수직선', '벤다이어그램',
    '평면도형', '분자구조', '풀이낙서', '별표'
]
class_mapping = {name: idx for idx, name in enumerate(custom_class_names)}

# 바운딩 박스를 폴리곤(세그멘테이션)으로 변환
def convert_bbox_to_polygon(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    x_max = (x_center + width / 2) * img_width
    y_max = (y_center + height / 2) * img_height
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

def convert_json_to_yolo(json_path, origin_image_path, labels_dir, class_mapping):
    """
    JSON 어노테이션 파일을 YOLO 세그멘테이션 형식으로 변환하여 저장
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_filename = os.path.basename(origin_image_path)
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_filename)

    if not os.path.exists(origin_image_path):
        print(f"Image file does not exist for {json_path}: {origin_image_path}")
        return

    image = cv2.imread(origin_image_path)
    if image is None:
        print(f"Failed to read image: {origin_image_path}")
        return
    img_height, img_width = image.shape[:2]

    with open(label_path, 'w', encoding='utf-8') as f:
        for segment in data.get('segments', []):
            type_detail = segment.get('type_detail')
            if type_detail not in class_mapping:
                print(f"Unknown type_detail '{type_detail}' in {json_path}")
                continue
            class_id = class_mapping[type_detail]

            # 폴리곤 정보 추출
            polygon_points = segment.get('polygon', [])
            if not polygon_points:
                print(f"No polygon data for {type_detail} in {json_path}")
                continue
            
            polygon_normalized = [
                [point[0] / img_width, point[1] / img_height] for point in polygon_points
            ]
            polygon_flattened = [
                f"{coord:.6f}" for point in polygon_normalized for coord in point
            ]
            polygon_str = ' '.join(polygon_flattened)

            # 바운딩 박스 계산 (YOLO 형식)
            x_coords = [point[0] for point in polygon_points]
            y_coords = [point[1] for point in polygon_points]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)

            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            bbox_width = (x_max - x_min) / img_width
            bbox_height = (y_max - y_min) / img_height

            # YOLO 세그멘테이션 형식으로 저장
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f} {polygon_str}\n")


# 데이터셋 처리 함수
def process_dataset(label_root, origin_root, images_output_dir, labels_output_dir, class_mapping):
    for root, dirs, files in os.walk(label_root):
        for file in files:
            if file.lower().endswith('.json'):
                label_path = os.path.join(root, file)
                origin_image_path = map_label_to_origin(label_path, label_root, origin_root)
                if origin_image_path is None:
                    continue
                convert_json_to_yolo(label_path, origin_image_path, labels_output_dir, class_mapping)
                if os.path.exists(origin_image_path):
                    shutil.copy(origin_image_path, os.path.join(images_output_dir, os.path.basename(origin_image_path)))
                else:
                    print(f"Origin image not found: {origin_image_path}")

# 데이터 경로 매핑
def map_label_to_origin(label_path, label_root, origin_root):
    relative_path = os.path.relpath(label_path, label_root)
    parts = relative_path.split(os.sep)
    if len(parts) < 2:
        print(f"Unexpected label path structure: {label_path}")
        return None
    folder_prefix = parts[0]
    if folder_prefix.startswith('TL'):
        origin_folder_prefix = 'TS' + folder_prefix[2:]
    elif folder_prefix.startswith('VL'):
        origin_folder_prefix = 'VS' + folder_prefix[2:]
    else:
        print(f"Unknown folder prefix in label path: {folder_prefix}")
        return None
    origin_parts = [origin_folder_prefix] + parts[1:]
    return os.path.join(origin_root, *origin_parts).replace('.json', '.png')

# 데이터 처리
print("Processing Training Data...")
process_dataset(training_label_dir, training_origin_dir,
                os.path.join(processed_dir, "train/images"),
                os.path.join(processed_dir, "train/labels"),
                class_mapping)

print("Processing Validation Data...")
process_dataset(validation_label_dir, validation_origin_dir,
                os.path.join(processed_dir, "val/images"),
                os.path.join(processed_dir, "val/labels"),
                class_mapping)

# 데이터셋 설정 파일 생성
data_yaml_path = "./data.yaml"
with open(data_yaml_path, "w", encoding='utf-8') as f:
    yaml_content = {
        "train": os.path.join(processed_dir, "train/images"),
        "val": os.path.join(processed_dir, "val/images"),
        "nc": len(custom_class_names),
        "names": custom_class_names,
        "task": "segment"  # 세그멘테이션 태스크
    }
    json.dump(yaml_content, f, ensure_ascii=False, indent=4)

print("data.yaml 파일 생성 완료.")

# YOLO 모델 학습
model = YOLO('yolo11n-seg.pt')  # YOLO 세그멘테이션 모델

model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name="math_shapes_segmentation",
    augment=True,  # 데이터 증강 활성화
    project="runs/train",
    exist_ok=True
)

print("모델 학습 완료.")

