# 라이브러리 임포트
import os
import json
import cv2
import uuid
import numpy as np
from ultralytics import YOLO
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import time
import requests
import chardet

# ### 2. 데이터셋 준비 (JSON 어노테이션을 YOLO 형식으로 변환) ###

# 데이터셋 경로 설정 (Kaggle 데이터셋으로 업로드된 경우)
training_label_dir = "./1.Training/Label"
training_origin_dir = "./1.Training/Origin"
validation_label_dir = "./2.Validation/Label"
validation_origin_dir = "./2.Validation/Origin"

# 결과 디렉토리 생성 (YOLO 형식 어노테이션 저장용)
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

def map_label_to_origin(label_path, label_root, origin_root):
    """
    Label 경로를 Origin 경로로 변환하는 함수
    예: /1.Training/Label/TL_E4/411/file.json -> /1.Training/Origin/TS_E4/411/file.png
    """
    relative_path = os.path.relpath(label_path, label_root)
    parts = relative_path.split(os.sep)
    # 첫 번째 폴더명 (예: TL_E4)에서 'L'을 'S'로 변경 (TL_E4 -> TS_E4)
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
    # Reconstruct Origin path
    origin_parts = [origin_folder_prefix] + parts[1:]
    origin_path = os.path.join(origin_root, *origin_parts)
    # Replace .json with .png
    origin_image_path = os.path.splitext(origin_path)[0] + '.png'
    return origin_image_path

def convert_json_to_yolo(json_path, origin_image_path, labels_dir, class_mapping):
    """
    JSON 어노테이션 파일을 YOLO 형식으로 변환하여 저장하는 함수
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
    height, width = image.shape[:2]
    
    with open(label_path, 'w', encoding='utf-8') as f:
        for segment in data.get('segments', []):
            type_detail = segment.get('type_detail')
            if type_detail not in class_mapping:
                print(f"Unknown type_detail '{type_detail}' in {json_path}")
                continue
            class_id = class_mapping[type_detail]
            
            # Bounding box 변환 (YOLO 형식: class_id x_center y_center width height)
            box = segment.get('box', [])
            if len(box) != 4:
                print(f"Invalid box format in {json_path}: {box}")
                continue
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height
            
            # YOLO 형식으로 기록
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

def process_dataset(label_root, origin_root, images_output_dir, labels_output_dir, class_mapping):
    """
    데이터셋의 모든 JSON 어노테이션 파일을 YOLO 형식으로 변환하고 이미지 파일을 복사
    """
    for root, dirs, files in os.walk(label_root):
        for file in files:
            if file.lower().endswith('.json'):
                label_path = os.path.join(root, file)
                origin_image_path = map_label_to_origin(label_path, label_root, origin_root)
                if origin_image_path is None:
                    continue
                convert_json_to_yolo(label_path, origin_image_path, labels_output_dir, class_mapping)
                # 이미지 파일을 YOLO 형식 디렉토리로 복사
                if os.path.exists(origin_image_path):
                    shutil.copy(origin_image_path, os.path.join(images_output_dir, os.path.basename(origin_image_path)))
                else:
                    print(f"Origin image not found: {origin_image_path}")

# 훈련 데이터 처리
print("Processing Training Data...")
process_dataset(training_label_dir, training_origin_dir, 
               os.path.join(processed_dir, "train/images"), 
               os.path.join(processed_dir, "train/labels"), 
               class_mapping)

# 검증 데이터 처리
print("Processing Validation Data...")
process_dataset(validation_label_dir, validation_origin_dir, 
               os.path.join(processed_dir, "val/images"), 
               os.path.join(processed_dir, "val/labels"), 
               class_mapping)