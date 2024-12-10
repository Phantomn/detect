# process/utils.py

import os
import json
import yaml
import shutil
from .config import PROCESSED_DIR, CUSTOM_CLASS_NAMES

def create_dirs():
    """데이터셋을 처리할 폴더 생성"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "val/labels"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "val/images"), exist_ok=True)
    print("필요한 디렉토리들이 생성되었습니다.")

def create_data_yaml():
    """YOLO 학습에 필요한 data.yaml 파일 생성"""
    data_yaml_path = "./data.yaml"
    yaml_content = {
        "train": os.path.join(PROCESSED_DIR, "train/images"),
        "val": os.path.join(PROCESSED_DIR, "val/images"),
        "nc": len(CUSTOM_CLASS_NAMES),
        "names": CUSTOM_CLASS_NAMES,
        "task": "segment"  # 세그멘테이션 태스크
    }
    with open(data_yaml_path, "w", encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)
    print("data.yaml 파일이 생성되었습니다.")

def map_label_to_origin(label_path, label_root, origin_root):
    """라벨 파일 경로를 원본 이미지 경로로 매핑"""
    relative_path = os.path.relpath(label_path, label_root)
    parts = relative_path.split(os.sep)
    if len(parts) < 2:
        print(f"예상치 못한 라벨 경로 구조: {label_path}")
        return None
    folder_prefix = parts[0]
    origin_folder_prefix = folder_prefix.replace('L', 'S', 1)  # 'L' -> 'S'로 변경
    origin_parts = [origin_folder_prefix] + parts[1:]
    origin_path = os.path.join(origin_root, *origin_parts)
    return os.path.splitext(origin_path)[0] + '.png'

def restore_polygon(box):
    """바운딩 박스를 사각형 형태로 복원"""
    if len(box) == 4:  # 4개 점으로 이루어진 경우
        return box
    elif len(box) > 4:
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    else:
        print(f"유효하지 않은 박스 데이터: {box}")
        return None

def normalize_coordinates(polygon, image_width, image_height):
    """다각형 좌표를 0~1 사이로 정규화"""
    normalized_polygon = []
    for point in polygon:
        x = point[0] / image_width
        y = point[1] / image_height
        x = max(0, min(x, 1))  # [0,1] 범위로 클리핑
        y = max(0, min(y, 1))
        normalized_polygon.append([x, y])
    return normalized_polygon
