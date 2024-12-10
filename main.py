# main.py

import os
import shutil
from process.converter import convert_json_to_yolo_with_polygon
from process.yolo_utils import validate_and_fix_yolo_labels
from process.train import train_model
from process.config import (
    TRAINING_LABEL_DIR,
    TRAINING_ORIGIN_DIR,
    VALIDATION_LABEL_DIR,
    VALIDATION_ORIGIN_DIR,
    PROCESSED_DIR,
    CUSTOM_CLASS_NAMES,
    CLASS_MAPPING
)
from process.utils import create_dirs, create_data_yaml, map_label_to_origin

def process_dataset(label_root, origin_root, images_output_dir, labels_output_dir, class_mapping):
    """훈련 또는 검증 데이터셋을 처리하는 함수"""
    for root, dirs, files in os.walk(label_root):
        for file in files:
            if file.lower().endswith('.json'):
                label_path = os.path.join(root, file)
                origin_image_path = map_label_to_origin(label_path, label_root, origin_root)
                if origin_image_path is None:
                    continue
                convert_json_to_yolo_with_polygon(label_path, origin_image_path, labels_output_dir, class_mapping)
                if os.path.exists(origin_image_path):
                    shutil.copy(origin_image_path, os.path.join(images_output_dir, os.path.basename(origin_image_path)))
                else:
                    print(f"원본 이미지 파일이 존재하지 않습니다: {origin_image_path}")

if __name__ == "__main__":
    # 디렉토리 생성
    create_dirs()

    # 훈련 데이터 처리
    print("훈련 데이터를 처리 중입니다...")
    process_dataset(
        label_root=TRAINING_LABEL_DIR,
        origin_root=TRAINING_ORIGIN_DIR,
        images_output_dir=os.path.join(PROCESSED_DIR, "train/images"),
        labels_output_dir=os.path.join(PROCESSED_DIR, "train/labels"),
        class_mapping=CLASS_MAPPING
    )

    # 검증 데이터 처리
    print("검증 데이터를 처리 중입니다...")
    process_dataset(
        label_root=VALIDATION_LABEL_DIR,
        origin_root=VALIDATION_ORIGIN_DIR,
        images_output_dir=os.path.join(PROCESSED_DIR, "val/images"),
        labels_output_dir=os.path.join(PROCESSED_DIR, "val/labels"),
        class_mapping=CLASS_MAPPING
    )

    # data.yaml 파일 생성
    create_data_yaml()

    # YOLO 라벨 데이터 검증 및 수정
    print("YOLO 라벨 파일을 검증하고 수정 중입니다...")
    validate_and_fix_yolo_labels(os.path.join(PROCESSED_DIR, "train/labels"))
    validate_and_fix_yolo_labels(os.path.join(PROCESSED_DIR, "val/labels"))

    # 모델 학습
    print("모델 학습을 시작합니다...")
    train_model()
