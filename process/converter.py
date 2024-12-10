# process/converter.py

import json
import os
import cv2
from .utils import restore_polygon, normalize_coordinates, map_label_to_origin

def convert_json_to_yolo_with_polygon(json_path, origin_image_path, labels_dir, class_mapping):
    """JSON 라벨을 YOLO 포맷으로 변환"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_filename = os.path.basename(origin_image_path)
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_filename)

    if not os.path.exists(origin_image_path):
        print(f"이미지 파일이 존재하지 않습니다: {origin_image_path} (라벨: {json_path})")
        return

    image = cv2.imread(origin_image_path)
    if image is None:
        print(f"이미지를 읽는데 실패했습니다: {origin_image_path}")
        return
    height, width = image.shape[:2]

    with open(label_path, 'w', encoding='utf-8') as f:
        for segment in data.get('segments', []):
            type_detail = segment.get('type_detail')
            if type_detail not in class_mapping:
                print(f"알 수 없는 type_detail '{type_detail}' (라벨 파일: {json_path})")
                continue
            class_id = class_mapping[type_detail]

            box = segment.get('box', [])
            polygon = restore_polygon(box)
            if not polygon:
                print(f"유효하지 않은 폴리곤 형식 (라벨 파일: {json_path}): {box}")
                continue

            # 세그멘테이션 좌표 저장
            segmentation = [coord for point in polygon for coord in point]

            # 바운딩 박스 계산 (YOLO 형식)
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            # YOLO 라벨 파일에 바운딩 박스 기록
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
            
            # 세그멘테이션 좌표 추가 (YOLOv5/YOLOv8의 마스크 형식에 맞게 조정 필요)
            # 여기서는 단순히 평평하게 작성했지만, 실제 모델의 요구사항에 맞게 조정해야 합니다.
            f.write(f"{' '.join(map(str, segmentation))}\n")

            # 정규화된 다각형 좌표 저장 (필요 시)
            normalized_polygon = normalize_coordinates(polygon, width, height)
            f.write(f"Normalized polygon: {' '.join(map(str, normalized_polygon))}\n")
