# process/config.py

import os
import torch

# 데이터셋 경로 설정
TRAINING_LABEL_DIR = "./1.Training/Label"
TRAINING_ORIGIN_DIR = "./1.Training/Origin"
VALIDATION_LABEL_DIR = "./2.Validation/Label"
VALIDATION_ORIGIN_DIR = "./2.Validation/Origin"
PROCESSED_DIR = "./processed_data"

# 클래스 리스트 및 매핑
CUSTOM_CLASS_NAMES = [
    '체크', '수식', '동그라미', '화살표', '밑줄', '취소선', '수식/텍스트', '텍스트',
    '2차원 그래프', '기타', '수직선(범위)', '표', '스마일', '수직선', '벤다이어그램',
    '평면도형', '분자구조', '풀이낙서', '별표'
]
CLASS_MAPPING = {name: idx for idx, name in enumerate(CUSTOM_CLASS_NAMES)}

# 학습 설정
YOLO_MODEL_PATH = 'yolo11n-seg.pt'
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
