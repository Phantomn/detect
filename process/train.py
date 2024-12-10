# process/train.py

import torch
from ultralytics import YOLO
from .config import YOLO_MODEL_PATH, EPOCHS, IMG_SIZE, BATCH_SIZE, DEVICE
from .utils import create_data_yaml

def train_model():
    """YOLO 모델을 학습시키는 함수"""
    # GPU 장치 설정
    device = torch.device(DEVICE)
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        print(f"사용 중인 GPU: {torch.cuda.get_device_name(device)}")
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(device)/1024**3, 1), 'GB')

    # data.yaml 파일 생성 (이미 utils.py에서 생성됨)
    # create_data_yaml()

    # YOLO 모델 학습
    model = YOLO(YOLO_MODEL_PATH)  # YOLO 세그멘테이션 모델

    model.train(
        data="./data.yaml",                     # 데이터셋 경로
        epochs=EPOCHS,                          # 학습 epoch 수
        imgsz=IMG_SIZE,                         # 이미지 크기
        batch=BATCH_SIZE,                       # 배치 크기
        name="math_shapes_segmentation",         # 프로젝트 이름
        augment=True,                           # 데이터 증강 활성화
        project="runs/train",                   # 저장 경로
        exist_ok=True,                          # 기존 프로젝트 덮어쓰기
        device=device,                          # 사용할 디바이스 (GPU 또는 CPU)
        half=True,                              # 혼합 정밀도 학습 활성화
        workers=8,                              # 데이터 로딩을 위한 CPU 코어 수 (병렬화)
        lr0=0.01,                               # 초기 학습률
        lrf=0.1,                                # 학습률 스케일링
        warmup_epochs=3,                        # 워밍업 에폭
        optimizer="Adam",                       # 옵티마이저 설정 (옵션: Adam, SGD)
        patience=100                            # 조기 종료를 위한 인내력 (Early Stopping)
    )

    print("모델 학습이 완료되었습니다.")
