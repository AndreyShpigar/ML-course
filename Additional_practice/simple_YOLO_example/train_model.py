"""

"""
import torch
from ultralytics import YOLO

model = YOLO(r'D:\Projects\YOLO_FACE_ENV\PRJ\yolov11m_face.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

results = model.train(
    data = r'labeled_ds\data.yaml',
    epochs = 50,
    patience = 15,
    batch = 8,
    imgsz = 640,
    device = device,
    workers = 0,
    pretrained = True,
    degrees = 5.0,
    shear = 2.0,
    perspective = 0.0005,
    hsv_h = 0.015,
    hsv_s = 0.6,
    hsv_v = 0.4,
    fliplr = 0.5,
    mosaic = 0.0,
    mixup = 0.1
)


