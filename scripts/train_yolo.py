# from ultralytics import YOLO

# # command for training yolo:
# # yolo train data=f1-dataset/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=7
# model = YOLO("runs/detect/train/weights/best.pt")  # Cargar modelo entrenado
# # Probar en una imagen
# results = model.predict("f1-dataset/test/images/image_001.jpg", save=True)


import os
from PIL import Image

carpeta = "f1-strategy/f1-dataset/train/images"
for archivo in os.listdir(carpeta):
    if archivo.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(os.path.join(carpeta, archivo))
        print(f"{archivo}: {img.size}")
