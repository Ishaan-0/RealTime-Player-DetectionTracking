from ultralytics import YOLO
import torch

class YOLOv11Detector:
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = YOLO(model_path).to(self.device)
            print(f"Model loaded successfully from {model_path}.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise

    def detect(self, frame, conf=0.4, iou=0.5, classes=[2]):
        return self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            stream=False,
            verbose=False,
            classes=classes
        )[0]
