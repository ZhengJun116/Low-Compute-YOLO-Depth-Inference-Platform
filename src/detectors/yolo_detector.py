from ultralytics import YOLO
import numpy as np
from typing import List, Tuple
import os
from .base_detector import BaseDetector


class YoloDetector(BaseDetector):
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        self.model = YOLO(model_path, task="detect")

    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> List[Tuple[np.ndarray, float, int]]:
        results = self.model.predict(image, conf=conf, iou=iou, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            for box, c, cls_id in zip(boxes, confs, class_ids):
                detections.append((box, float(c), int(cls_id)))
        return detections
