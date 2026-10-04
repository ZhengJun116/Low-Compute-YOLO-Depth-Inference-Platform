from .base_detector import BaseDetector
from .yolo_detector import YoloDetector


class DetectorFactory:
    @staticmethod
    def create_detector(model_type: str, model_path: str, **kwargs) -> BaseDetector:
        if model_type.lower() == "yolo":
            return YoloDetector(model_path=model_path, **kwargs)
        else:
            raise ValueError(f"Unknown detector type: {model_type}")
