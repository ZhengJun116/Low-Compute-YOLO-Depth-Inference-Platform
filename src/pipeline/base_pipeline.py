from abc import ABC, abstractmethod
import cv2


class BasePipeline(ABC):
    def __init__(self, detector, depth_estimator, bbox3d_builder, visualizer):
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.bbox3d_builder = bbox3d_builder
        self.visualizer = visualizer

    @abstractmethod
    def run(self, source_path: str, output_path: str):
        pass
