import numpy as np
import openvino as ov
import cv2
from transformers import AutoImageProcessor
import os
from .base_depth import BaseDepthEstimator


class DepthAnythingOpenVINO(BaseDepthEstimator):
    def __init__(self, model_dir: str):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        xml_path = os.path.join(model_dir, "depth_anything.xml")
        bin_path = os.path.join(model_dir, "depth_anything.bin")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"OpenVINO XML file not found: {xml_path}")

        core = ov.Core()
        model = core.read_model(model=xml_path, weights=bin_path)
        self.compiled_model = core.compile_model(model, "CPU")
        self.output_layer = self.compiled_model.output(0)

        self.processor = AutoImageProcessor.from_pretrained(model_dir)

    def estimate(self, image: np.ndarray) -> np.ndarray:
        original_height, original_width = image.shape[:2]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="np")
        pixel_values = inputs["pixel_values"]

        result = self.compiled_model([pixel_values])[self.output_layer]

        depth_map = result[0]
        if len(depth_map.shape) == 3:
            depth_map = depth_map[0]

        depth_map = cv2.resize(depth_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        return depth_map
