import yaml
import numpy as np


class CameraIntrinsics:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        matrix = data.get("camera_matrix", [])
        if len(matrix) == 9:
            fx = matrix[0]
            cx = matrix[2]
            fy = matrix[4]
            cy = matrix[5]
            return cls(fx, fy, cx, cy)
        else:
            raise ValueError(f"Invalid camera matrix in {yaml_path}")

    @classmethod
    def create_default(cls, image_width: int, image_height: int):
        fx = fy = image_width * 1.2
        cx = image_width / 2.0
        cy = image_height / 2.0
        return cls(fx, fy, cx, cy)
