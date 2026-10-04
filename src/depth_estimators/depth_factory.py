from .base_depth import BaseDepthEstimator
from .depth_anything import DepthAnythingOpenVINO


class DepthEstimatorFactory:
    @staticmethod
    def create_estimator(model_type: str, model_path: str, **kwargs) -> BaseDepthEstimator:
        if model_type.lower() == "depth_anything_ov":
            return DepthAnythingOpenVINO(model_dir=model_path, **kwargs)
        else:
            raise ValueError(f"Unknown depth estimator type: {model_type}")
