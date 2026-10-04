from .base_pipeline import BasePipeline
from .pipelines import ImagePipeline, VideoPipeline


class PipelineFactory:
    @staticmethod
    def create_pipeline(source_type: str, detector, depth_estimator, bbox3d_builder, visualizer) -> BasePipeline:
        if source_type.lower() == "image":
            return ImagePipeline(detector, depth_estimator, bbox3d_builder, visualizer)
        elif source_type.lower() in ["video", "webcam"]:
            return VideoPipeline(detector, depth_estimator, bbox3d_builder, visualizer)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
