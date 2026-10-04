import argparse
import os
from src.detectors.detector_factory import DetectorFactory
from src.depth_estimators.depth_factory import DepthEstimatorFactory
from src.fusion.camera_params import CameraIntrinsics
from src.fusion.bbox3d_builder import BBox3DBuilder
from src.visualization.visualizer import Visualizer
from src.pipeline.pipeline_factory import PipelineFactory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_SOURCE = os.path.join(BASE_DIR, "test1")  # 圖片預設路徑
VIDEO_SOURCE = os.path.join(BASE_DIR, "test2")  # 影片預設路徑


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--source_type", type=str, required=True, choices=["image", "video", "webcam"])
    parser.add_argument("--output", type=str, default="output/result.jpg")
    parser.add_argument("--yolo_model", type=str, default="weights/yolo11s_openvino_model")
    parser.add_argument("--depth_model", type=str, default="weights/depth_anything")
    args = parser.parse_args()

    if args.source is None:
        if args.source_type == "image":
            args.source = IMAGE_SOURCE
        elif args.source_type in ["video", "webcam"]:
            args.source = VIDEO_SOURCE

    detector = DetectorFactory.create_detector("yolo", args.yolo_model)
    depth_estimator = DepthEstimatorFactory.create_estimator("depth_anything_ov", args.depth_model)
    camera_intrinsics = CameraIntrinsics.create_default(640, 480)
    bbox3d_builder = BBox3DBuilder(camera_intrinsics)
    visualizer = Visualizer()

    pipeline = PipelineFactory.create_pipeline(
        args.source_type,
        detector,
        depth_estimator,
        bbox3d_builder,
        visualizer
    )

    pipeline.run(args.source, args.output)


if __name__ == "__main__":
    main()
