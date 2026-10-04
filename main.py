import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detectors.detector_factory import DetectorFactory
from src.depth_estimators.depth_factory import DepthEstimatorFactory
from src.fusion.camera_params import CameraIntrinsics
from src.fusion.bbox3d_builder import BBox3DBuilder
from src.visualization.visualizer import Visualizer
from src.pipeline.pipeline_factory import PipelineFactory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "test1")   # 圖片資料夾路徑
VIDEO_DIR = os.path.join(BASE_DIR, "test2")   # 影片資料夾路徑
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # 輸出資料夾路徑

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")


def main():
    detector = DetectorFactory.create_detector("yolo", os.path.join(BASE_DIR, "weights", "yolo11s_openvino_model"))
    depth_estimator = DepthEstimatorFactory.create_estimator("depth_anything_ov", os.path.join(BASE_DIR, "weights", "depth_anything"))
    camera_intrinsics = CameraIntrinsics.create_default(640, 480)
    bbox3d_builder = BBox3DBuilder(camera_intrinsics)
    visualizer = Visualizer()

    saved_paths = []

    images = sorted([
        f for f in glob.glob(os.path.join(IMAGE_DIR, "*"))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])

    videos = sorted([
        f for f in glob.glob(os.path.join(VIDEO_DIR, "*"))
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ])

    if not images and not videos:
        print(f"在 {IMAGE_DIR} 和 {VIDEO_DIR} 中都沒有找到可處理的檔案")
        print(f"請將圖片放入 {IMAGE_DIR}，影片放入 {VIDEO_DIR}")
        return

    for img_path in images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{basename}_result.jpg")
        pipeline = PipelineFactory.create_pipeline("image", detector, depth_estimator, bbox3d_builder, visualizer)
        pipeline.run(img_path, out_path)
        saved_paths.append(out_path)

    for vid_path in videos:
        basename = os.path.splitext(os.path.basename(vid_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{basename}_result.mp4")
        pipeline = PipelineFactory.create_pipeline("video", detector, depth_estimator, bbox3d_builder, visualizer)
        pipeline.run(vid_path, out_path)
        saved_paths.append(out_path)

    print("\n========== 推論完成 ==========")
    print(f"共處理 {len(images)} 張圖片, {len(videos)} 部影片")
    print("結果儲存位置:")
    for p in saved_paths:
        print(f"  {p}")
    print("==============================")


if __name__ == "__main__":
    main()
