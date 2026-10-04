import cv2
import os
import numpy as np
from .base_pipeline import BasePipeline


class ImagePipeline(BasePipeline):
    def run(self, source_path: str, output_path: str):
        image = cv2.imread(source_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source_path}")

        detections = self.detector.detect(image)
        depth_map = self.depth_estimator.estimate(image)

        bbox_3d_info = []
        for box, conf, cls_id in detections:
            info = self.bbox3d_builder.build_from_2d_and_depth(box, depth_map)
            bbox_3d_info.append(info)

        depth_color = self.visualizer.colorize_depth(depth_map)
        depth_color = cv2.resize(depth_color, (image.shape[1], image.shape[0]))
        result_img = self.visualizer.draw_results(depth_color, detections, bbox_3d_info, depth_map)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"Saved result to {output_path}")


class VideoPipeline(BasePipeline):
    def run(self, source_path: str, output_path: str):
        if source_path.isdigit():
            cap = cv2.VideoCapture(int(source_path))
        else:
            cap = cv2.VideoCapture(source_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            depth_map = self.depth_estimator.estimate(frame)

            bbox_3d_info = []
            for box, conf, cls_id in detections:
                info = self.bbox3d_builder.build_from_2d_and_depth(box, depth_map)
                bbox_3d_info.append(info)

            depth_color = self.visualizer.colorize_depth(depth_map)
            depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
            result_img = self.visualizer.draw_results(depth_color, detections, bbox_3d_info, depth_map)
            out.write(result_img)

        cap.release()
        out.release()
        print(f"Saved video result to {output_path}")
