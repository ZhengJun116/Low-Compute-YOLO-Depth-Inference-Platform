import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class Visualizer:
    @staticmethod
    def draw_results(image: np.ndarray, detections: List[Tuple[np.ndarray, float, int]], bbox_3d_info: List[Dict[str, Any]], depth_map: np.ndarray = None) -> np.ndarray:
        result_img = image.copy()

        for (box, conf, cls_id), info in zip(detections, bbox_3d_info):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if info is not None:
                dist = info["distance"]
                text = f"Class {cls_id} | {conf:.2f} | Dist: {dist:.2f}m"
            else:
                text = f"Class {cls_id} | {conf:.2f} | Dist: N/A"

            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_img, (x1, y1 - 20), (x1 + text_w, y1), (0, 255, 0), -1)

            cv2.putText(result_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result_img

    @staticmethod
    def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_norm = depth_map

        depth_norm = depth_norm.astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        return depth_color
