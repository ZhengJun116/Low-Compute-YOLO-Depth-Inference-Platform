import numpy as np
from typing import Tuple, Dict, Any
from .camera_params import CameraIntrinsics


class BBox3DBuilder:
    def __init__(self, camera_intrinsics: CameraIntrinsics):
        self.K = camera_intrinsics

    def build_from_2d_and_depth(self, box_2d: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        x1, y1, x2, y2 = box_2d.astype(int)

        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        obj_depth_map = depth_map[y1:y2, x1:x2]

        valid_mask = (obj_depth_map > 0) & np.isfinite(obj_depth_map)
        if not np.any(valid_mask):
            return None

        u = np.arange(x1, x2).reshape(1, -1).repeat(y2 - y1, axis=0)
        v = np.arange(y1, y2).reshape(-1, 1).repeat(x2 - x1, axis=1)

        u = u[valid_mask]
        v = v[valid_mask]
        z = obj_depth_map[valid_mask]

        x = (u - self.K.cx) * z / self.K.fx
        y = (v - self.K.cy) * z / self.K.fy

        points_3d = np.stack([x, y, z], axis=-1)

        center = np.median(points_3d, axis=0)
        low = np.percentile(points_3d, 5, axis=0)
        high = np.percentile(points_3d, 95, axis=0)
        size = high - low

        distance = float(np.linalg.norm(center))

        return {
            "center_3d": center,
            "size_3d": size,
            "distance": distance
        }
