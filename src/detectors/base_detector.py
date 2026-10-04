from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def __init__(self, model_path: str):
        pass

    @abstractmethod
    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> List[Tuple[np.ndarray, float, int]]:
        pass
