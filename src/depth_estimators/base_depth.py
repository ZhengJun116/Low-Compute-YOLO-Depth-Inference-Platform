from abc import ABC, abstractmethod
import numpy as np


class BaseDepthEstimator(ABC):
    @abstractmethod
    def __init__(self, model_path: str):
        pass

    @abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:
        pass
