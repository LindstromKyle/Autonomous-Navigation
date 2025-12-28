import cv2
import numpy as np
from abc import ABC, abstractmethod
from autonomous_nav.config import PreprocessorConfig


class Preprocessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        pass


class CLAHEPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig):
        nbins = 256
        clip_limit = config.clahe_clip_limit_normalized * nbins
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=config.clahe_tile_grid_size
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        return self.clahe.apply(image)


class GaussianBlurPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig):
        self.ksize = config.gaussian_ksize
        self.sigma = config.gaussian_sigma

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, self.ksize, self.sigma)


class PreprocessorPipeline:
    def __init__(self, preprocessors: list[Preprocessor]):
        self.preprocessors = preprocessors

    def process(self, image: np.ndarray) -> np.ndarray:
        for proc in self.preprocessors:
            image = proc.process(image)
        return image
