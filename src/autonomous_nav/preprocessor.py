import cv2
import numpy as np
from abc import ABC, abstractmethod
from autonomous_nav.config import PreprocessorConfig


class Preprocessor(ABC):
    """
    ABC for preprocessor classes
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Subclasses must implement this method
        """
        pass


class CLAHEPreprocessor(Preprocessor):
    """
    Contrast limited adaptive histogram equalization preprocessor
    """

    def __init__(self, config: PreprocessorConfig):
        nbins = 256
        clip_limit = config.clahe_clip_limit_normalized * nbins
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=config.clahe_tile_grid_size
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the input image
        """
        return self.clahe.apply(image)


class GaussianBlurPreprocessor(Preprocessor):
    """
    Gaussian blur preprocessor
    """

    def __init__(self, config: PreprocessorConfig):
        self.ksize = config.gaussian_ksize
        self.sigma = config.gaussian_sigma

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gaussian blur to the input image
        """
        return cv2.GaussianBlur(image, self.ksize, self.sigma)


class PreprocessorPipeline:
    """
    Pipeline for chaining together preprocessor steps
    """

    def __init__(self, preprocessors: list[Preprocessor]):
        self.preprocessors = preprocessors

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessor steps in order on the input image
        """
        for proc in self.preprocessors:
            image = proc.process(image)
        return image
