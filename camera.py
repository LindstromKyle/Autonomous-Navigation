from picamera2 import Picamera2
import cv2
import time
import numpy as np
from config import AppConfig


class CameraModule:
    def __init__(self, config: AppConfig):
        self.config = config
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(
            main={"size": config.global_.frame_size, "format": "RGB888"}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        time.sleep(2)

    def capture_frame(self) -> np.ndarray:
        return self.picam2.capture_array()

    def stop(self):
        self.picam2.stop()
