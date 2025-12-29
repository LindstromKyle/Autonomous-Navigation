from picamera2 import Picamera2
import cv2
import time
import numpy as np
from autonomous_nav.config import AppConfig


class CameraModule:
    def __init__(self, config: AppConfig):
        self.config = config
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(
            main={"size": config.global_.frame_size, "format": "RGB888"}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()

    def capture_frame(self) -> np.ndarray:
        return self.picam2.capture_array()

    def stop(self):
        self.picam2.stop()

    def run_countdown_preview(self, countdown_duration):

        start_time = time.time()

        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = countdown_duration - elapsed

            if remaining <= 0:
                break

            frame = self.capture_frame()
            overlay = frame.copy()

            cv2.putText(
                overlay,
                f"{remaining:.1f}",
                ((frame.shape[1] // 2) - 50, (frame.shape[0] // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 255, 0),
                5,
                cv2.LINE_AA,
            )

            cv2.imshow("Martian Rover Navigation", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit during countdown")
                self.stop()
                cv2.destroyAllWindows()
                return
