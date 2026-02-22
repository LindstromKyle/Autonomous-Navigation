from picamera2 import Picamera2
import cv2
import time
import numpy as np
from autonomous_nav.config import AppConfig


class CameraModule:
    """
    Camera module
    """

    def __init__(self, config: AppConfig):
        # Configure camera settings
        self.config = config
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(
            main={"size": config.global_.frame_size, "format": "RGB888"}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()

    def capture_frame(self) -> np.ndarray:
        """
        Grabs a single frame
        """
        return self.picam2.capture_array()

    def stop(self):
        """
        Stops the camera module
        """
        self.picam2.stop()

    def run_countdown_preview(self):
        """
        Displays a countdown preview so user can position the camera
        """
        countdown_duration = self.config.global_.countdown_duration
        start_time = time.time()

        while True:
            # Check time
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = countdown_duration - elapsed
            if remaining <= 0:
                break

            # Grab the frame
            frame = self.capture_frame()
            overlay = frame.copy()

            # Text to display
            countdown_text = f"{remaining:.1f}"

            # Get text size for centering
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 5
            (text_width, text_height), baseline = cv2.getTextSize(
                countdown_text, font, font_scale, thickness
            )

            # Get center coordinates
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2

            # Position text
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2

            # Draw the text
            cv2.putText(
                overlay,
                countdown_text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 0),  # Green
                thickness,
                cv2.LINE_AA,
            )

            # Show image
            cv2.imshow("Martian Rover Navigation", overlay)

            # Loop exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit during countdown")
                self.stop()
                cv2.destroyAllWindows()
                return
