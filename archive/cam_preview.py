from picamera2 import Picamera2, Preview
import time

# Initialize the camera
picam2 = Picamera2()

# Configure for preview (similar resolution to rpicam-hello, e.g., 1920x1080)
preview_config = picam2.create_preview_configuration(main={"size": (1920, 1080)})
picam2.configure(preview_config)

# Start the camera with a Qt preview window
picam2.start_preview(Preview.QTGL)  # QTGL is a good choice for Raspberry Pi 5
picam2.start()

# Display the stream for 5 seconds (adjust as needed)
time.sleep(10)

image_array = picam2.capture_array()


# Stop the camera
picam2.stop()
picam2.stop_preview()

import cv2

image_name = "/home/kyle/repos/Autonomous-Navigation/readme_imgs/hardware.jpeg"
cv2.imwrite(
    image_name,
    cv2.cvtColor(
        image_array,
        cv2.COLOR_BGR2RGB,
    ),
)
