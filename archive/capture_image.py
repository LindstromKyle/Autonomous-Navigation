from picamera2 import Picamera2
import numpy as np
import matplotlib.pyplot as plt

# Select name
image_name = "/home/kyle/repos/Autonomous-Navigation/bbox_images/validation_img_04.jpg"

# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()

# Capture a single image as a NumPy array (RGB format by default)
image_array = picam2.capture_array()

# Stop the camera
picam2.stop()

# Inspect the array structure
print("Image shape:", image_array.shape)
print("Data type:", image_array.dtype)

plt.imshow(image_array)
plt.show()

import cv2

cv2.imwrite(image_name, image_array)
