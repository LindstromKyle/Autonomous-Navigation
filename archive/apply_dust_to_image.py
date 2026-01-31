from picamera2 import Picamera2
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from autonomous_nav.dust import (
    DustSimulator,
)
from autonomous_nav.config import AppConfig  # Load your config

# Load config and detector
config = AppConfig()

images_path = Path("/home/kyle/repos/Autonomous-Navigation/bbox_images")

for image_path in images_path.glob("*.jpg"):
    print(f"Processing {image_path.name}")
    dust_sim = DustSimulator(config)
    frame = cv2.imread(str(image_path))
    new_frame_num = int(str(image_path)[64:66]) + 20

    new_image_name = f"/home/kyle/repos/Autonomous-Navigation/bbox_images/training_img_{new_frame_num}.jpg"

    # Add noise and re-detect
    dust_frame = dust_sim.apply_dust(frame)

    cv2.imwrite(new_image_name, dust_frame)
