# Martian Helicopter Autonomous Navigation Project

<img src="readme_imgs/banner.jpeg" alt="Project screenshot" width="70%"/>

## Table of Contents

- [Overview](#overview)
- [Key Achievements](#key-achievements)
- [Hardware](#hardware)
- [Software](#software)
- [Mission Architecture](#mission-architecture)
- [Results](#results)
- [Future Work](#future-work)

## Overview

This project demonstrates an autonomous navigation system for a Martian helicopter drone (inspired by NASA's Ingenuity) which is designed to explore and safely land in an unpredictable and hazardous environment. As a proof of concept, it simulates a drone that must successfully navigate to a predetermined target region, detect and avoid hazards in a dense boulder field, and execute a precision landing in a safe zone free from obstructions - all while handling persistent dust storms that obscure its vision. The system combines sensor fusion, state estimation, computer vision, AI, and mission planning. 


The drone uses a downward-facing camera for optical flow velocity estimation using Shi-Tomasi feature detection and Lucas-Kanade tracking. For hazard avoidance, there are two selectable modes: a baseline cearance based approach using the same features from the corner detector, and an AI variant with a YOLOv8 neural network trained on a custom dataset of labeled images that I generated. The AI approach proves essential in dusty conditions where traditional methods can fail. Sensor fusion is handled by integrating the camera, an IMU, and a LIDAR transceiver into an Extended Kalman Filter for accurate 3D pose and velocity estimation, and a state machine manages the mission phases with hysteresis for stable mode transitions.

**Demo Video**

![Demo video](readme_imgs/recording_20260201_203336.gif)


> Note: This project uses a human-in-the-loop. The real-time algorithms produce on-screen flight directions that are carried out manually - this allows me to thoroughly validate the system without risking physical flight hardware.

## Key Achievements

- **Multi-sensor Fusion** — Implemented a 6-DoF Extended Kalman Filter combining visual velocity, LIDAR altitude, and IMU accelerometer/gyroscope data for robust 3D state estimation.
- **AI Hazard Detection** — Demonstrated significantly improved safe landing zone identification and mission success rate in dusty conditions when using a custom-trained YOLOv8 neural network hazard detector.
- **Martian Dust Simulation** — Built a realistic dust effects module with correlated Gaussian noise and drifting particle overlays to provide a challenging, low-visibility environment.
- **Intelligent Safe Zone Selection** — Engineered a hazard avoidance algorithm using distance transforms, hazard dilation, and proximity scoring to select the safest landing spot within a configurable search zone.
- **Real-time Visualizations** — Created a rich real-time overlay with feature trails, navigation arrows, safe-zone heatmaps, mode banners, altitude bars, and hover countdown, providing intuitive feedback during live hardware testing.

## Hardware

<img src="readme_imgs/hardware.jpeg" alt="Hardware" width="30%"/>

*The system runs on a Raspberry Pi 5 with several sensors attached to mimic a drone helicopter's payload.*

**Components:**
- **Raspberry Pi 5** — Core processing unit
- **ArduCam Camera Module 3** — Optical flow and visual input
- **IMU (ICM-20948)** — Acceleration and gyro data for orientation
- **LIDAR (VL53L1X)** — Precise altitude measurement

> Note: Also shown are a GPS receiver and an atmospheric pressure sensor. Although these were actively collecting data, they were ultimately unnecessary for this project. 

## Software

The software is written entirely in Python and emphasizes modularity, configurability, and real-time performance on embedded hardware.

**Directory structure**:

```text
src/autonomous_nav/
├── app.py                # Main application loop
├── config.py             # Centralized configuration
├── camera.py             # Camera handling & countdown
├── imu.py                # IMU driver & calibration
├── lidar.py              # LIDAR driver
├── preprocessor.py       # Image enhancement pipeline (CLAHE, blur)
├── feature_detector.py   # Shi-Tomasi corner detection
├── optical_flow.py       # Lucas-Kanade optical flow
├── state_estimator.py    # Extended Kalman Filter
├── hazard_avoidance.py   # Hazard detection (AI + classical)
├── dust.py               # Realistic Martian dust simulation
├── mission_manager.py    # Mission state machine
├── visualizer.py         # Real-time UI overlays & annotations
└── utils.py              # Helper functions
```

## Mission Architecture

The system follows a state machine that simulates a full Martian exploration and precision landing scenario:

1. **Ascent** — Climb to cruising altitude.

2. **Navigation** — Travel toward a predefined target location using optical flow velocity estimates and fused state estimation.

3. **Searching** — Once inside the inner search radius, scan for safe landing zones using real-time hazard detection and select the safest candidate spot.

4. **Landing Approach** — Lock onto a stable safe site (median-filtered over multiple frames), navigate precisely toward it, and maintain clearance checks with hysteresis to avoid false triggers.

5. **Descent & Hovering** — Descend to final altitude, transition to stable hover, and confirm landing readiness with a timed hover duration within position tolerance.

6. **Landed Safe** / **No Safe Zone** — Declare mission success on stable hover completion, or fall back to NO_SAFE_ZONE mode if no viable spot is found.

Transitions use hysteresis via consecutive-frame counters to ensure reliable mode changes even under noisy detection.


## Results

**Chalenges and Solutions:**

| Challenge                               | Solution                                                                                   |
|-----------------------------------------|--------------------------------------------------------------------------------------------|
| Dust particles fool corner detectors    | Switched to semantic AI (YOLO) for hazard detection; dust simulator for rigorous testing |
| Accumulating drift in visual odometry   | Sensor fusion with IMU & LiDAR; proper bias calibration & quaternion handling             |
| Real-time performance on Raspberry Pi   | Optimized YOLO inference (320×320), lightweight preprocessing, median flow               |
| Unstable landing mode transitions       | Added hysteresis (consecutive frame confirmation) & median filtering of landing points   |
| Initial orientation/tilt errors         | Gravity-based automatic initial quaternion alignment during IMU calibration              |


| Clean Image                  | With Dust Simulation              |
|------------------------------|-----------------------------------|
| ![Clean](readme_imgs/no_dust.png)   | ![Noisy](readme_imgs/dusty.png)        |
| **Dust + Shi-Tomasi Corners** | **Dust + YOLO Rock Detection**    |
| ![Corners](readme_imgs/dusty_corners.png) | ![YOLO](readme_imgs/dusty_yolo.png)         |

This design makes it easy to swap components (e.g. classical vs. AI hazard detection).