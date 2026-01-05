import os
import qwiic_icm20948
import numpy as np
import time
from autonomous_nav.config import IMUConfig


class IMUModule:
    def __init__(self, config: IMUConfig):
        self.config = config
        # Apply fixed mounting rotation
        self.R_mount = np.array(
            [
                [1, 0, 0],  # rover X = body X
                [0, 0, 1],  # rover Y = body Z
                [0, -1, 0],  # rover Z = -body Y
            ]
        )
        self.imu = qwiic_icm20948.QwiicIcm20948()
        if not self.imu.connected:
            raise Exception("IMU not connected!")
        self.imu.begin()

        # Load or calibrate biases
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        bias_file = self.config.bias_file
        if os.path.exists(bias_file):
            biases = np.load(bias_file)
            self.accel_bias = biases["accel_bias"]
            self.gyro_bias = biases["gyro_bias"]
            self.init_quat = biases["init_quat"]
            print(
                f"Loaded biases: accel={self.accel_bias} (g), gyro={self.gyro_bias} (deg/s), init quat={self.init_quat}"
            )
        else:
            print("Calibrating IMU biases (keep stationary)...")
            self.calibrate_biases()
            print(
                f"Calibrated biases: accel={self.accel_bias} (g), gyro={self.gyro_bias} (deg/s)"
            )

        self.last_time = time.time()

    def calibrate_biases(self):
        accel_samples = []
        gyro_samples = []
        print(
            "Calibrating IMU biases — keep the device completely stationary in its final mounted orientation..."
        )

        for _ in range(self.config.bias_calibration_samples):
            while not self.imu.dataReady():
                time.sleep(0.01)
            self.imu.getAgmt()

            # Raw readings in standard order: X=ax, Y=ay, Z=az
            accel_raw_x = self.imu.axRaw
            accel_raw_y = self.imu.ayRaw
            accel_raw_z = self.imu.azRaw

            gyro_raw_x = self.imu.gxRaw
            gyro_raw_y = self.imu.gyRaw
            gyro_raw_z = self.imu.gzRaw

            # Raw vectors
            accel_raw = np.array([accel_raw_x, accel_raw_y, accel_raw_z])
            gyro_raw = np.array([gyro_raw_x, gyro_raw_y, gyro_raw_z])

            # Convert to physical units
            accel_g_body = accel_raw / self.config.accel_sensitivity
            gyro_deg_s_body = gyro_raw / self.config.gyro_sensitivity

            accel_g = self.R_mount @ accel_g_body
            gyro_deg_s = self.R_mount @ gyro_deg_s_body

            accel_samples.append(accel_g)
            gyro_samples.append(gyro_deg_s)

            time.sleep(1.0 / self.config.sample_rate_hz)

        # Average measurements
        avg_accel_g = np.mean(accel_samples, axis=0)
        avg_gyro = np.mean(gyro_samples, axis=0)

        # === Gyro bias: simple average (should be near zero when stationary) ===
        self.gyro_bias = avg_gyro

        # === Accel bias and initial quaternion ===
        accel_magnitude = np.linalg.norm(avg_accel_g)
        if accel_magnitude < 0.5 or accel_magnitude > 1.5:
            print(
                f"Warning: Unusual accel magnitude {accel_magnitude:.3f} g during calibration"
            )
            # Fallback
            gravity_dir_rover = np.array([0.0, 0.0, 1.0])  # Positive up
            self.accel_bias = np.zeros(3)
            init_q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Unit vector in direction of measured acceleration (points "up" opposing gravity)
            gravity_dir_rover = avg_accel_g / accel_magnitude  # Up direction

            # Expected gravity magnitude = 1 g; bias
            expected_gravity_g = np.array([0.0, 0.0, 1.0])  # Positive up
            self.accel_bias = avg_accel_g - expected_gravity_g * accel_magnitude

            # Desired: gravity opposes in world frame → world gravity dir [0, 0, -1] (down)
            world_gravity_dir = np.array([0.0, 0.0, -1.0])

            # Align rover "up" to world "up" (negative gravity dir)
            v1 = gravity_dir_rover
            v2 = -world_gravity_dir  # World up [0,0,+1]
            cross = np.cross(v1, v2)
            dot = np.dot(v1, v2)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))

            if np.linalg.norm(cross) < 1e-6:
                init_q = (
                    np.array([1.0, 0.0, 0.0, 0.0])
                    if dot > 0
                    else np.array([0.0, 0.0, 0.0, 1.0])
                )
            else:
                axis = cross / np.linalg.norm(cross)
                init_q = np.array(
                    [
                        np.cos(angle / 2),
                        axis[0] * np.sin(angle / 2),
                        axis[1] * np.sin(angle / 2),
                        axis[2] * np.sin(angle / 2),
                    ]
                )

        # Normalize quaternion
        norm = np.linalg.norm(init_q)
        if norm > 1e-8:
            init_q /= norm
        self.init_quat = init_q

        # Save
        np.savez(
            self.config.bias_file,
            accel_bias=self.accel_bias,
            gyro_bias=self.gyro_bias,
            init_quat=self.init_quat,
        )

        print(f"Calibration complete:")
        print(f"  Accel bias (g): {self.accel_bias}")
        print(f"  Gyro bias (deg/s): {self.gyro_bias}")
        print(f"  Initial quaternion (w,x,y,z): {self.init_quat}")
        tilt_deg = np.degrees(2 * np.arccos(np.clip(init_q[0], -1.0, 1.0)))
        print(f"  Estimated tilt from level: ~{tilt_deg:.2f} degrees")

    def read(self) -> dict:
        if not self.imu.dataReady():
            return None
        self.imu.getAgmt()

        dt = time.time() - self.last_time
        self.last_time = time.time()

        # Raw readings in standard order: X=ax, Y=ay, Z=az
        accel_raw_x = self.imu.axRaw
        accel_raw_y = self.imu.ayRaw
        accel_raw_z = self.imu.azRaw

        gyro_raw_x = self.imu.gxRaw
        gyro_raw_y = self.imu.gyRaw
        gyro_raw_z = self.imu.gzRaw

        # Raw vectors
        accel_raw = np.array([accel_raw_x, accel_raw_y, accel_raw_z])
        gyro_raw = np.array([gyro_raw_x, gyro_raw_y, gyro_raw_z])

        # Convert to physical units
        accel_g_body = accel_raw / self.config.accel_sensitivity
        gyro_deg_s_body = gyro_raw / self.config.gyro_sensitivity

        accel_g = self.R_mount @ accel_g_body
        gyro_deg_s = self.R_mount @ gyro_deg_s_body

        # Subtract bias
        accel_g -= self.accel_bias
        gyro_deg_s -= self.gyro_bias

        # Print debug
        print(f"DEBUG: accel: {accel_g}")

        # To cm/s²
        accel_cm_s2 = accel_g * 980.665

        return {
            "accel": accel_cm_s2,
            "gyro": gyro_deg_s,
            "dt": dt,
        }
