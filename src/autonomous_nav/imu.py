import qwiic_icm20948
import numpy as np
import time
from autonomous_nav.config import IMUConfig


class IMUModule:
    """
    IMU module
    """

    def __init__(self, config: IMUConfig):
        # Set up config params
        self.config = config
        self.imu = qwiic_icm20948.QwiicIcm20948()
        if not self.imu.connected:
            raise Exception("IMU not connected!")
        self.imu.begin()

        # Calibrate IMU biases
        print("Calibrating gyro bias and initial orientation...")
        self.calibrate_biases()
        print(f"Calibrated gyro bias (g), gyro={self.gyro_bias} (deg/s)")
        self.last_time = time.time()

    def calibrate_biases(self):
        """
        Calibration sequence for IMU biases
        """
        accel_samples = []
        gyro_samples = []

        for _ in range(self.config.bias_calibration_samples):
            while not self.imu.dataReady():
                time.sleep(0.01)
            self.imu.getAgmt()

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
            accel_g = accel_raw / self.config.accel_sensitivity
            gyro_deg_s = gyro_raw / self.config.gyro_sensitivity

            accel_samples.append(accel_g)
            gyro_samples.append(gyro_deg_s)

            time.sleep(1.0 / self.config.sample_rate_hz)

        # Average measurements
        avg_accel_g = np.mean(accel_samples, axis=0)
        avg_gyro = np.mean(gyro_samples, axis=0)

        # Gyro bias: simple average
        self.gyro_bias = avg_gyro

        # Compute gravity direction from raw average
        accel_magnitude = np.linalg.norm(avg_accel_g)
        gravity_dir_body = avg_accel_g / accel_magnitude
        world_gravity_dir = np.array([0.0, 0.0, 1.0])

        # Compute initial quaternion to align body gravity to world gravity
        v1 = gravity_dir_body
        v2 = world_gravity_dir
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

        # Print results
        print(f"Calibration complete:")
        print(f"  Gyro bias (deg/s): {self.gyro_bias}")
        print(f"  Initial quaternion (w,x,y,z): {self.init_quat}")
        tilt_deg = np.degrees(2 * np.arccos(np.clip(init_q[0], -1.0, 1.0)))
        print(f"  Estimated tilt from level: ~{tilt_deg:.2f} degrees")

    def read(self) -> dict:
        """
        Reads data from IMU sensor
        """
        if not self.imu.dataReady():
            return None
        self.imu.getAgmt()

        dt = time.time() - self.last_time
        self.last_time = time.time()

        # Raw readings
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
        accel_g = accel_raw / self.config.accel_sensitivity
        gyro_deg_s = gyro_raw / self.config.gyro_sensitivity

        # Subtract bias
        gyro_deg_s -= self.gyro_bias

        # Convert to cm/s²
        accel_cm_s2 = accel_g * 980.665

        return {
            "accel": accel_cm_s2,
            "gyro": gyro_deg_s,
            "dt": dt,
        }
