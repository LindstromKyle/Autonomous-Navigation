import os
import qwiic_icm20948
import numpy as np
import time
from autonomous_nav.config import IMUConfig


class IMUModule:
    def __init__(self, config: IMUConfig):
        self.config = config
        self.imu = qwiic_icm20948.QwiicIcm20948()
        if not self.imu.connected:
            raise Exception("IMU not connected!")
        self.imu.begin()

        # Biases (estimated at init)
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        if os.path.exists(self.config.bias_file):
            print("Loading saved IMU biases...")
            biases = np.load(self.config.bias_file)
            self.accel_bias = biases["accel_bias"]
            self.gyro_bias = biases["gyro_bias"]
        else:
            self.calibrate_biases()
            np.savez(
                self.config.bias_file,
                accel_bias=self.accel_bias,
                gyro_bias=self.gyro_bias,
            )
            print("Biases saved for future runs.")
        self.last_time = time.time()

    def calibrate_biases(self):
        print("Calibrating IMU biases... Keep device stationary.")
        accel_samples = []
        gyro_samples = []
        for _ in range(self.config.bias_calibration_samples):
            if self.imu.dataReady():
                self.imu.getAgmt()
                accel_raw = np.array([self.imu.axRaw, self.imu.ayRaw, self.imu.azRaw])
                gyro_raw = np.array([self.imu.gxRaw, self.imu.gyRaw, self.imu.gzRaw])
                accel_samples.append(accel_raw)
                gyro_samples.append(gyro_raw)
            time.sleep(1 / self.config.sample_rate_hz)
        self.accel_bias = np.mean(accel_samples, axis=0)
        self.gyro_bias = np.mean(gyro_samples, axis=0)
        print("Calibration done.")

    def read(self) -> dict:
        if not self.imu.dataReady():
            return None
        self.imu.getAgmt()

        dt = time.time() - self.last_time
        self.last_time = time.time()

        # Subtract bias
        accel_raw = (
            np.array([self.imu.axRaw, self.imu.ayRaw, self.imu.azRaw]) - self.accel_bias
        )

        # Invert Y to align with physcial coordinates
        accel_raw[1] = -accel_raw[1]

        accel_g = accel_raw / self.config.accel_sensitivity
        accel_m_s2 = accel_g * 9.80665  # To SI units

        gyro_raw = (
            np.array([self.imu.gxRaw, self.imu.gyRaw, self.imu.gzRaw]) - self.gyro_bias
        )
        gyro_deg_s = gyro_raw / self.config.gyro_sensitivity

        return {
            "accel": accel_m_s2,  # np.array [x, y, z] in m/s²
            "gyro": gyro_deg_s,  # np.array [x, y, z] in deg/s
            "dt": dt,  # Time delta since last read
        }
