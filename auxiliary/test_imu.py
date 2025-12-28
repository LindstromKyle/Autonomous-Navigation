import qwiic_icm20948
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

IMU = qwiic_icm20948.QwiicIcm20948()

if IMU.connected == False:
    print("IMU is not connected...", file=sys.stderr)
    sys.exit(1)

IMU.begin()

while True:
    if IMU.dataReady():

        IMU.getAgmt()

        accel_sens = 16384.0
        accels_raw = np.array([IMU.axRaw, IMU.ayRaw, IMU.azRaw])
        accels_g = accels_raw / accel_sens
        accels_si = np.round(accels_g * 9.80665, 2)

        gyro_sens = 131.0
        gyros_raw = np.array([IMU.gxRaw, IMU.gyRaw, IMU.gzRaw])
        gyros_deg_s = np.round(gyros_raw / gyro_sens, 2)

        mag_sens = 0.15
        mags_raw = np.array([IMU.mxRaw, IMU.myRaw, IMU.mzRaw])
        mags_uT = np.round(mags_raw * mag_sens, 2)

        print(f"Accel (m/s^2): X={accels_si[0]}    Y={accels_si[1]}    Z={accels_si[2]}")
        print(f"Gyro (deg/s):  X={gyros_deg_s[0]}    Y={gyros_deg_s[1]}    Z={gyros_deg_s[2]}")
        print(f"Mag (uT):      X={mags_uT[0]}    Y={mags_uT[1]}    Z={mags_uT[2]}")
        print(f"Temp (C):      {np.round((IMU.tmpRaw / 333.87) + 21.0,2)}")
        print("")
        time.sleep(0.5)

    else:
        print("Waiting for data...")
        time.sleep(0.5)

