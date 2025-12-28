import qwiic_icm20948
import time
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# IMU setup
IMU = qwiic_icm20948.QwiicIcm20948()

if IMU.connected == False:
    print("The device isn't connected. Check your wiring.", file=sys.stderr)
    sys.exit(1)

IMU.begin()

# Constants for conversions
G = 9.80665  # m/s² per g (but we'll use g for accel)
DPS_TO_RADS = math.pi / 180.0  # Not using rad/s, sticking to dps
ACCEL_SENS = 16384.0  # LSB/g for ±2g
GYRO_SENS = 131.0  # LSB/dps for ±250 dps
MAG_SENS = 0.15  # μT/LSB

# Rolling window: 10 seconds, assuming ~100 Hz sampling -> ~1000 points
WINDOW_SECONDS = 10.0
MAX_POINTS = 1000  # Adjust based on actual rate; deque will handle overflow

# Data storage: deques for each axis and time
times = deque(maxlen=MAX_POINTS)
accel_x = deque(maxlen=MAX_POINTS)
accel_y = deque(maxlen=MAX_POINTS)
accel_z = deque(maxlen=MAX_POINTS)
gyro_x = deque(maxlen=MAX_POINTS)
gyro_y = deque(maxlen=MAX_POINTS)
gyro_z = deque(maxlen=MAX_POINTS)
mag_x = deque(maxlen=MAX_POINTS)
mag_y = deque(maxlen=MAX_POINTS)
mag_z = deque(maxlen=MAX_POINTS)

# Last update time to trim window
last_time = time.time()


def update(frame):
    global last_time
    current_time = time.time()

    if IMU.dataReady():
        IMU.getAgmt()

        # Convert to units
        ax_g = IMU.axRaw / ACCEL_SENS
        ay_g = IMU.ayRaw / ACCEL_SENS
        az_g = IMU.azRaw / ACCEL_SENS

        gx_dps = IMU.gxRaw / GYRO_SENS
        gy_dps = IMU.gyRaw / GYRO_SENS
        gz_dps = IMU.gzRaw / GYRO_SENS

        mx_ut = IMU.mxRaw * MAG_SENS
        my_ut = IMU.myRaw * MAG_SENS
        mz_ut = IMU.mzRaw * MAG_SENS

        # Append new data
        times.append(current_time)
        accel_x.append(ax_g)
        accel_y.append(ay_g)
        accel_z.append(az_g)
        gyro_x.append(gx_dps)
        gyro_y.append(gy_dps)
        gyro_z.append(gz_dps)
        mag_x.append(mx_ut)
        mag_y.append(my_ut)
        mag_z.append(mz_ut)

        # Trim old data if older than window (deque maxlen handles count, but we enforce time)
        while times and current_time - times[0] > WINDOW_SECONDS:
            times.popleft()
            accel_x.popleft()
            accel_y.popleft()
            accel_z.popleft()
            gyro_x.popleft()
            gyro_y.popleft()
            gyro_z.popleft()
            mag_x.popleft()
            mag_y.popleft()
            mag_z.popleft()

        # Update plots (relative time: subtract oldest time)
        rel_times = [t - times[0] for t in times] if times else []

        # Accel
        line_ax.set_data(rel_times, accel_x)
        line_ay.set_data(rel_times, accel_y)
        line_az.set_data(rel_times, accel_z)
        ax1.relim()
        ax1.autoscale_view()

        # Gyro
        line_gx.set_data(rel_times, gyro_x)
        line_gy.set_data(rel_times, gyro_y)
        line_gz.set_data(rel_times, gyro_z)
        ax2.relim()
        ax2.autoscale_view()

        # Mag
        line_mx.set_data(rel_times, mag_x)
        line_my.set_data(rel_times, mag_y)
        line_mz.set_data(rel_times, mag_z)
        ax3.relim()
        ax3.autoscale_view()

        # Shared x-axis
        ax3.set_xlim(0, WINDOW_SECONDS)  # Fix x to 10ss

    return (line_ax, line_ay, line_az, line_gx, line_gy, line_gz, line_mx, line_my, line_mz)


# Set up plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
fig.suptitle('Real-Time IMU Data (Rolling 10s Window)')

# Accel subplot
ax1.set_title('Accelerometer (g)')
ax1.set_ylabel('Acceleration (g)')
line_ax, = ax1.plot([], [], color='blue', label='X')
line_ay, = ax1.plot([], [], color='green', label='Y')
line_az, = ax1.plot([], [], color='red', label='Z')
ax1.legend()
ax1.grid(True)

# Gyro subplot
ax2.set_title('Gyroscope (dps)')
ax2.set_ylabel('Angular Velocity (dps)')
line_gx, = ax2.plot([], [], color='blue', label='X')
line_gy, = ax2.plot([], [], color='green', label='Y')
line_gz, = ax2.plot([], [], color='red', label='Z')
ax2.legend()
ax2.grid(True)

# Mag subplot
ax3.set_title('Magnetometer (μT)')
ax3.set_ylabel('Magnetic Field (μT)')
ax3.set_xlabel('Time (seconds)')
line_mx, = ax3.plot([], [], color='blue', label='X')
line_my, = ax3.plot([], [], color='green', label='Y')
line_mz, = ax3.plot([], [], color='red', label='Z')
ax3.legend()
ax3.grid(True)

# Animation: update every 50ms (~20 Hz), adjust as needed
ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()