import time
import board
import adafruit_vl53l1x

i2c = board.I2C()
vl53 = adafruit_vl53l1x.VL53L1X(i2c)

# Long distance mode (up to ~4 m real range)
vl53.distance_mode = 2
# 100 ms timing budget = good accuracy vs speed
vl53.timing_budget = 100

vl53.start_ranging()
print("VL53L1X ranging started — point at objects 10-350 cm away\n")

try:
    while True:
        if vl53.data_ready:
            dist_cm = vl53.distance
            vl53.clear_interrupt()

            if dist_cm is None:
                print(
                    "No valid reading (out of range, reflective surface, or too close)"
                )
            else:
                print(f"Distance: {dist_cm:.1f} cm")

        time.sleep(0.05)  # ~20 Hz polling

except KeyboardInterrupt:
    vl53.stop_ranging()
    print("\nStopped")
