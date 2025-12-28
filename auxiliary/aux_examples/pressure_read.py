import time
import board
import adafruit_bmp3xx

# Initialize I2C bus and sensor
i2c = board.I2C()  # Uses default I2C pins on RPi
bmp = adafruit_bmp3xx.BMP3XX_I2C(i2c)

# Optional: Configure oversampling for higher accuracy (default is 1x)
bmp.pressure_oversampling = 8  # Options: 1, 2, 4, 8, 16, 32
bmp.temperature_oversampling = 2  # Options: 1, 2, 4, 8, 16, 32

# Optional: Set IIR filter for noise reduction (default is off)
bmp.iir_filter = 3  # Options: 0 (off), 1, 3, 7, 15, 31, 63, 127

print("BMP388 Sensor Initialized")

while True:
    temperature = bmp.temperature
    pressure = bmp.pressure  # In hPa (hectopascals)
    
    # Optional: Calculate approximate altitude (requires sea-level pressure reference)
    sea_level_pressure_hpa = 1013.25  # Adjust to your local sea-level pressure
    altitude = bmp.altitude  # Uses sea_level_pressure from library (set bmp.sea_level_pressure = value)
    
    print(f"Temperature: {temperature:.2f} °C")
    print(f"Pressure: {pressure:.2f} hPa")
    print(f"Approximate Altitude: {altitude:.2f} meters")
    
    time.sleep(1)  # Read every second