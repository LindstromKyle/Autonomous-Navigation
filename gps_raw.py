import serial
import time

# Open the serial port (adjust path/baud if needed)
ser = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=1.0)

print("Listening for raw GPS data... (Ctrl+C to stop)")
print("---------------------------------------------")

try:
    while True:
        line = ser.readline().decode('ascii', errors='replace').strip()
        if line:  # Only print non-empty lines
            print(line)
        time.sleep(0.1)  # Small delay to prevent overwhelming the terminal
except KeyboardInterrupt:
    print("\nStopping raw reader.")
finally:
    ser.close()