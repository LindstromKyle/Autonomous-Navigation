import serial
import pynmea2
import time

# Open the serial port
ser = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=1.0)

try:
    while True:
        try:
            line = ser.readline().decode('ascii', errors='replace').strip()
            if line.startswith('$'):
                msg = pynmea2.parse(line)
                if isinstance(msg, pynmea2.types.talker.GGA):  # Use GGA for position data
                    print(f"Latitude: {msg.latitude} {msg.lat_dir}")
                    print(f"Longitude: {msg.longitude} {msg.lon_dir}")
                    print(f"Altitude: {msg.altitude} {msg.altitude_units}")
                    print(f"Satellites: {msg.num_sats}")
                    print("---")
                elif isinstance(msg, pynmea2.types.talker.RMC):  # Use RMC for speed/date
                    print(f"Speed: {msg.spd_over_grnd} knots")
                    print(f"Date/Time: {msg.timestamp} {msg.datestamp}")
                    print("---")
#                 elif isinstance(msg, pynmea2.types.talker.GSA):
#                     print(f"Fix Mode: {msg.mode_fix_type}")
#                     print(f"PDOP: {msg.pdop}, HDOP: {msg.hdop}, VDOP: {msg.vdop}")
#                     print(f"Satellites Used: {msg.sv_prn_num_1} {msg.sv_prn_num_2} ...")  # List all PRNs
#                     print("---")
#                 elif isinstance(msg, pynmea2.types.talker.GSV):
#                     print(f"Satellites in View: {msg.num_sv_in_view}")
#                     # For each satellite in the message (up to 4 per GSV sentence)
#                     # print(f"Sat 1: PRN {msg.sv_prn_num_1}, Elev {msg.elevation_deg_1}, Azim {msg.azimuth_1}, SNR {msg.snr_1}")
#                     # Repeat for sats 2-4 if present
#                     print("---")
#                 elif isinstance(msg, pynmea2.types.talker.VTG):
#                     print(f"Course (True): {msg.true_track}")
#                     print(f"Course (Magnetic): {msg.mag_track}")
#                     print(f"Speed (km/h): {msg.spd_over_grnd_kmph}")
#                     print("---")
        except pynmea2.ParseError:
            pass  # Ignore invalid sentences
        #time.sleep(0.5)  # Throttle to avoid flooding output
except KeyboardInterrupt:
    print("Stopping GPS reader.")
finally:
    ser.close()