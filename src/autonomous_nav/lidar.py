import time
import board
import adafruit_vl53l1x

from autonomous_nav.config import LidarConfig


class LidarModule:
    """
    Lidar Module
    """

    def __init__(self, config: LidarConfig):
        self.i2c = board.I2C()
        self.vl53 = adafruit_vl53l1x.VL53L1X(self.i2c)
        self.vl53.distance_mode = config.distance_mode
        self.vl53.timing_budget = config.timing_budget
        self.vl53.start_ranging()
        self.last_time = time.time()

    def read(self) -> dict | None:
        """
        Reads data from lidar sensor
        """

        if self.vl53.data_ready:
            dist_cm = self.vl53.distance
            self.vl53.clear_interrupt()
            if dist_cm is not None:
                dt = time.time() - self.last_time
                self.last_time = time.time()
                return {"distance_cm": dist_cm, "dt": dt}
        return None

    def stop(self):
        """
        Stops the lidar sensor
        """
        self.vl53.stop_ranging()
