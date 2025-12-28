from config import AppConfig
from utils import pixels_to_cm


class PositionEstimator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pos_x: float = 0.0
        self.pos_y: float = 0.0

    def update(self, flow_dx_px: float, flow_dy_px: float):
        dx_cm = -pixels_to_cm(flow_dx_px, self.config.global_.pixels_per_cm)
        dy_cm = pixels_to_cm(flow_dy_px, self.config.global_.pixels_per_cm)
        self.pos_x += dx_cm
        self.pos_y += dy_cm

    def reset(self):
        self.pos_x = self.pos_y = 0.0

    @property
    def position(self) -> tuple[float, float]:
        return self.pos_x, self.pos_y
