import numpy as np
from autonomous_nav.config import AppConfig
from autonomous_nav.utils import pixels_to_cm


class PositionEstimator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pos_x: float = 0.0
        self.pos_y: float = 0.0
        self.vel_x: float = 0.0  # New: cm/s
        self.vel_y: float = 0.0  # New: cm/s

        # Kalman state: [pos_x, pos_y, vel_x, vel_y] in cm, cm/s
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1000  # Initial uncertainty

        # Process noise (tune: high for IMU)
        self.Q = np.diag([0.01, 0.01, 1.0, 1.0])
        # Measurement noise (tune: lower for visual)
        self.R = np.diag([1.0, 1.0])  # For vel_x, vel_y measurements

        # Navigation
        self.target_x = config.navigation.target_offset_x_cm
        self.target_y = config.navigation.target_offset_y_cm
        self.arrival_threshold_cm = config.navigation.arrival_threshold_cm

    @property
    def remaining_x(self) -> float:
        return self.target_x - self.pos_x

    @property
    def remaining_y(self) -> float:
        return self.target_y - self.pos_y

    @property
    def remaining_distance(self) -> float:
        return np.hypot(self.remaining_x, self.remaining_y)

    @property
    def in_landing_mode(self) -> bool:
        return self.remaining_distance < self.arrival_threshold_cm

    def predict(self, accel: np.array, dt: float):
        # Accel in cm/s² (convert from m/s²)
        a_x, a_y = accel[0] * 100, -1 * accel[1] * 100  # Adjust axes if needed

        # State transition matrix (constant vel + accel input)
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        # Control input matrix for accel
        B = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])
        u = np.array([[a_x], [a_y]])

        self.x = A @ self.x + B @ u
        self.P = A @ self.P @ A.T + self.Q

    def update(self, vis_vel_x: float, vis_vel_y: float):
        # Measurement: vel from visual (cm/s)
        z = np.array([[vis_vel_x], [vis_vel_y]])
        H = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])  # Observe velocities

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        self.pos_x, self.pos_y, self.vel_x, self.vel_y = self.x.flatten()

    def reset(self):
        self.x.fill(0)
        self.P = np.eye(4) * 1000

    @property
    def position(self) -> tuple[float, float]:
        return self.pos_x, self.pos_y
