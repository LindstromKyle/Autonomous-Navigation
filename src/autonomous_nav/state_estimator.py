import numpy as np
from autonomous_nav.config import AppConfig
from scipy.spatial.transform import Rotation as R


import numpy as np


class StateEstimator:
    def __init__(self, config):
        self.config = config

        self.state = np.zeros(6)

        # Covariance (initial uncertainty)
        self.P = np.eye(6) * 1.0
        self.P[0:3, 0:3] *= 10.0  # Higher initial pos uncertainty
        self.P[3:6, 3:6] *= 5.0  # Velocity

        # Process noise (tune these!) - assuming constant velocity model
        self.Q = np.diag(
            [
                1000,
                1000,
                1000,  # pos (integrated noise)
                1000,
                1000,
                1000,  # vel
            ]
        )

        # Measurement noise
        self.R_visual = np.diag([0.01, 0.01])
        self.R_lidar = np.array([[0.01]])  # lidar distance (cm)

    @property
    def dx_to_search_center(self) -> float:
        return self.config.navigation.planned_route_dx - self.position[0]

    @property
    def dy_to_search_center(self) -> float:
        return self.config.navigation.planned_route_dy - self.position[1]

    @property
    def dist_to_search_center(self) -> float:
        return np.hypot(self.dx_to_search_center, self.dy_to_search_center)

    @property
    def in_landing_mode(self) -> bool:
        return (
            self.dist_to_search_center
            < self.config.navigation.arrival_inner_threshold_cm
        )

    @property
    def position(self):
        return self.state[0:3].copy()

    @property
    def velocity(self):
        return self.state[3:6].copy()

    def predict(self, dt: float):
        """Prediction step assuming constant velocity (no IMU)"""
        pos = self.state[0:3]
        vel = self.state[3:6]

        # Integrate position
        pos_new = pos + vel * dt

        # Update state
        self.state[0:3] = pos_new

        # Jacobian F for covariance propagation
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt

        self.P = F @ self.P @ F.T + self.Q * dt

    def update_visual(self, vis_vel_x: float, vis_vel_y: float):
        z = np.array([[vis_vel_x], [vis_vel_y]])  # (2,1)
        H = np.zeros((2, 6))
        H[0, 3] = 1
        H[1, 4] = 1

        y = z - H @ self.state.reshape(6, 1)
        S = H @ self.P @ H.T + self.R_visual
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state.reshape(6, 1) + K @ y
        self.state = self.state.flatten()
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_lidar(self, distance_cm: float):
        z = np.array([[distance_cm]])
        H = np.zeros((1, 6))
        H[0, 2] = 1.0

        y = z - H @ self.state.reshape(6, 1)
        S = H @ self.P @ H.T + self.R_lidar
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state.reshape(6, 1) + K @ y
        self.state = self.state.flatten()
        self.P = (np.eye(6) - K @ H) @ self.P

    def reset(self):
        self.state.fill(0.0)
        self.P = np.eye(6) * 1.0
        self.P[0:3, 0:3] *= 10.0
        self.P[3:6, 3:6] *= 5.0


class InertialStateEstimator:
    def __init__(self, config):
        # State: 16D
        self.state = np.zeros(16)
        self.state[6] = 1.0  # q_w = 1.0 (identity quaternion)

        # Covariance (initial uncertainty)
        self.P = np.eye(16) * 1.0
        self.P[0:3, 0:3] *= 10.0  # Higher initial pos uncertainty
        self.P[3:6, 3:6] *= 5.0  # Velocity
        self.P[6:10, 6:10] *= 0.01  # Tight on orientation
        self.P[10:13, 10:13] *= 0.1  # Accel bias
        self.P[13:16, 13:16] *= 0.1  # Gyro bias

        # Process noise (tune these!)
        self.Q = np.diag(
            [
                1000,
                1000,
                1000,  # pos
                1000,
                1000,
                1000,  # vel
                0.1,
                0.1,
                0.1,
                0.1,  # quat
                0.1,
                0.1,
                0.1,  # accel bias
                0.1,
                0.1,
                0.1,  # gyro bias
            ]
        )

        # Measurement noise
        self.R_visual = np.diag([0.01, 0.01])
        self.R_lidar = np.array([[0.01]])  # lidar distance (cm)

        self.gravity = np.array([0.0, 0.0, 9.80665 * 100])  # cm/s²

        # Navigation
        self.target_x = config.navigation.target_offset_x_cm
        self.target_y = config.navigation.target_offset_y_cm
        self.arrival_inner_threshold_cm = config.navigation.arrival_inner_threshold_cm

    def normalize_quaternion(self):
        q = self.state[6:10]
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            self.state[6:10] /= norm

    @property
    def remaining_x(self) -> float:
        return self.target_x - self.position[0]

    @property
    def remaining_y(self) -> float:
        return self.target_y - self.position[1]

    @property
    def remaining_distance(self) -> float:
        return np.hypot(self.remaining_x, self.remaining_y)

    @property
    def in_landing_mode(self) -> bool:
        return self.remaining_distance < self.arrival_inner_threshold_cm

    @property
    def position(self):
        return self.state[0:3].copy()

    @property
    def velocity(self):
        return self.state[3:6].copy()

    @property
    def quaternion(self):
        return self.state[6:10].copy()  # [qw, qx, qy, qz]

    @property
    def rotation_matrix(self):
        q = self.quaternion
        return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # x,y,z,w order

    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """EKF prediction step using IMU"""
        # Extract current state
        pos = self.state[0:3]
        vel = self.state[3:6]
        q = self.state[6:10]
        ba = self.state[10:13]
        bg = self.state[13:16]

        # Bias-corrected measurements
        # a_meas = accel - ba
        a_meas = accel
        # omega_meas = np.deg2rad(gyro - bg)
        omega_meas = np.deg2rad(gyro)

        # Rotate acceleration to world frame and remove gravity
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        a_world = rot.apply(a_meas) - self.gravity
        print(f"DEBUG: a_read = {a_meas}")
        print(f"DEBUG: a_world minus gravity = {a_world}")

        # Integrate linear motion
        vel_new = vel + a_world * dt
        pos_new = pos + vel * dt + 0.5 * a_world * dt**2

        # Integrate quaternion
        if np.linalg.norm(omega_meas) > 1e-8:
            angle = np.linalg.norm(omega_meas) * dt
            axis = omega_meas / np.linalg.norm(omega_meas)
            # [w, x, y, z]
            dq = np.array(
                [
                    np.cos(angle / 2),
                    axis[0] * np.sin(angle / 2),
                    axis[1] * np.sin(angle / 2),
                    axis[2] * np.sin(angle / 2),
                ]
            )

            # [x,y,z,w] for SciPy
            dq_quat = np.array([dq[1], dq[2], dq[3], dq[0]])
            dq_rot = R.from_quat(dq_quat)
            q_new_rot = rot * dq_rot

            q_new = np.array(
                [
                    q_new_rot.as_quat()[3],
                    q_new_rot.as_quat()[0],
                    q_new_rot.as_quat()[1],
                    q_new_rot.as_quat()[2],
                ]
            )  # z
        else:
            q_new = q.copy()

        # Update state
        self.state[0:3] = pos_new
        self.state[3:6] = vel_new
        self.state[6:10] = q_new

        self.normalize_quaternion()

        # Simplified Jacobian F for covariance propagation
        F = np.eye(16)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 10:13] = -rot.as_matrix() * dt  # accel bias effect

        self.P = F @ self.P @ F.T + self.Q * dt

    def update_visual(self, vis_vel_x: float, vis_vel_y: float):
        z = np.array([[vis_vel_x], [vis_vel_y]])  # (2,1)
        H = np.zeros((2, 16))
        H[0, 3] = 1
        H[1, 4] = 1

        y = z - H @ self.state.reshape(16, 1)
        S = H @ self.P @ H.T + self.R_visual
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state.reshape(16, 1) + K @ y
        self.state = self.state.flatten()
        self.P = (np.eye(16) - K @ H) @ self.P
        self.normalize_quaternion()

    def update_lidar(self, distance_cm: float):
        z = np.array([[distance_cm]])
        H = np.zeros((1, 16))
        H[0, 2] = 1.0

        y = z - H @ self.state.reshape(16, 1)
        S = H @ self.P @ H.T + self.R_lidar
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state.reshape(16, 1) + K @ y
        self.state = self.state.flatten()
        self.P = (np.eye(16) - K @ H) @ self.P
        self.normalize_quaternion()

    def reset(self):
        self.state.fill(0.0)
        self.state[6] = 1.0
        self.P = np.eye(16) * 1.0
        self.P[6:10, 6:10] *= 0.01
