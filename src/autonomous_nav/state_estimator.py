import numpy as np
from autonomous_nav.config import AppConfig
from scipy.spatial.transform import Rotation as R

import numpy as np


class InertialStateEstimator:
    """
    State estimator module - Extended Kalman Filter
    """

    def __init__(self, config: AppConfig):
        self.config = config
        # State: 16D
        self.state = np.zeros(16)
        # Identity quaternion
        self.state[6] = 1.0

        # Covariance
        self.P = np.eye(16) * 1.0
        self.P[0:3, 0:3] *= 10.0  # position
        self.P[3:6, 3:6] *= 5.0  # velocity
        self.P[6:10, 6:10] *= 0.01  # quat
        self.P[10:13, 10:13] *= 0.1  # accel bias
        self.P[13:16, 13:16] *= 0.1  # gyro bias

        # Process noise
        self.Q = np.diag(
            [
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ]
        )

        # Measurement noise
        self.R_visual = np.diag([0.01, 0.01])
        self.R_lidar = np.array([[0.01]])

        # Convert to cm
        self.gravity = np.array([0.0, 0.0, 9.80665 * 100])

        # Navigation
        self.target_x = config.navigation.planned_route_dx
        self.target_y = config.navigation.planned_route_dy
        self.arrival_inner_threshold_cm = config.navigation.search_zone_outer_thresh

    def normalize_quaternion(self):
        """
        Normalize quaternion
        """
        q = self.state[6:10]
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            self.state[6:10] /= norm

    @property
    def dx_to_search_center(self) -> float:
        """
        Compute remaining x distance to target
        """
        return self.config.navigation.planned_route_dx - self.position[0]

    @property
    def dy_to_search_center(self) -> float:
        """
        Compute remaining y distance to target
        """
        return self.config.navigation.planned_route_dy - self.position[1]

    @property
    def dist_to_search_center(self) -> float:
        """
        Compute remaining euclidean distance to target
        """
        return np.hypot(self.dx_to_search_center, self.dy_to_search_center)

    @property
    def in_landing_mode(self) -> bool:
        """
        Check if in landing mode
        """
        return self.remaining_distance < self.arrival_inner_threshold_cm

    @property
    def position(self):
        """
        Return current position
        """
        return self.state[0:3].copy()

    @property
    def velocity(self):
        """
        Return current velocity
        """
        return self.state[3:6].copy()

    @property
    def quaternion(self):
        """
        Return current quaternion
        """
        return self.state[6:10].copy()

    @property
    def rotation_matrix(self):
        """
        Compute rotation matrix from quaternion
        """
        q = self.quaternion
        # Scipy flips order of quat
        return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """
        EKF prediction step using IMU
        """
        # Extract current state
        pos = self.state[0:3]
        vel = self.state[3:6]
        q = self.state[6:10]
        ba = self.state[10:13]
        bg = self.state[13:16]

        # Bias-corrected measurements
        a_meas = accel - ba
        omega_meas = np.deg2rad(gyro - bg)

        # Rotate acceleration to world frame and remove gravity
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        a_world = rot.apply(a_meas) - self.gravity

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

            # Flip order for Scipy
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
            )
        else:
            q_new = q.copy()

        # Update state
        self.state[0:3] = pos_new
        self.state[3:6] = vel_new
        self.state[6:10] = q_new

        self.normalize_quaternion()

        # Covariance propagation
        F = np.eye(16)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 10:13] = -rot.as_matrix() * dt
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_visual(self, vis_vel_x: float, vis_vel_y: float):
        """
        EKF update step using measurement from visual velocity
        """
        z = np.array([[vis_vel_x], [vis_vel_y]])
        H = np.zeros((2, 16))
        H[0, 3] = 1
        H[1, 4] = 1

        # Residual
        y = z - H @ self.state.reshape(16, 1)
        S = H @ self.P @ H.T + self.R_visual

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state.reshape(16, 1) + K @ y
        self.state = self.state.flatten()

        # Covariance
        self.P = (np.eye(16) - K @ H) @ self.P
        self.normalize_quaternion()

    def update_lidar(self, distance_cm: float):
        """
        EKF update step using measurement from lidar
        """
        z = np.array([[distance_cm]])
        H = np.zeros((1, 16))
        H[0, 2] = 1.0

        # Residual
        y = z - H @ self.state.reshape(16, 1)
        S = H @ self.P @ H.T + self.R_lidar

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state.reshape(16, 1) + K @ y
        self.state = self.state.flatten()

        # Covariance
        self.P = (np.eye(16) - K @ H) @ self.P
        self.normalize_quaternion()

    def reset(self):
        """
        Set the EKF state to default
        """
        self.state.fill(0.0)
        self.state[6] = 1.0
        self.P = np.eye(16) * 1.0
        self.P[6:10, 6:10] *= 0.01
