import cv2
import numpy as np

from autonomous_nav.config import AppConfig


class DustSimulator:
    """
    Dust simulation module
    """

    def __init__(self, config: AppConfig):
        # Set up config params
        self.frame_w = config.global_.frame_size[0]
        self.frame_h = config.global_.frame_size[1]
        self.map_scale = config.dust_simulator.map_scale
        self.map_h = int(self.frame_h * self.map_scale)
        self.map_w = int(self.frame_w * self.map_scale)
        self.correlation_distance = config.dust_simulator.correlation_distance
        self.vel_x = config.dust_simulator.vel_x
        self.vel_y = config.dust_simulator.vel_y
        self.vel = np.array([self.vel_x, self.vel_y], dtype=float)
        self.dust_intensity = config.dust_simulator.dust_intensity
        self.dust_contrast = config.dust_simulator.dust_contrast
        self.particle_density = config.dust_simulator.particle_density
        self.particle_size = config.dust_simulator.particle_size

        # Precompute large static dust map
        self._generate_large_map()

        # Initial offset
        self.offset = np.array([0.0, 0.0], dtype=float)

    def _generate_large_map(self):
        """
        Computes the large static dust map overlay
        """
        print(f"Generating dust map...")

        # Correlated large-scale dust
        white_noise = np.random.normal(0, 1, (self.map_h, self.map_w))
        kernel_size = max(3, int(self.correlation_distance * 2) | 1)
        correlated = cv2.GaussianBlur(
            white_noise.astype(np.float32),
            (kernel_size, kernel_size),
            sigmaX=self.correlation_distance / 2.0,
        )
        correlated = (correlated - correlated.min()) / (
            correlated.max() - correlated.min() + 1e-8
        )

        # Apply contrast
        if self.dust_contrast != 1.0:
            mean_val = np.mean(correlated)
            correlated = (correlated - mean_val) * self.dust_contrast + mean_val
            correlated = np.clip(correlated, 0.0, 1.0)

        # Apply fine particle layer
        if self.particle_density > 0:
            total_pixels = self.map_h * self.map_w
            expected_particles = int(total_pixels * self.particle_density)

            if expected_particles > 0:
                # Generate random positions across the entire large map
                positions = np.random.uniform(0, 1, (expected_particles, 2))
                positions[:, 0] *= self.map_w
                positions[:, 1] *= self.map_h
                positions = positions.astype(int)

                # Create fine particle intensity map
                fine_map = np.zeros((self.map_h, self.map_w), dtype=np.float32)

                # Insert small Gaussian blobs at each position
                sigma = self.particle_size / 3
                half_size = max(5, int(sigma * 7.0)) // 2 * 2 + 1
                center = half_size // 2
                yy, xx = np.ogrid[-center : center + 1, -center : center + 1]
                dist_sq = xx**2 + yy**2
                kernel = np.exp(-dist_sq / (2 * sigma**2))
                kernel /= kernel.max() + 1e-8  # peak = 1.0

                for py, px in positions:
                    # Calculate intended bounds
                    y_start = max(0, py - center)
                    y_end = min(self.map_h, py + center + 1)
                    x_start = max(0, px - center)
                    x_end = min(self.map_w, px + center + 1)

                    # Skip if the region is empty
                    if y_start >= y_end or x_start >= x_end:
                        continue

                    # Compute corresponding kernel region
                    ky_start = center - (py - y_start)
                    ky_end = ky_start + (y_end - y_start)
                    kx_start = center - (px - x_start)
                    kx_end = kx_start + (x_end - x_start)
                    patch_h = y_end - y_start
                    patch_w = x_end - x_start
                    if ky_end - ky_start != patch_h or kx_end - kx_start != patch_w:
                        continue
                    patch = kernel[ky_start:ky_end, kx_start:kx_end]
                    fine_map[y_start:y_end, x_start:x_end] = np.maximum(
                        fine_map[y_start:y_end, x_start:x_end], patch
                    )

                # Boost particle brightness
                fine_map *= 1.2
                fine_map = np.clip(fine_map, 0, 1)

                # Combine fine map and correlated noise map
                self.large_map = correlated + fine_map
                self.large_map = np.clip(self.large_map, 0, 1)

        # Particle density is zero - just use correlated noise
        else:
            self.large_map = correlated

    def apply_dust(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies dust map to an image
        """
        if frame.shape[:2] != (self.frame_h, self.frame_w):
            raise ValueError("Frame size mismatch")

        # Extract shifted section
        ox, oy = self.offset.astype(int) % (self.map_w, self.map_h)
        dust_gray = np.zeros((self.frame_h, self.frame_w), dtype=np.float32)

        # Handle wrapping
        w1 = min(self.frame_w, self.map_w - ox)
        h1 = min(self.frame_h, self.map_h - oy)
        dust_gray[:h1, :w1] = self.large_map[oy : oy + h1, ox : ox + w1]
        if w1 < self.frame_w:
            dust_gray[:h1, w1:] = self.large_map[oy : oy + h1, 0 : self.frame_w - w1]
        if h1 < self.frame_h:
            dust_gray[h1:, :w1] = self.large_map[0 : self.frame_h - h1, ox : ox + w1]
            if w1 < self.frame_w:
                dust_gray[h1:, w1:] = self.large_map[
                    0 : self.frame_h - h1, 0 : self.frame_w - w1
                ]

        # Create RGB dust overlay. Reddish tint for Mars
        dust_rgb = np.zeros_like(frame, dtype=np.float32)
        dust_rgb[..., 0] = dust_gray * 40
        dust_rgb[..., 1] = dust_gray * 80
        dust_rgb[..., 2] = dust_gray * 140

        # Blend dust onto frame
        dusty = cv2.addWeighted(
            frame.astype(np.float32),
            1.0 - self.dust_intensity,
            dust_rgb,
            self.dust_intensity,
            gamma=0,
        )
        dusty = np.clip(dusty, 0, 255).astype(np.uint8)

        # Update offset for next frame
        self.offset += self.vel
        self.offset %= (self.map_w, self.map_h)

        return dusty
