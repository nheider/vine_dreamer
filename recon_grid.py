"""recon_grid.py -- Sparse voxel grid for vine wood reconstruction.

Maintains a sparse set of revealed wood voxels and a ground-truth set built
from the generator's capsule geometry.  The per-step reward signal is the
count of newly revealed voxels (first-visit only, no double counting).

At 2 cm resolution a full vine stand has ~5k-15k occupied wood voxels,
so Python sets are fast enough for per-step computation.
"""

from __future__ import annotations
import numpy as np


class ReconGrid:
    """Sparse binary occupancy grid for revealed vine wood structure."""

    def __init__(self, voxel_size: float = 0.02):
        self.voxel_size     = voxel_size
        self._inv_vs        = 1.0 / voxel_size
        self.revealed:      set[tuple[int, int, int]] = set()
        self.ground_truth:  set[tuple[int, int, int]] = set()
        self.total_revealed = 0

    # ------------------------------------------------------------------ GT --

    def set_ground_truth(self, trunk_segs, shoot_segs):
        """Build the ground-truth voxel set from generator capsule segments.

        trunk_segs : list of ((x0,y0,z0), (x1,y1,z1), radius)
        shoot_segs : list of ((x0,y0,z0), (x1,y1,z1))
        """
        self.ground_truth.clear()
        for p0, p1, radius in trunk_segs:
            self._voxelise_capsule(p0, p1, radius)
        for p0, p1 in shoot_segs:
            self._voxelise_capsule(p0, p1, radius=0.008)

    def _voxelise_capsule(self, p0, p1, radius: float):
        a      = np.asarray(p0, dtype=np.float64)
        b      = np.asarray(p1, dtype=np.float64)
        length = np.linalg.norm(b - a)
        if length < 1e-6:
            return

        direction = (b - a) / length
        step      = self.voxel_size

        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, [0., 0., 1.])
        else:
            perp1 = np.cross(direction, [1., 0., 0.])
        perp1 /= np.linalg.norm(perp1)
        perp2  = np.cross(direction, perp1)

        n_along  = max(2, int(np.ceil(length / step)) + 1)
        n_radial = max(1, int(np.ceil(radius / step)))

        for i in range(n_along):
            t      = i / (n_along - 1)
            center = a + direction * length * t
            self.ground_truth.add(self._vi(center))
            for ri in range(1, n_radial + 1):
                r       = radius * ri / n_radial
                n_angle = max(4, int(np.ceil(2 * np.pi * r / step)))
                for ai in range(n_angle):
                    theta = 2 * np.pi * ai / n_angle
                    pt    = center + perp1 * (r * np.cos(theta)) \
                                   + perp2 * (r * np.sin(theta))
                    self.ground_truth.add(self._vi(pt))

    def _vi(self, pt) -> tuple[int, int, int]:
        inv = self._inv_vs
        return (int(pt[0] * inv), int(pt[1] * inv), int(pt[2] * inv))

    # --------------------------------------------------------------- reset --

    def reset(self):
        self.revealed.clear()
        self.total_revealed = 0

    # ---------------------------------------------------------------- fuse --

    def fuse(
        self,
        depth:     np.ndarray,  # (H, W)  depth in metres
        cam_pos:   np.ndarray,  # (3,)    camera world position
        cam_rot:   np.ndarray,  # (3, 3)  camera-to-world rotation
        fov_y_deg: float,       # vertical field of view in degrees
        max_depth: float = 3.0,
    ) -> int:
        """Back-project depth pixels into world-frame wood voxels.

        The scene contains only structural wood geometry, so every valid
        depth pixel is a wood observation.  Returns the number of newly
        revealed ground-truth voxels this step.
        """
        H, W  = depth.shape
        valid = (depth > 0.01) & (depth < max_depth) & np.isfinite(depth)
        if not np.any(valid):
            return 0

        vs, us = np.where(valid)
        ds     = depth[vs, us]

        fy     = H / (2.0 * np.tan(np.deg2rad(fov_y_deg) / 2.0))
        fx     = fy                          # square pixels
        cx, cy = W / 2.0, H / 2.0

        # Back-project to camera frame (OpenGL convention: -Z forward, Y up)
        x_cam =  (us - cx) / fx * ds
        y_cam = -(vs - cy) / fy * ds
        z_cam = -ds

        pts_world = (cam_rot @ np.stack([x_cam, y_cam, z_cam])).T + cam_pos

        inv = self._inv_vs
        ix  = (pts_world[:, 0] * inv).astype(np.int64)
        iy  = (pts_world[:, 1] * inv).astype(np.int64)
        iz  = (pts_world[:, 2] * inv).astype(np.int64)

        new_voxels = (
            {(int(ix[k]), int(iy[k]), int(iz[k])) for k in range(len(ix))}
            - self.revealed
        )
        self.revealed      |= new_voxels
        self.total_revealed = len(self.revealed & self.ground_truth)
        return len(new_voxels)

    # ---------------------------------------------------------- properties --

    @property
    def coverage(self) -> float:
        """Fraction of ground-truth wood voxels revealed so far."""
        if not self.ground_truth:
            return 0.0
        return len(self.revealed & self.ground_truth) / len(self.ground_truth)

    def as_points(self) -> np.ndarray:
        """Revealed voxels as (N, 3) world-frame point cloud."""
        if not self.revealed:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array(list(self.revealed), dtype=np.float64) * self.voxel_size

    def render_topdown(self, size: int = 128) -> np.ndarray:
        """Render a top-down (X-Z) reconstruction image.

        Returns an (size, size, 3) uint8 RGB array:
          * dark gray  — ground-truth voxels not yet revealed
          * green      — revealed ground-truth voxels
        """
        img = np.zeros((size, size, 3), dtype=np.uint8)
        if not self.ground_truth:
            return img

        coords = np.array(list(self.ground_truth), dtype=np.float64)
        xs, zs = coords[:, 0], coords[:, 2]
        x_min, x_max = xs.min() - 1, xs.max() + 1
        z_min, z_max = zs.min() - 1, zs.max() + 1
        span = max(x_max - x_min, z_max - z_min, 1)

        def _px(vx, vz):
            col = int((vx - x_min) / span * (size - 1))
            row = int((1.0 - (vz - z_min) / span) * (size - 1))  # Z up
            return np.clip(row, 0, size - 1), np.clip(col, 0, size - 1)

        # Ground truth not yet revealed (dark gray)
        for vx, _, vz in self.ground_truth:
            r, c = _px(vx, vz)
            img[r, c] = [60, 60, 60]

        # Revealed ground-truth voxels (green)
        for vx, _, vz in (self.revealed & self.ground_truth):
            r, c = _px(vx, vz)
            img[r, c] = [0, 220, 60]

        return img
