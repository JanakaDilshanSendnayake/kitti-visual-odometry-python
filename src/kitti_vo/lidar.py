"""Projection of Velodyne scans into the camera image."""

from __future__ import annotations

import numpy as np


def project_to_image(
    pointcloud: np.ndarray,
    imheight: int,
    imwidth: int,
    Tr: np.ndarray,
    P0: np.ndarray,
) -> np.ndarray:
    """Render a scan as sparse depth, float32 (H, W), 0 where no point landed."""
    xyz = pointcloud[pointcloud[:, 0] > 0, :3]  # drop points behind the sensor
    cam = Tr @ np.hstack([xyz, np.ones((len(xyz), 1))]).T
    cam = cam[:, cam[2] > 0]

    render = np.zeros((imheight, imwidth), dtype=np.float32)
    if cam.shape[1] == 0:
        return render

    depth = cam[2].copy()
    pixels = np.round((P0 @ np.vstack([cam / cam[2], np.ones(cam.shape[1])]))[:2].T)
    u, v = pixels.astype(np.int32).T

    inside = (u >= 0) & (u < imwidth) & (v >= 0) & (v < imheight)
    u, v, depth = u[inside], v[inside], depth[inside]

    # Nearest point wins where several land on one pixel.
    buffer = np.full((imheight, imwidth), np.inf, dtype=np.float32)
    np.minimum.at(buffer, (v, u), depth)
    return np.where(np.isinf(buffer), 0.0, buffer)


def fuse_depth(stereo_depth: np.ndarray, lidar_depth: np.ndarray) -> np.ndarray:
    """Use LiDAR depth where it exists, stereo everywhere else."""
    fused = stereo_depth.copy()
    measured = lidar_depth > 0
    fused[measured] = lidar_depth[measured]
    return fused
