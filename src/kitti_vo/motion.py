"""Frame-to-frame camera motion from matched features and a depth map."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_POINTS = 6  # solvePnPRansac needs at least this many


@dataclass
class Motion:
    """Motion of the camera between two frames, plus how well it was supported."""

    rmat: np.ndarray
    tvec: np.ndarray
    n_points: int
    n_inliers: int

    @property
    def transform(self) -> np.ndarray:
        """Motion as a 4x4 matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rmat
        T[:3, 3] = self.tvec.ravel()
        return T


def estimate_motion(
    pts1: np.ndarray,
    pts2: np.ndarray,
    depth: np.ndarray,
    k: np.ndarray,
    max_depth: float = 3000.0,
) -> Motion | None:
    """Solve for the motion that takes the frame-1 points to where they appear in frame 2.

    Points in frame 1 are back-projected to 3D using the depth map, then PnP with
    RANSAC finds the pose that reprojects them onto their frame-2 matches. Returns
    None when there is not enough to solve with, so the caller can hold the last pose.

    ``max_depth`` drops points too far off to trust, which is also what removes
    the unmatched pixels carried through the depth map as very large values.
    """
    height, width = depth.shape
    # Round only to look up the depth map. Back-projection keeps the sub-pixel
    # keypoint coordinates, which is where the 3D points get their precision.
    rows = np.clip(np.round(pts1[:, 1]).astype(int), 0, height - 1)
    cols = np.clip(np.round(pts1[:, 0]).astype(int), 0, width - 1)
    z = depth[rows, cols]

    # Drop points with no depth, or too far away to trust.
    usable = (z > 0) & (z <= max_depth)
    if usable.sum() < MIN_POINTS:
        return None

    u = pts1[usable, 0].astype(np.float64)
    v = pts1[usable, 1].astype(np.float64)
    z = z[usable]
    cx, cy, fx, fy = k[0, 2], k[1, 2], k[0, 0], k[1, 1]
    object_points = np.column_stack([z * (u - cx) / fx, z * (v - cy) / fy, z])
    image_points = pts2[usable]

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, k, None)
    if not ok:
        return None

    return Motion(
        rmat=cv2.Rodrigues(rvec)[0],
        tvec=tvec,
        n_points=len(object_points),
        n_inliers=0 if inliers is None else len(inliers),
    )