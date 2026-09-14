"""Camera and LiDAR calibration for a KITTI sequence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CALIB_KEYS = ("P0", "P1", "P2", "P3", "Tr")


def decompose_projection_matrix(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 3x4 projection matrix into intrinsics, rotation and camera centre."""
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]  # OpenCV returns t homogeneous
    return k, r, t


@dataclass
class Calibration:
    """P0/P1 are the grey stereo pair, P2/P3 the colour pair, Tr maps LiDAR to camera 0."""

    P0: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    P3: np.ndarray
    Tr: np.ndarray

    def __post_init__(self) -> None:
        # Derived once here. These never change for a sequence.
        k_left, _, t_left = decompose_projection_matrix(self.P0)
        _, _, t_right = decompose_projection_matrix(self.P1)
        self.k_left = k_left
        self.focal_length = float(k_left[0, 0])
        self.baseline = float(t_right[0, 0] - t_left[0, 0])  # metres

    @classmethod
    def from_file(cls, path: str | Path) -> "Calibration":
        """Read calib.txt. Each row is 'KEY: v1 ... v12'."""
        mats = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            key, _, values = line.partition(":")
            mats[key.strip()] = np.fromstring(values, sep=" ").reshape(3, 4)

        missing = set(CALIB_KEYS) - mats.keys()
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        return cls(*(mats[k] for k in CALIB_KEYS))

    @property
    def Tr_homogeneous(self) -> np.ndarray:
        """Tr as 4x4, so it composes with 4x4 poses."""
        return np.vstack([self.Tr, [0.0, 0.0, 0.0, 1.0]])