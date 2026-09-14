"""Reader for one KITTI odometry sequence."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration


class KittiSequence:
    """Reads images, point clouds and poses by frame index, straight from disk.

    Frames load on demand, so playback can seek in either direction and opening a
    sequence is cheap. Listings are sorted because filesystem order is not guaranteed.
    """

    def __init__(self, root: str | Path, sequence: str = "00") -> None:
        self.root = Path(root)
        self.sequence = sequence
        self.seq_dir = self.root / "sequences" / sequence
        if not self.seq_dir.is_dir():
            raise FileNotFoundError(f"no such sequence: {self.seq_dir}")

        self.left_files = sorted((self.seq_dir / "image_0").glob("*.png"))
        self.right_files = sorted((self.seq_dir / "image_1").glob("*.png"))
        self.velodyne_files = sorted((self.seq_dir / "velodyne").glob("*.bin"))

        if not self.left_files:
            raise FileNotFoundError(f"no images in {self.seq_dir / 'image_0'}")
        if len(self.right_files) != len(self.left_files):
            raise ValueError(
                f"sequence {sequence}: {len(self.left_files)} left vs "
                f"{len(self.right_files)} right images"
            )

        self.calib = Calibration.from_file(self.seq_dir / "calib.txt")

    def __len__(self) -> int:
        return len(self.left_files)

    def __repr__(self) -> str:
        return (
            f"KittiSequence({self.sequence!r}, frames={len(self)}, "
            f"size={self.imwidth}x{self.imheight}, lidar={self.has_lidar}, "
            f"gt={self.has_ground_truth})"
        )

    @property
    def has_lidar(self) -> bool:
        return len(self.velodyne_files) > 0

    @property
    def has_ground_truth(self) -> bool:
        return len(self.poses) > 0

    @cached_property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) of the left images."""
        return self.load_left(0).shape[:2]

    @property
    def imheight(self) -> int:
        return self.image_shape[0]

    @property
    def imwidth(self) -> int:
        return self.image_shape[1]

    @cached_property
    def poses(self) -> np.ndarray:
        """Ground truth as (N, 4, 4). Empty for sequences 11-21, which have none."""
        path = self.root / "poses" / f"{self.sequence}.txt"
        if not path.exists():
            return np.empty((0, 4, 4))

        rows = np.loadtxt(path).reshape(-1, 3, 4)
        poses = np.tile(np.eye(4), (len(rows), 1, 1))
        poses[:, :3, :] = rows
        return poses

    @cached_property
    def timestamps(self) -> np.ndarray:
        """Frame times in seconds, used to play back at the real rate."""
        path = self.seq_dir / "times.txt"
        return np.loadtxt(path) if path.exists() else np.empty(0)

    def load_left(self, index: int) -> np.ndarray:
        """Left grey image, uint8 (H, W)."""
        return self._imread(self.left_files[index])

    def load_right(self, index: int) -> np.ndarray:
        """Right grey image, uint8 (H, W)."""
        return self._imread(self.right_files[index])

    def load_stereo_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.load_left(index), self.load_right(index)

    def load_pointcloud(self, index: int) -> np.ndarray:
        """Velodyne scan as (N, 4): x forward, y left, z up, reflectance."""
        if not self.has_lidar:
            raise FileNotFoundError(f"sequence {self.sequence} has no velodyne data")
        return np.fromfile(self.velodyne_files[index], dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def _imread(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"could not read {path}")
        return image