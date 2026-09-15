"""Per-frame visual odometry: depth, features, motion, accumulated pose."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .dataset import KittiSequence
from .features import FeatureTracker, matched_points
from .lidar import fuse_depth, project_to_image
from .motion import estimate_motion
from .stereo import StereoDepth, feature_mask


@dataclass
class FrameResult:
    """Everything the dashboard draws for one frame."""

    index: int
    pose: np.ndarray  # 4x4, camera to world
    depth: np.ndarray
    lidar_depth: np.ndarray | None
    points_prev: np.ndarray  # matched pixels in this frame, (N, 2)
    points_curr: np.ndarray  # where they moved to in the next frame
    n_points: int
    n_inliers: int
    elapsed: float
    tracking_lost: bool = False


@dataclass
class VOConfig:
    detector: str = "sift"
    matching: str = "BF"
    ratio: float = 0.7
    stereo_matcher: str = "sgbm"
    num_disparities: int = 96
    block_size: int = 11
    max_depth: float = 50.0
    use_lidar: bool = True


class VisualOdometry:
    """Runs the pipeline over a sequence, one frame at a time.

    ``run`` is a generator, so the dashboard can draw each frame as it arrives and
    the runner can save them. Motion is estimated between frame i and i+1, so a
    sequence of N frames yields N-1 results after the initial pose.
    """

    def __init__(self, sequence: KittiSequence, config: VOConfig | None = None) -> None:
        self.seq = sequence
        self.config = config or VOConfig()
        c = sequence.calib

        self.stereo = StereoDepth(
            c.focal_length,
            c.baseline,
            matcher=self.config.stereo_matcher,
            num_disparities=self.config.num_disparities,
            block_size=self.config.block_size,
        )
        self.tracker = FeatureTracker(self.config.detector, self.config.matching, self.config.ratio)
        self.mask = feature_mask(sequence.imheight, sequence.imwidth, self.config.num_disparities)
        self.use_lidar = self.config.use_lidar and sequence.has_lidar

    def run(self, start: int = 0, end: int | None = None) -> Iterator[FrameResult]:
        """Yield one FrameResult per frame, from ``start`` up to ``end``."""
        end = len(self.seq) if end is None else min(end, len(self.seq))
        pose = np.eye(4)

        image = self.seq.load_left(start)
        kp, des = self.tracker.detect(image, self.mask)

        for i in range(start, end - 1):
            t0 = time.perf_counter()

            depth = self.stereo.depth(image, self.seq.load_right(i))
            lidar_depth = None
            if self.use_lidar:
                lidar_depth = project_to_image(
                    self.seq.load_pointcloud(i),
                    self.seq.imheight,
                    self.seq.imwidth,
                    self.seq.calib.Tr,
                    self.seq.calib.P0,
                )
                depth = fuse_depth(depth, lidar_depth)

            next_image = self.seq.load_left(i + 1)
            next_kp, next_des = self.tracker.detect(next_image, self.mask)
            matches = self.tracker.match(des, next_des)
            pts_prev, pts_curr = matched_points(matches, kp, next_kp)

            motion = estimate_motion(
                pts_prev, pts_curr, depth, self.seq.calib.k_left, self.config.max_depth
            )
            if motion is None:
                # Not enough to solve with. Hold the last pose rather than jump.
                n_points = n_inliers = 0
                lost = True
            else:
                pose = pose @ np.linalg.inv(motion.transform)
                n_points, n_inliers = motion.n_points, motion.n_inliers
                lost = False

            yield FrameResult(
                index=i,
                pose=pose.copy(),
                depth=depth,
                lidar_depth=lidar_depth,
                points_prev=pts_prev,
                points_curr=pts_curr,
                n_points=n_points,
                n_inliers=n_inliers,
                elapsed=time.perf_counter() - t0,
                tracking_lost=lost,
            )

            image, kp, des = next_image, next_kp, next_des


def trajectory_error(estimated: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Distance between estimated and true camera position at each frame, in metres."""
    n = min(len(estimated), len(ground_truth))
    return np.linalg.norm(estimated[:n, :3, 3] - ground_truth[:n, :3, 3], axis=1)