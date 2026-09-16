"""Frame sources for the dashboard: cached replay, or a live pipeline run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..dataset import KittiSequence
from ..pipeline import VisualOdometry, VOConfig
from ..runner import load_depth


@dataclass
class FrameView:
    """One frame's worth of everything the panels draw."""

    index: int
    image: np.ndarray
    depth: np.ndarray | None
    pointcloud: np.ndarray | None
    points_prev: np.ndarray
    points_curr: np.ndarray
    pose: np.ndarray
    n_points: int
    n_inliers: int
    elapsed: float


class ReplaySource:
    """Reads a finished run from disk. Supports seeking to any frame."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        meta = json.loads((self.run_dir / "config.json").read_text())
        self.meta = meta
        self.start = meta["start"]
        self.cached = meta["cached_frames"]
        self.frames_dir = self.run_dir / "frames"

        self.seq = KittiSequence(meta["root"], meta["sequence"])
        self.trajectory = np.load(self.run_dir / "trajectory.npy")
        self.metrics = np.loadtxt(self.run_dir / "metrics.csv", delimiter=",", skiprows=1, ndmin=2)

    def __len__(self) -> int:
        return len(self.trajectory)

    @property
    def ground_truth(self) -> np.ndarray:
        """Ground-truth poses aligned with the estimated trajectory."""
        if not self.seq.has_ground_truth:
            return np.empty((0, 4, 4))
        return self.seq.poses[self.start + 1 : self.start + 1 + len(self)]

    def __getitem__(self, i: int) -> FrameView:
        row = self.metrics[i]
        frame = int(row[0])

        depth = points_prev = points_curr = None
        if self.cached:
            depth_path = self.frames_dir / f"depth_{frame:06d}.png"
            if depth_path.exists():
                depth = load_depth(depth_path)
            npz_path = self.frames_dir / f"points_{frame:06d}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                points_prev, points_curr = data["prev"], data["curr"]

        empty = np.empty((0, 2), dtype=np.float32)
        return FrameView(
            index=frame,
            image=self.seq.load_left(frame),
            depth=depth,
            pointcloud=self.seq.load_pointcloud(frame) if self.seq.has_lidar else None,
            points_prev=empty if points_prev is None else points_prev,
            points_curr=empty if points_curr is None else points_curr,
            pose=self.trajectory[i],
            n_points=int(row[1]),
            n_inliers=int(row[2]),
            elapsed=float(row[3]),
        )


class LiveSource(QThread):
    """Runs the pipeline in a worker thread, emitting frames as they finish.

    Playback is bound by compute, and frames arrive in order only, so the
    dashboard disables seeking while this is the active source.
    """

    frame_ready = pyqtSignal(object)
    finished_run = pyqtSignal()

    def __init__(self, sequence: KittiSequence, config: VOConfig, start: int = 0,
                 end: int | None = None) -> None:
        super().__init__()
        self.seq = sequence
        self.config = config
        self.start_frame = start
        self.end = end
        self._stop = False

    @property
    def ground_truth(self) -> np.ndarray:
        if not self.seq.has_ground_truth:
            return np.empty((0, 4, 4))
        end = len(self.seq) if self.end is None else self.end
        return self.seq.poses[self.start_frame + 1 : end]

    def stop(self) -> None:
        self._stop = True
        self.wait()

    def run(self) -> None:
        vo = VisualOdometry(self.seq, self.config)
        for r in vo.run(self.start_frame, self.end):
            if self._stop:
                break
            self.frame_ready.emit(
                FrameView(
                    index=r.index,
                    image=self.seq.load_left(r.index),
                    depth=r.depth,
                    pointcloud=(
                        self.seq.load_pointcloud(r.index) if self.seq.has_lidar else None
                    ),
                    points_prev=r.points_prev,
                    points_curr=r.points_curr,
                    pose=r.pose,
                    n_points=r.n_points,
                    n_inliers=r.n_inliers,
                    elapsed=r.elapsed,
                )
            )
        self.finished_run.emit()