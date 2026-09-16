"""CLI that runs a sequence and saves the result for replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .dataset import KittiSequence
from .pipeline import VisualOdometry, VOConfig, trajectory_error

# Depth is stored as uint16 PNG. 128 steps per metre keeps sub-centimetre
# precision and still reaches 511 m, past anything stereo reports.
DEPTH_SCALE = 128.0


def save_depth(path: Path, depth: np.ndarray) -> None:
    scaled = np.clip(depth * DEPTH_SCALE, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), scaled)


def load_depth(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE


def run(
    root: str,
    sequence: str,
    out_dir: Path,
    config: VOConfig,
    start: int = 0,
    end: int | None = None,
    cache_frames: bool = True,
) -> Path:
    """Process a sequence and write the results to ``out_dir``."""
    seq = KittiSequence(root, sequence)
    vo = VisualOdometry(seq, config)
    end = len(seq) if end is None else min(end, len(seq))

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if cache_frames:
        frames_dir.mkdir(exist_ok=True)

    poses, rows = [], []
    for r in tqdm(vo.run(start, end), total=end - start - 1, unit="frame"):
        poses.append(r.pose)
        rows.append((r.index, r.n_points, r.n_inliers, r.elapsed, r.tracking_lost))
        if cache_frames:
            save_depth(frames_dir / f"depth_{r.index:06d}.png", r.depth)
            np.savez_compressed(
                frames_dir / f"points_{r.index:06d}.npz",
                prev=r.points_prev.astype(np.float32),
                curr=r.points_curr.astype(np.float32),
            )

    poses = np.array(poses)
    np.save(out_dir / "trajectory.npy", poses)

    header = "frame,n_points,n_inliers,elapsed,tracking_lost"
    metrics = np.array(rows, dtype=np.float64)
    if seq.has_ground_truth:
        err = trajectory_error(poses, seq.poses[start + 1 :])
        metrics = np.column_stack([metrics, err])
        header += ",error"
    np.savetxt(out_dir / "metrics.csv", metrics, delimiter=",", header=header, comments="")

    meta = {
        "root": str(root),
        "sequence": sequence,
        "start": start,
        "end": end,
        "frames": len(poses),
        "cached_frames": cache_frames,
        "depth_scale": DEPTH_SCALE,
        "config": asdict(config),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{len(poses)} frames -> {out_dir}")
    print(f"  mean {metrics[:, 3].mean()*1000:.1f} ms/frame, tracking lost on {int(metrics[:, 4].sum())}")
    if seq.has_ground_truth:
        travelled = np.linalg.norm(np.diff(seq.poses[start:end, :3, 3], axis=0), axis=1).sum()
        print(f"  final error {err[-1]:.2f} m over {travelled:.0f} m travelled ({err[-1]/travelled*100:.2f}%)")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run visual odometry over a KITTI sequence.")
    p.add_argument("--root", default="../dataset", help="dataset root holding sequences/ and poses/")
    p.add_argument("--seq", default="00")
    p.add_argument("--out", default=None, help="output directory (default: runs/<seq>)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--detector", default="sift", choices=["sift", "orb"])
    p.add_argument("--matching", default="BF", choices=["BF", "FLANN"])
    p.add_argument("--stereo", default="sgbm", choices=["bm", "sgbm"])
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--max-depth", type=float, default=float("inf"))
    p.add_argument("--no-lidar", action="store_true")
    p.add_argument("--no-cache", action="store_true", help="skip saving per-frame depth and points")
    args = p.parse_args(argv)

    config = VOConfig(
        detector=args.detector,
        matching=args.matching,
        ratio=args.ratio,
        stereo_matcher=args.stereo,
        max_depth=args.max_depth,
        use_lidar=not args.no_lidar,
    )
    out = Path(args.out) if args.out else Path("runs") / args.seq
    run(args.root, args.seq, out, config, args.start, args.end, not args.no_cache)


if __name__ == "__main__":
    main()