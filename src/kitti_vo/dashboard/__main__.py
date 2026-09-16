"""Entry point: python -m kitti_vo.dashboard"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6 import QtWidgets

from ..dataset import KittiSequence
from ..pipeline import VOConfig
from .sources import LiveSource, ReplaySource
from .window import Dashboard


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Visual odometry dashboard.")
    p.add_argument("--run", default=None, help="replay a finished run directory")
    p.add_argument("--live", action="store_true", help="run the pipeline live instead")
    p.add_argument("--root", default="../dataset")
    p.add_argument("--seq", default="00")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--detector", default="sift", choices=["sift", "orb"])
    p.add_argument("--stereo", default="sgbm", choices=["bm", "sgbm"])
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--no-lidar", action="store_true")
    args = p.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv)

    if args.live:
        seq = KittiSequence(args.root, args.seq)
        config = VOConfig(
            detector=args.detector,
            stereo_matcher=args.stereo,
            ratio=args.ratio,
            use_lidar=not args.no_lidar,
        )
        source = LiveSource(seq, config, args.start, args.end)
        title = f"KITTI VO — live, sequence {args.seq}"
    else:
        run_dir = Path(args.run) if args.run else Path("runs") / args.seq
        if not (run_dir / "config.json").exists():
            p.error(f"no run found at {run_dir}. Run kitti_vo.runner first, or pass --live.")
        source = ReplaySource(run_dir)
        title = f"KITTI VO — replay {run_dir}"

    window = Dashboard(source, title)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()