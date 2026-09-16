# KITTI Visual Odometry

Stereo + LiDAR visual odometry on the KITTI odometry benchmark, with a real-time
dashboard showing the camera, LiDAR scan, depth map, feature tracking and the
estimated trajectory.

## Setup

```bash
uv sync                    # add --extra notebook for Jupyter
```

The dataset is not included. Point `--root` at a directory holding `sequences/`
and `poses/` (default `../dataset`).

## Usage

Process a sequence and save the result:

```bash
python -m kitti_vo.runner --seq 00 --end 400
```

Replay it in the dashboard:

```bash
python -m kitti_vo.dashboard --run runs/00
```

Or run the pipeline live, without a precomputed run:

```bash
python -m kitti_vo.dashboard --live --seq 00 --detector orb --stereo bm --no-lidar
```

Replay supports play/pause, seeking and speed control. Live mode streams frames in
order as they are computed, so those controls are disabled.

## Pipeline

| Module | Does |
|---|---|
| `calibration.py` | Reads `calib.txt`; focal length and baseline |
| `dataset.py` | Indexed access to images, scans and poses |
| `stereo.py` | Disparity (BM/SGBM) and metric depth |
| `lidar.py` | Projects scans to a sparse depth image; fuses with stereo |
| `features.py` | SIFT/ORB detection and ratio-tested matching |
| `motion.py` | Back-projection and PnP + RANSAC |
| `pipeline.py` | Chains the above into a trajectory |
| `runner.py` | CLI; writes `runs/<seq>/` |

Depth uses `0` as the invalid sentinel throughout.

## Measured on sequence 00

Full sequence, 4541 frames over 3724 m, with default settings:

| | final error | drift |
|---|---|---|
| sift + sgbm + lidar | 26.6 m | 0.71% |

Stereo depth against LiDAR:

| matcher | coverage | median error | within 1 m |
|---|---|---|---|
| `bm` | 44% | 0.24 m | 77% |
| `sgbm` | 86% | 0.20 m | 80% |

Whole pipeline, per frame:

| config | ms/frame | inliers |
|---|---|---|
| sift + sgbm + lidar | 264 | 98% |
| sift + sgbm | 194 | 98% |
| orb + bm | 78 | 98% |

### Why these defaults

`ratio` and `max_depth` both matter far more to drift than to per-frame error, so
they have to be tuned on a long run. Over 1500 frames:

| | drift |
|---|---|
| ratio 0.7, max_depth 50 m | 1.79% |
| ratio 0.7, max_depth open | 1.48% |
| ratio 0.3, max_depth 50 m | 1.32% |
| **ratio 0.3, max_depth open** | **0.99%** |

Capping depth throws away the distant points that constrain rotation, which drifts
the heading. A loose ratio lets weak matches through: RANSAC drops gross outliers
but not their small shared bias, which compounds over thousands of frames.

Short runs are not representative. The first ~12 frames of sequence 00 carry a
scale transient, so a 100-frame run reports several times the drift of a full one.
