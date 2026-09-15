"""Stereo disparity and depth.

Invalid depth is 0.0 everywhere, the same convention KITTI uses in its own depth
maps. Nothing sits at zero distance, so the sentinel is never ambiguous.
"""

from __future__ import annotations

import cv2
import numpy as np

# Below this a match is either flagged bad by OpenCV, or so far away that
# sub-pixel error swings the depth by hundreds of metres. This also sets the
# furthest depth we ever report: f * b / MIN_DISPARITY.
MIN_DISPARITY = 1.0


class StereoDepth:
    """Disparity and depth for a rectified stereo pair.

    The OpenCV matcher is built once and reused for every frame.
    """

    def __init__(
        self,
        focal_length: float,
        baseline: float,
        matcher: str = "sgbm",
        num_disparities: int = 96,
        block_size: int = 11,
    ) -> None:
        if num_disparities % 16:
            raise ValueError(f"num_disparities must be a multiple of 16, got {num_disparities}")

        self.focal_length = focal_length
        self.baseline = baseline
        self.matcher_name = matcher
        self.num_disparities = num_disparities
        self.block_size = block_size

        if matcher == "bm":
            self._matcher = cv2.StereoBM_create(
                numDisparities=num_disparities, blockSize=block_size
            )
        elif matcher == "sgbm":
            # P1/P2 penalise disparity jumps between neighbouring pixels. Both
            # scale with blockSize^2 for single-channel input, and P2 must exceed P1.
            self._matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * block_size**2,
                P2=32 * block_size**2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
        else:
            raise ValueError(f"unknown matcher {matcher!r}, expected 'bm' or 'sgbm'")

    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Disparity in pixels, float32, 0 where unmatched. Inputs are not modified."""
        disp = self._matcher.compute(left, right).astype(np.float32) / 16.0  # fixed point
        return np.where(disp >= MIN_DISPARITY, disp, 0.0)

    def depth(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Depth in metres, float32, 0 where not measurable."""
        disp = self.disparity(left, right)
        depth = np.zeros_like(disp)
        valid = disp >= MIN_DISPARITY
        depth[valid] = (self.focal_length * self.baseline) / disp[valid]
        return depth


def feature_mask(height: int, width: int, border: int) -> np.ndarray:
    """Mask off the left edge, where the right camera has nothing to match against.

    Keypoints there never get a depth, so skip detecting them at all.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, border:] = 255
    return mask