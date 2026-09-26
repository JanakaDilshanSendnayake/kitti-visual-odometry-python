"""Stereo disparity and depth."""

from __future__ import annotations

import cv2
import numpy as np

# Disparity values OpenCV uses to mark a pixel it could not match. They are
# replaced with a small disparity so the division below stays finite; the
# resulting depths are far beyond max_depth and get dropped in estimate_motion.
INVALID_DISPARITY = (0.0, -1.0)
DISPARITY_FLOOR = 0.1


class StereoDepth:
    """Disparity and depth for a rectified stereo pair.

    The OpenCV matcher is built once and reused for every frame.
    """

    def __init__(
        self,
        focal_length: float,
        baseline: float,
        matcher: str = "sgbm",
        sad_window: int = 6,
        block_size: int = 11,
    ) -> None:
        self.focal_length = focal_length
        self.baseline = baseline
        self.matcher_name = matcher
        self.sad_window = sad_window
        self.block_size = block_size
        self.num_disparities = sad_window * 16

        if matcher == "bm":
            self._matcher = cv2.StereoBM_create(
                numDisparities=self.num_disparities, blockSize=block_size
            )
        elif matcher == "sgbm":
            # P1 penalises a one-step disparity change between neighbours, P2 a
            # larger jump. P1 is scaled by sad_window so gradual surfaces such as
            # the road can slope smoothly instead of breaking into flat terraces.
            self._matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disparities,
                blockSize=block_size,
                P1=8 * 1 * sad_window**2,
                P2=32 * 1 * block_size**2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
        else:
            raise ValueError(f"unknown matcher {matcher!r}, expected 'bm' or 'sgbm'")

    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Disparity in pixels, float32. Inputs are not modified."""
        return self._matcher.compute(left, right).astype(np.float32) / 16  # fixed point

    def depth(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Depth in metres from Z = f * b / d, for every pixel.

        Unmatched pixels come back at roughly f * b / 0.1 metres, far outside any
        real scene. estimate_motion drops them with its max_depth cut.
        """
        disp = self.disparity(left, right)
        for value in INVALID_DISPARITY:
            disp[disp == value] = DISPARITY_FLOOR
        return self.focal_length * self.baseline / disp


def feature_mask(height: int, width: int, border: int) -> np.ndarray:
    """Mask off the left edge, where the right camera has nothing to match against."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, border:] = 255
    return mask