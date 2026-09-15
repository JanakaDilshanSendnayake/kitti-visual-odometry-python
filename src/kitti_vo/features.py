"""Feature detection and matching between consecutive frames."""

from __future__ import annotations

import cv2
import numpy as np


class FeatureTracker:
    """Detects keypoints and matches them between two frames.

    Ratio test keeps a match only if the best candidate is clearly better than the
    second best, which throws out ambiguous matches in repetitive texture.
    """

    def __init__(
        self,
        detector: str = "sift",
        matching: str = "BF",
        ratio: float = 0.7,
    ) -> None:
        if detector == "sift":
            self._detector = cv2.SIFT_create()
            norm = cv2.NORM_L2
        elif detector == "orb":
            self._detector = cv2.ORB_create(nfeatures=3000)
            norm = cv2.NORM_HAMMING  # ORB descriptors are binary
        else:
            raise ValueError(f"unknown detector {detector!r}, expected 'sift' or 'orb'")

        self.detector_name = detector
        self.matching = matching
        self.ratio = ratio

        if matching == "BF":
            self._matcher = cv2.BFMatcher_create(norm, crossCheck=False)
        elif matching == "FLANN":
            if detector == "sift":
                index_params = dict(algorithm=1, trees=5)  # KD-tree, float descriptors
            else:
                index_params = dict(algorithm=6, table_number=6, key_size=12,
                                    multi_probe_level=1)  # LSH, binary descriptors
            self._matcher = cv2.FlannBasedMatcher(index_params, dict(checks=50))
        else:
            raise ValueError(f"unknown matching {matching!r}, expected 'BF' or 'FLANN'")

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None):
        """Return (keypoints, descriptors) for one image."""
        return self._detector.detectAndCompute(image, mask)

    def match(self, des1: np.ndarray | None, des2: np.ndarray | None) -> list[cv2.DMatch]:
        """Match descriptors and apply the ratio test. Empty list if there is nothing to match."""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []

        if self.matching == "FLANN" and self.detector_name == "sift":
            des1, des2 = np.float32(des1), np.float32(des2)

        pairs = self._matcher.knnMatch(des1, des2, k=2)
        # knnMatch can return fewer than k candidates for a descriptor.
        return [p[0] for p in pairs if len(p) == 2 and p[0].distance <= self.ratio * p[1].distance]


def matched_points(matches, kp1, kp2) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates of the matched keypoints, as two (N, 2) float32 arrays."""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return pts1, pts2
