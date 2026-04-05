"""
Geometry helpers for drowsiness detection:
- Eye Aspect Ratio (EAR): how "open" the eye is (drops when eyes close).
- Mouth Aspect Ratio (MAR): mouth opening height vs width (rises when yawning).
- Head tilt: simple roll + pitch proxy from landmark positions (no extra calibration).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# --- MediaPipe Face Mesh landmark indices (468 points) -----------------------
# Six points per eye — standard layout for the EAR formula (see research papers).
RIGHT_EYE_IDX: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
LEFT_EYE_IDX: tuple[int, ...] = (362, 385, 387, 263, 373, 380)

# Inner / outer mouth points for a simple MAR (vertical / horizontal).
# 13 upper lip inner, 14 lower lip inner, 78 left corner, 308 right corner
MOUTH_IDX: tuple[int, ...] = (13, 14, 78, 308)

# Nose tip and eyes for head orientation proxy
NOSE_TIP_IDX = 1
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(p1 - p2))


def eye_aspect_ratio(
    landmarks: Sequence[object],
    eye_indices: tuple[int, ...],
    frame_w: int,
    frame_h: int,
) -> float:
    """
    Eye Aspect Ratio (EAR) for one eye.

    Uses six landmarks around the eye. When the eye closes, vertical distances
    shrink faster than horizontal — EAR drops. Typical open eye EAR is ~ 0.25–0.35
    (depends on person and camera); closed is near 0.

    Formula (common in drowsiness papers):
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    where p1…p6 follow the fixed order in eye_indices.
    """
    if len(eye_indices) != 6:
        raise ValueError("EAR needs exactly 6 landmarks per eye.")

    # Build pixel coordinates from normalized MediaPipe landmarks (0–1).
    pts = []
    for i in eye_indices:
        lm = landmarks[i]
        pts.append(np.array([lm.x * frame_w, lm.y * frame_h], dtype=np.float64))
    p1, p2, p3, p4, p5, p6 = pts

    vertical_1 = _dist(p2, p6)
    vertical_2 = _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(
    landmarks: Sequence[object],
    frame_w: int,
    frame_h: int,
) -> float:
    """
    Mouth Aspect Ratio (MAR): height of opening divided by mouth width.

    Yawning stretches the mouth vertically — MAR increases. Normal talking
    may also raise MAR briefly; we use consecutive frames + cooldown in main.
    """
    top = np.array(
        [landmarks[MOUTH_IDX[0]].x * frame_w, landmarks[MOUTH_IDX[0]].y * frame_h],
        dtype=np.float64,
    )
    bottom = np.array(
        [landmarks[MOUTH_IDX[1]].x * frame_w, landmarks[MOUTH_IDX[1]].y * frame_h],
        dtype=np.float64,
    )
    left_c = np.array(
        [landmarks[MOUTH_IDX[2]].x * frame_w, landmarks[MOUTH_IDX[2]].y * frame_h],
        dtype=np.float64,
    )
    right_c = np.array(
        [landmarks[MOUTH_IDX[3]].x * frame_w, landmarks[MOUTH_IDX[3]].y * frame_h],
        dtype=np.float64,
    )
    vertical = _dist(top, bottom)
    horizontal = _dist(left_c, right_c)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def head_tilt_metric_deg(
    landmarks: Sequence[object],
    frame_w: int,
    frame_h: int,
) -> float:
    """
    Single number combining roll-like tilt and a simple pitch proxy.

    - Roll: angle of the line between the two outer eye corners vs horizontal.
      If the driver tilts the head sideways, this angle moves away from ~0°.
    - Pitch proxy: vertical offset of nose tip relative to eye midpoint (scaled).

    This is not full 6-DoF head pose (no solvePnP), but it is easy to explain
    and works as a behavioral fatigue cue for an assignment.
    """
    le = np.array(
        [landmarks[LEFT_EYE_OUTER_IDX].x * frame_w, landmarks[LEFT_EYE_OUTER_IDX].y * frame_h],
        dtype=np.float64,
    )
    re = np.array(
        [landmarks[RIGHT_EYE_OUTER_IDX].x * frame_w, landmarks[RIGHT_EYE_OUTER_IDX].y * frame_h],
        dtype=np.float64,
    )
    nose = np.array(
        [landmarks[NOSE_TIP_IDX].x * frame_w, landmarks[NOSE_TIP_IDX].y * frame_h],
        dtype=np.float64,
    )

    dx = float(re[0] - le[0])
    dy = float(re[1] - le[1])
    roll_deg = abs(math.degrees(math.atan2(dy, dx)))

    eye_mid = (le + re) / 2.0
    # How far the nose sits below/above the eye line (normalized by face width)
    face_w = max(_dist(le, re), 1.0)
    pitch_proxy = abs(float(nose[1] - eye_mid[1])) / face_w
    pitch_deg = math.degrees(math.atan(pitch_proxy))

    # Combined "abnormal head pose" score in degrees (tuned for display / fatigue weight)
    return roll_deg + 0.4 * pitch_deg


def average_ear(
    landmarks: Sequence[object],
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float]:
    """
    Returns (left_ear, right_ear, mean_ear) for both eyes.
    """
    le = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, frame_w, frame_h)
    re = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
    return le, re, (le + re) / 2.0
