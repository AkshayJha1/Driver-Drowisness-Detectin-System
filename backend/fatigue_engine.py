"""
Fatigue scoring engine: combines blink rate, eye closure time, yawns, and head tilt
over a sliding time window, then maps to Normal / Slightly Tired / High Risk.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import time
from typing import Deque, Tuple


@dataclass
class FatigueState:
    """What we expose to the UI each frame."""

    score: float  # 0–100 (higher = more fatigued / risky)
    level: str  # "Normal" | "Slightly Tired" | "High Risk"
    blink_rate: float  # estimated blinks per minute
    eye_closed_sec: float  # seconds eyes closed in window
    yawn_count: int  # yawns in window
    head_tilt_deg: float  # average tilt in window


# Tunable weights (simple linear model — easy to explain in viva)
W_BLINK = 1.2  # unusual blink rate adds fatigue
W_EYE = 2.5  # long eyes-closed time is a strong signal
W_YAWN = 8.0  # each yawn adds a lot
W_TILT = 0.35  # head tilt / abnormal pose

# Sliding window length in seconds (behavior is summarized over this period)
WINDOW_SEC = 60.0

# Classification thresholds on final score (0–100)
THRESH_SLIGHT = 35.0
THRESH_HIGH = 65.0


class FatigueEngine:
    """
    Maintains time-stamped events and computes a fatigue score over WINDOW_SEC.
    """

    def __init__(self) -> None:
        self._window_sec = WINDOW_SEC
        # Timestamp when each blink occurred
        self._blink_times: Deque[float] = deque()
        # Timestamp when each yawn occurred
        self._yawn_times: Deque[float] = deque()
        # (timestamp, duration in seconds) for eyes-closed segments
        self._eye_chunks: Deque[Tuple[float, float]] = deque()
        # (timestamp, tilt in degrees)
        self._tilt_samples: Deque[Tuple[float, float]] = deque()

    def _trim(self, now: float) -> None:
        """Remove entries older than the sliding window."""
        cutoff = now - self._window_sec
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        while self._yawn_times and self._yawn_times[0] < cutoff:
            self._yawn_times.popleft()
        while self._eye_chunks and self._eye_chunks[0][0] < cutoff:
            self._eye_chunks.popleft()
        while self._tilt_samples and self._tilt_samples[0][0] < cutoff:
            self._tilt_samples.popleft()

    def register_blink(self) -> None:
        """Call once when a blink is detected."""
        self._blink_times.append(time())

    def register_yawn(self) -> None:
        """Call once when a yawn is detected."""
        self._yawn_times.append(time())

    def register_eyes_closed(self, delta_sec: float) -> None:
        """Add time spent with eyes closed this frame."""
        if delta_sec <= 0:
            return
        self._eye_chunks.append((time(), delta_sec))

    def register_head_tilt(self, tilt_deg: float) -> None:
        """Absolute tilt from upright (degrees); 0 = straight."""
        if tilt_deg < 0:
            tilt_deg = 0.0
        self._tilt_samples.append((time(), tilt_deg))

    def compute(self) -> FatigueState:
        """Compute fatigue score and label from the current sliding window."""
        now = time()
        self._trim(now)

        n_blinks = len(self._blink_times)
        blink_rate = (n_blinks / self._window_sec) * 60.0

        # Normal resting blink rate is often quoted ~ 15–20/min; penalize too low or too high
        expected_min, expected_max = 8.0, 28.0
        if expected_min <= blink_rate <= expected_max:
            blink_component = 0.0
        elif blink_rate < expected_min:
            blink_component = (expected_min - blink_rate) * 2.0
        else:
            blink_component = (blink_rate - expected_max) * 1.5

        yawn_count = len(self._yawn_times)
        eye_closed_sec = sum(dt for _, dt in self._eye_chunks)

        if self._tilt_samples:
            head_tilt_deg = sum(v for _, v in self._tilt_samples) / len(self._tilt_samples)
        else:
            head_tilt_deg = 0.0

        raw = (
            W_BLINK * blink_component
            + W_EYE * min(eye_closed_sec, 15.0)
            + W_YAWN * yawn_count
            + W_TILT * head_tilt_deg
        )
        score = float(max(0.0, min(100.0, raw)))

        if score < THRESH_SLIGHT:
            level = "Normal"
        elif score < THRESH_HIGH:
            level = "Slightly Tired"
        else:
            level = "High Risk"

        return FatigueState(
            score=score,
            level=level,
            blink_rate=blink_rate,
            eye_closed_sec=eye_closed_sec,
            yawn_count=yawn_count,
            head_tilt_deg=head_tilt_deg,
        )
