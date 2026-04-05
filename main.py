"""
Driver Drowsiness Detection — main entry point.

Uses webcam + MediaPipe Face Mesh + OpenCV window.
Run from project folder:
    python main.py

Press 'q' to quit.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from backend.database import init_db, log_event
from backend.fatigue_engine import FatigueEngine
from face_tracker import create_face_landmarker
from utils import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    average_ear,
    head_tilt_metric_deg,
    mouth_aspect_ratio,
)

# ---------------------------------------------------------------------------
# Tunable detection thresholds (adjust if your lighting/face differs)
# ---------------------------------------------------------------------------
# EAR below this = eyes likely closed (typical open EAR ~ 0.25–0.35)
EAR_CLOSED = 0.20
# EAR above this = eyes clearly open again
EAR_OPEN = 0.23
# If EAR stays below EAR_CLOSED for more than this many frames, treat as
# prolonged closure (drowsiness), not a short blink.
MAX_BLINK_FRAMES = 10
# MAR above this for several frames = possible yawn
MAR_YAWN = 0.38
YAWN_CONSEC_FRAMES = 18
YAWN_COOLDOWN_SEC = 4.0

# Smoothing: exponential moving average for EAR (reduces jitter)
EAR_ALPHA = 0.35

# Alert: repeat beep at most every N seconds when High Risk
ALERT_COOLDOWN_SEC = 2.0

# Log to SQLite when fatigue level changes, and at most every N seconds
LOG_INTERVAL_SEC = 30.0


def _play_alert_sound() -> None:
    """Short beep — uses Windows winsound; other OS: terminal bell."""
    if sys.platform == "win32":
        try:
            import winsound

            winsound.Beep(1200, 280)
        except (RuntimeError, ImportError):
            print("\a", end="", flush=True)
    else:
        print("\a", end="", flush=True)


def _draw_hud(
    frame: np.ndarray,
    ear: float,
    mar: float,
    tilt: float,
    level: str,
    score: float,
    blink_rate: float,
    yawns: int,
    eye_sec: float,
    alert: bool,
) -> None:
    """Draw text overlay for the assignment demo / viva."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y0 = 28
    line = 26
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"EAR: {ear:.3f}  MAR: {mar:.3f}  Tilt: {tilt:.1f}deg", (12, y0), font, 0.55, (220, 220, 220), 1)
    cv2.putText(
        frame,
        f"Fatigue: {level}  (score {score:.0f}/100)",
        (12, y0 + line),
        font,
        0.65,
        (80, 220, 80) if level == "Normal" else (0, 200, 255) if level == "Slightly Tired" else (0, 80, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Blinks/min: {blink_rate:.1f}  Eye-closed(60s): {eye_sec:.1f}s  Yawns: {yawns}",
        (12, y0 + 2 * line),
        font,
        0.5,
        (200, 200, 200),
        1,
    )
    if alert:
        cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (12, h - 24), font, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)


def main() -> None:
    init_db()
    engine = FatigueEngine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0). Try closing other apps using the camera.")
        sys.exit(1)

    # MediaPipe Tasks Face Landmarker (Python 3.13 wheels no longer include mp.solutions).
    # See face_tracker.py — downloads models/face_landmarker.task on first run.
    landmarker = create_face_landmarker()

    # Blink FSM: count frames EAR stayed below threshold before reopening
    closed_frames = 0
    ema_ear: Optional[float] = None

    yawn_streak = 0
    last_yawn_time = 0.0

    last_alert_time = 0.0
    last_log_time = 0.0
    prev_level: Optional[str] = None

    fps_t0 = time.time()
    fps_count = 0
    dt_estimate = 1.0 / 30.0
    # VIDEO mode requires monotonically increasing timestamps (milliseconds).
    t_mono_ms = int(time.monotonic() * 1000)

    print("Starting… Press 'q' in the video window to quit.")

    # ---------------- Main loop: capture -> landmarks -> metrics -> fatigue -> UI ----------------
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed — exiting.")
            break

        fps_count += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            dt_estimate = 1.0 / max(fps_count, 1)
            fps_t0 = now
            fps_count = 0

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_mono_ms += 1  # strictly increasing ms per frame (required for VIDEO mode)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = landmarker.detect_for_video(mp_image, t_mono_ms)

        ear_avg = 0.0
        mar = 0.0
        tilt_deg = 0.0

        if result.face_landmarks:
            # One face: list of landmarks with .x / .y / .z (normalized 0–1).
            lm = result.face_landmarks[0]
            _, _, ear_avg = average_ear(lm, w, h)
            mar = mouth_aspect_ratio(lm, w, h)
            tilt_deg = head_tilt_metric_deg(lm, w, h)

            # Smooth EAR (reduces noise)
            if ema_ear is None:
                ema_ear = ear_avg
            else:
                ema_ear = EAR_ALPHA * ear_avg + (1.0 - EAR_ALPHA) * ema_ear

            e = ema_ear

            # --- Blink vs prolonged eye closure ---
            # Short drop in EAR (few frames) = blink; long drop = drowsy closure (engine accumulates time).
            if e < EAR_CLOSED:
                closed_frames += 1
            elif e > EAR_OPEN:
                if 0 < closed_frames <= MAX_BLINK_FRAMES:
                    engine.register_blink()
                closed_frames = 0
            # if between thresholds, keep closed_frames as-is (hysteresis)

            # Time with eyes closed this frame (both eyes — use average EAR)
            if e < EAR_CLOSED:
                engine.register_eyes_closed(dt_estimate)

            engine.register_head_tilt(tilt_deg)

            # --- Yawn: MAR stays high for many frames; cooldown avoids double-counting one yawn ---
            if mar > MAR_YAWN:
                yawn_streak += 1
            else:
                yawn_streak = 0

            if yawn_streak >= YAWN_CONSEC_FRAMES and (now - last_yawn_time) >= YAWN_COOLDOWN_SEC:
                engine.register_yawn()
                last_yawn_time = now
                yawn_streak = 0

            # Optional: draw a few landmarks for the examiner (eyes + mouth)
            def _pt(idx: int) -> tuple[int, int]:
                p = lm[idx]
                return int(p.x * w), int(p.y * h)

            for idx in list(LEFT_EYE_IDX) + list(RIGHT_EYE_IDX):
                cv2.circle(frame, _pt(idx), 1, (0, 255, 200), -1)
            for idx in (13, 14, 78, 308):
                cv2.circle(frame, _pt(idx), 2, (180, 120, 255), -1)

        else:
            # No face — slowly decay EMA so we do not spike false blinks
            ema_ear = None
            closed_frames = 0

        state = engine.compute()
        show_alert = state.level == "High Risk"
        if show_alert and (now - last_alert_time) >= ALERT_COOLDOWN_SEC:
            _play_alert_sound()
            last_alert_time = now

        # SQLite: log when level changes or periodic heartbeat
        should_log = prev_level is None or state.level != prev_level or (now - last_log_time) >= LOG_INTERVAL_SEC
        if should_log:
            log_event(
                level=state.level,
                score=state.score,
                blink_rate=state.blink_rate,
                eye_closed_sec=state.eye_closed_sec,
                yawn_count=state.yawn_count,
                head_tilt_deg=state.head_tilt_deg,
                message="snapshot",
            )
            prev_level = state.level
            last_log_time = now

        _draw_hud(
            frame,
            float(ema_ear or 0.0),
            mar,
            tilt_deg,
            state.level,
            state.score,
            state.blink_rate,
            state.yawn_count,
            state.eye_closed_sec,
            show_alert,
        )

        cv2.imshow("Driver Drowsiness Detection (AI + Fatigue)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Stopped cleanly. Logs saved in fatigue_logs.db (SQLite).")


if __name__ == "__main__":
    main()
