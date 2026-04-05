"""
MediaPipe **Tasks** Face Landmarker (current PyPI wheels for Python 3.13 on Windows).

Newer `mediapipe` packages no longer ship the old `mp.solutions.face_mesh` API.
We use `FaceLandmarker` in VIDEO mode instead; landmark indices 0–467 match the
classic face mesh layout used in `utils.py` (eyes/mouth points).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Official Google-hosted model (~3.7 MB). Downloaded once into models/.
MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def get_model_path(project_root: Path | None = None) -> Path:
    """
    Return path to face_landmarker.task, downloading it the first time if needed.
    Requires internet on first run only.
    """
    root = project_root or Path(__file__).resolve().parent
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / MODEL_FILENAME
    if not out.is_file() or out.stat().st_size < 100_000:
        print("Downloading Face Landmarker model (one-time, ~3.7 MB)...")
        print(f"URL: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, out)
        print(f"Saved to: {out}")
    return out


def create_face_landmarker():
    """
    Build a FaceLandmarker for webcam VIDEO mode.
    Caller must call .close() on the returned object when done.
    """
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions

    model_path = str(get_model_path())
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)
