# Driver Drowsiness Detection System (AI + Fatigue Analysis)

Beginner-friendly academic project: **real-time webcam** processing with **Python**, **OpenCV**, and **MediaPipe Face Mesh**. It estimates **Eye Aspect Ratio (EAR)** for eye closure, **Mouth Aspect Ratio (MAR)** for yawning, a simple **head tilt** metric, then combines **time-based behavior** into a **fatigue score** and three levels: **Normal**, **Slightly Tired**, **High Risk**. Alerts use a **beep** (Windows) plus **on-screen warning**. Events are stored in **SQLite** (`fatigue_logs.db`).

---

## 1. What you should see when it works

1. A window titled **Driver Drowsiness Detection (AI + Fatigue)** showing your webcam feed.
2. Text overlay: **EAR**, **MAR**, **Tilt**, **Fatigue level**, **score**, **blinks/min**, **eye-closed time (60s window)**, **yawns**.
3. Small dots on your **eyes** and **mouth** landmarks (so examiners can see the model is tracking your face).
4. If conditions simulate strong fatigue (long eyes closed + yawns + high score), the label may show **High Risk**, red border, text **DROWSINESS ALERT**, and a **beep** (every few seconds while at High Risk).
5. A file **`fatigue_logs.db`** appears in the project folder after you run the program (SQLite database).

---

## 2. Project folder layout

```
Driver Drowsiness Detectin System System using AI/
├── main.py              # Start here — webcam loop, MediaPipe Tasks, alerts, HUD
├── face_tracker.py      # Face Landmarker (Tasks API) + one-time model download
├── models/              # face_landmarker.task downloaded automatically (~3.7 MB)
├── utils.py             # EAR, MAR, head tilt math + landmark index constants
├── requirements.txt     # Python libraries to install
├── README.md            # This guide
├── fatigue_logs.db      # Created automatically (SQLite) when you run main.py
└── backend/
    ├── __init__.py
    ├── database.py      # SQLite: create table, insert logs, read recent rows
    └── fatigue_engine.py  # Sliding-window fatigue score + Normal / Tired / High Risk
```

---

## 3. Install Python (if you have never installed it)

These steps are for **Windows** (your system). For macOS/Linux, use the same Python version idea but download the installer for that OS.

1. Open your browser and go to: **https://www.python.org/downloads/**
2. Download **Python 3.10**, **3.11**, or **3.12** (recommended for **MediaPipe** stability). Avoid very new versions if you hit install errors; 3.11 is a safe choice.
3. Run the installer.
4. **Important:** on the first screen, enable **“Add Python to PATH”**, then click **Install Now**.
5. Close the installer when it finishes.

**Check that Python works:**

1. Press **Win + R**, type `cmd`, press Enter (or use **PowerShell**).
2. Type:

```text
python --version
```

You should see something like `Python 3.11.x`.  
If `python` is not found, try:

```text
py --version
```

On many Windows PCs, `py` is the Python launcher.

---

## 4. Open the project folder in the terminal

You must run commands **inside** the folder that contains `main.py`.

**Option A — File Explorer**

1. Open the folder:  
   `Driver Drowisness Detectin System System using AI`
2. Click the address bar, type `cmd`, press Enter. A terminal opens **already in that folder**.

**Option B — `cd` command**

If your path has spaces, use quotes:

```text
cd "C:\Users\YOUR_NAME\Desktop\MY SPACE\Driver Drowisness Detectin System System using AI"
```

Replace `YOUR_NAME` with your Windows username.

---

## 5. (Recommended) Create a virtual environment

A **virtual environment** keeps libraries for this project separate from other Python projects.

```text
python -m venv venv
```

Activate it:

**Command Prompt:**

```text
venv\Scripts\activate.bat
```

**PowerShell:**

```text
venv\Scripts\Activate.ps1
```

If PowerShell says scripts are disabled, either use **cmd** instead, or run once (as Administrator is sometimes needed):

```text
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

After activation, your prompt often starts with `(venv)`.

---

## 6. Install required libraries (pip)

With the terminal **in the project folder** (and venv activated if you use one):

```text
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Exact one-liner** (same thing):

```text
pip install -r requirements.txt
```

This installs **OpenCV**, **MediaPipe**, and **NumPy** (listed in `requirements.txt`).

---

## 7. How to start the project

Still in the project folder:

```text
python main.py
```

**Alternative (Windows launcher):**

```text
py main.py
```

- Allow **camera access** if Windows asks.
- The OpenCV window should open. **To quit:** click the **video window** so it is focused (selected), then press the **`q`** key on your keyboard.
- **Do not type `q` in PowerShell or Command Prompt** — that is a different program. If you see `q: The term 'q' is not recognized`, you typed `q` in the terminal by mistake. Click the OpenCV window and press **`q`** there, or close the window with the mouse (**X**).

**Optional:** double-click `run.bat` in this folder (it runs `python main.py` and pauses on errors).

---

## 8. How to test if it is working

| Test | What to do | What you should see |
|------|------------|---------------------|
| Webcam | Run `python main.py` | Live video in the window |
| Face tracking | Sit in front of the camera | EAR/MAR numbers change; dots on eyes/mouth |
| Blink | Blink normally | Blinks/min increases over time |
| Eyes closed | Close eyes a few seconds | EAR drops; eye-closed time increases; score may rise |
| Yawn | Open mouth wide for ~1 second | MAR rises; after enough frames, yawn may register |
| Alert | Simulate fatigue (eyes closed + yawns) over a minute | Level can become **High Risk**; red border + beep |
| Database | After running, check project folder | `fatigue_logs.db` exists |

---

## 9. View SQLite logs (optional)

The database file is **`fatigue_logs.db`** in the project folder.

**If you have `sqlite3` on PATH** (some Python installs include it):

```text
sqlite3 fatigue_logs.db "SELECT * FROM fatigue_events ORDER BY id DESC LIMIT 10;"
```

**Or use a free GUI:** DB Browser for SQLite — open `fatigue_logs.db`, table `fatigue_events`.

Columns include: `timestamp`, `level`, `score`, `blink_rate`, `eye_closed_sec`, `yawn_count`, `head_tilt_deg`, `message`.

---

## 10. Common errors and fixes

### “python is not recognized”

- Reinstall Python with **Add Python to PATH**, or use `py main.py`.
- Or use the full path to `python.exe` from your Python installation folder.

### “No module named cv2” / “mediapipe” / “numpy”

You are not in the right environment or libraries did not install.

```text
pip install -r requirements.txt
```

If you use a venv, **activate it first** before `pip install`.

### Webcam does not open / “Could not open webcam”

- Close **Zoom**, **Teams**, **Camera** app, or other programs using the camera.
- Unplug/replug USB webcam if you use one.
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `main.py` if you have multiple cameras.

### `AttributeError: module 'mediapipe' has no attribute 'solutions'`

On **Python 3.13 (Windows)**, PyPI only provides newer MediaPipe wheels — the old `mp.solutions.face_mesh` API is **not included**. This project uses **MediaPipe Tasks** (`FaceLandmarker`) in `face_tracker.py` instead. Run:

```text
pip install -r requirements.txt
```

The first run downloads `models/face_landmarker.task` (needs internet once).

### MediaPipe fails to install or import

- Use **Python 3.10–3.12** from **python.org** (64-bit), or **3.13** with `mediapipe>=0.10.33`.
- Upgrade pip: `python -m pip install --upgrade pip`
- Install again: `pip install mediapipe`

### NumPy “MINGW-W64” warning or Python exits right after `python main.py` (Windows)

That message means NumPy was **not** the normal official build. OpenCV loads NumPy and the process can **crash** or return to the prompt with no window.

**Fix (use official wheels — run in the project folder):**

```text
pip uninstall numpy -y
pip cache purge
pip install "numpy>=2.0,<2.2" --upgrade --force-reinstall --only-binary :all:
pip install -r requirements.txt
python main.py
```

You should **not** see the MINGW warning after this. Then test:

```text
python -c "import numpy; import cv2; print('OK', numpy.__version__)"
```

If it still fails, install **Python 3.11 x64** from [python.org](https://www.python.org/downloads/), create a **venv**, and run `pip install -r requirements.txt` inside that venv.

### Other NumPy / OpenCV issues

If you see different errors when importing `cv2`:

```text
pip uninstall numpy opencv-python -y
pip install numpy opencv-python
```

If problems continue, use **Python 3.11 x64** from python.org, create a **new venv**, and reinstall `requirements.txt`.

### Detection is too sensitive / not sensitive enough

In **`main.py`**, adjust:

- `EAR_CLOSED`, `EAR_OPEN` — eye closure sensitivity  
- `MAR_YAWN`, `YAWN_CONSEC_FRAMES` — yawning  
In **`backend/fatigue_engine.py`**, adjust:

- `W_*` weights and `THRESH_SLIGHT`, `THRESH_HIGH` — fatigue score and labels  

---

## 11. What to say in a viva (short)

- **EAR**: ratio of eye height to width from six eye landmarks; drops when eyes close.  
- **MAR**: mouth opening height vs width; increases when yawning.  
- **Head tilt**: simple geometric mix of eye-line angle and nose position (not full 3D pose).  
- **Fatigue score**: weighted mix over **~60 seconds** of blink pattern, **eyes-closed duration**, **yawns**, and **head tilt**.  
- **Alerts + logs**: sound + overlay for **High Risk**; **SQLite** stores timestamps and metrics for analysis.

---

## 12. Ethics and real driving

This is an **educational prototype**. It is **not** certified for real vehicles. Never rely on it for safety while driving.

---

## License / academic use

Use freely for learning and assignments; cite your sources and list libraries (OpenCV, MediaPipe) in your report.
