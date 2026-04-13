# 🚗 Driver Drowsiness Detection System (AI + Fatigue Analysis)

This project is a real-time driver drowsiness detection system built using Python, OpenCV, and MediaPipe Face Landmarker. It uses a webcam to monitor facial features and detect signs of fatigue.

---

## 🔹 Features

- Eye closure detection using EAR (Eye Aspect Ratio)
- Yawn detection using MAR (Mouth Aspect Ratio)
- Head tilt detection (roll + pitch approximation)
- Blink detection using frame-based logic
- Fatigue score calculation using multiple parameters
- Driver state classification:
  - Normal  
  - Slightly Tired  
  - High Risk  
- Real-time display (HUD)
- Alert system (beep + warning message)
- Stores fatigue logs using SQLite database

---

## 🔹 Technologies Used

- Python  
- OpenCV  
- MediaPipe Face Landmarker (Tasks API)  
- NumPy  
- SQLite  

---

## 🔹 Project Structure

    ├── main.py
    ├── face_tracker.py
    ├── utils.py
    ├── backend/
    │   ├── fatigue_engine.py
    │   └── database.py
    ├── models/
    ├── requirements.txt
    └── fatigue_logs.db

---

## 🔹 Working

1. Webcam captures real-time frames.
2. MediaPipe Face Landmarker detects 468 facial landmarks.
3. Important facial points (eyes, mouth, nose) are extracted.
4. System computes:
   - Eye Aspect Ratio (EAR)
   - Mouth Aspect Ratio (MAR)
   - Head tilt  
5. Behavioral tracking:
   - Blink detection using EAR thresholds  
   - Eye closure duration tracking  
   - Yawn detection using consecutive frames  
6. These events are passed to a fatigue engine.
7. Fatigue score is calculated over a sliding window (~60 seconds).
8. Based on the score, driver state is classified.
9. If fatigue is high:
   - Visual warning is displayed  
   - Beep alert is triggered  
10. Events are stored in SQLite database (`fatigue_logs.db`).

---

## 🔹 Formulas Used

### 1. Eye Aspect Ratio (EAR)

EAR = (||p2 − p6|| + ||p3 − p5||) / (2 × ||p1 − p4||)

- Uses 6 eye landmarks  
- EAR decreases when eyes are closed  

---

### 2. Mouth Aspect Ratio (MAR)

MAR = ||top − bottom|| / ||left − right||

- Uses 4 mouth landmarks  
- MAR increases when mouth opens (yawning)  

---

### 3. Head Tilt (Combined Metric)

- Roll (side tilt):

  θ_roll = tan⁻¹((y₂ − y₁) / (x₂ − x₁))

- Pitch proxy:

  θ_pitch ≈ tan⁻¹(nose_offset / face_width)

Final Tilt:

Tilt = roll + 0.4 × pitch

- Represents deviation from normal head position using eye alignment and nose position  

---

## 🔹 Fatigue Scoring

Fatigue is calculated using a weighted model over a sliding window (~60 seconds):

Score = W₁·BlinkComponent + W₂·EyeClosure + W₃·Yawn + W₄·Tilt

Where:

- BlinkComponent → based on deviation of blink rate from normal range  
- EyeClosure → total time eyes remained closed  
- Yawn → number of yawns detected  
- Tilt → average head tilt  

Weights used:

- W_BLINK = 1.2  
- W_EYE = 2.5  
- W_YAWN = 8.0  
- W_TILT = 0.35  

Final score is limited between 0–100.

---

## 🔹 Classification

- Score < 35 → Normal  
- 35 ≤ Score < 65 → Slightly Tired  
- Score ≥ 65 → High Risk  

---

## 🔹 Database (SQLite)

- Database file: `fatigue_logs.db`
- Automatically created when the program runs
- Stores:
  - Timestamp  
  - Fatigue level  
  - Score  
  - Blink rate  
  - Eye closure time  
  - Yawn count  
  - Head tilt  

Used for:
- Tracking driver behavior  
- Debugging and analysis  

---

## 🔹 How to Run

1. Install Python (3.10–3.12 recommended)

2. Open terminal in project folder

3. (Optional) Create virtual environment:

    python -m venv venv  
    venv\Scripts\activate  

4. Install dependencies:

    pip install -r requirements.txt  

5. Run the project:

    python main.py  

6. Press **q** to exit

---

## 🔹 Output

- Webcam window shows:
  - EAR, MAR, head tilt  
  - Fatigue score and level  
  - Blink rate, yawns, eye closure time  
- Facial landmarks drawn on eyes and mouth  
- If fatigue is high:
  - Red alert on screen  
  - Beep sound  

---

## 🔹 Conclusion

This project demonstrates how computer vision and simple mathematical models can be used to detect driver fatigue in real time. It combines facial landmark detection with behavioral analysis and provides a lightweight and efficient solution.
