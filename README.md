# DataLake ML Backend

## Overview

DataLake ML Backend is a FastAPI-based AI verification service responsible for validating attendance through multiple computer vision pipelines.

The backend provides secure liveness verification and facial authentication services that integrate directly with the DataLake 3.0 mobile application.

---

## AI Verification Pipeline

### Stage 1

Head Movement Verification

Ensures the user performs required head movements and is physically present.

### Stage 2

Blink Detection

Confirms liveness through eye blink analysis.

### Stage 3

Face Recognition

Validates identity using facial embeddings.

---

## Technology Stack

| Layer            | Technology      |
| ---------------- | --------------- |
| API Framework    | FastAPI         |
| Web Server       | Uvicorn         |
| Computer Vision  | OpenCV          |
| Face Mesh        | MediaPipe       |
| Face Recognition | InsightFace     |
| Deep Learning    | TensorFlow 2.15 |
| Model Runtime    | ONNX Runtime    |
| Neural Networks  | PyTorch         |
| Data Processing  | NumPy           |

---

## Project Structure

```text
app/
├── api/
│   ├── blink_detection.py
│   ├── head_movement.py
│   └── face_recognition.py
│
├── services/
│   ├── blink_service.py
│   ├── head_movement_service.py
│   ├── face_service.py
│   ├── frame_decoder.py
│   └── mediapipe_service.py
│
├── schemas/
│   ├── requests.py
│   └── responses.py
│
└── main.py
```

---

## API Endpoints

### Blink Detection

```http
POST /blink-detection
```

### Head Movement Verification

```http
POST /head-movement
```

### Face Recognition

```http
POST /face-recognition
```

---

## Installation

### Python Version

```text
Python 3.10.18
```

### Create Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Models

Place model files:

```text
models/
├── blink_eye.keras
└── head_movement.pth
```

---

## Run Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## Security Features

* Multi-step Verification
* Liveness Detection
* Anti-Spoofing Checks
* Facial Authentication
* Modular ML Pipeline
