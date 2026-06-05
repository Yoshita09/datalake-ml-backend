# DataLake ML Backend

## Overview

DataLake ML Backend is a FastAPI-based AI verification service responsible for liveness detection and facial verification for attendance marking.

The backend exposes REST APIs that process image frames captured by the mobile application and return verification results.

---

## Features

### Face Recognition

Uses:

* InsightFace
* ONNX Runtime
* PyTorch

### Blink Detection

Uses:

* MediaPipe Face Mesh
* Eye Aspect Ratio (EAR)

### Head Movement Detection

Uses:

* MediaPipe Face Mesh
* Session-Based State Machine

### FastAPI REST APIs

* /blink-detection
* /head-movement
* /face-recognition

---

## Tech Stack

| Component        | Version |
| ---------------- | ------- |
| Python           | 3.10.18 |
| FastAPI          | Latest  |
| Uvicorn          | Latest  |
| TensorFlow       | 2.15.0  |
| TensorFlow Metal | Latest  |
| MediaPipe        | 0.10.14 |
| PyTorch          | Latest  |
| InsightFace      | Latest  |
| ONNX Runtime     | Latest  |
| OpenCV           | Latest  |
| NumPy            | < 2.0   |

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

Request

```json
{
  "frame": "base64_encoded_frame"
}
```

Response

```json
{
  "success": true,
  "blink_count": 1
}
```

---

### Head Movement Detection

```http
POST /head-movement
```

Request

```json
{
  "frame": "base64_encoded_frame",
  "session_id": "abc123"
}
```

Response

```json
{
  "success": true,
  "stage": "verified"
}
```

---

### Face Recognition

```http
POST /face-recognition
```

Request

```json
{
  "frame": "base64_encoded_frame"
}
```

Response

```json
{
  "success": true,
  "confidence": 0.97
}
```

---

## Run Locally

### Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Server

```bash
uvicorn main:app --reload
```

Server:

```text
http://localhost:8000
```

---

## Production Deployment

Recommended:

* Railway
* Render
* AWS EC2
* Azure App Service

---

## Security Recommendations

* Restrict CORS Origins
* Enable Authentication
* HTTPS Only
* Rate Limiting
* Request Validation

---

## License

MIT License
