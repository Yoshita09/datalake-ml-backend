<<<<<<< HEAD
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

# then all other imports below
=======
>>>>>>> parent of ff6206f (configured model service)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.head_movement import router as head_router
from app.api.blink_detection import router as blink_router
from app.api.face_recognition import router as face_router

app = FastAPI(title="AI Attendance Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(head_router)
app.include_router(blink_router)
app.include_router(face_router)


@app.get("/")
async def root():
    return {"message": "AI Attendance Backend Running"}