# pyrefly: ignore [missing-import]
from fastapi import FastAPI

app = FastAPI(
    title="IntelliTraffic API",
    description="AI-Based Intelligent Traffic Analytics & Congestion Prediction System API",
    version="0.1.0"
)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to IntelliTraffic API"
    }
