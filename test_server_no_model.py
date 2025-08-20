#!/usr/bin/env python3
"""
모델 로딩 없이 서버 테스트
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Test Server (No Model)")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Test server without model loading"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": False,
        "message": "Server running without model"
    }

@app.post("/plan")
async def create_plan(request: dict):
    return {
        "plan_id": "test_plan_001",
        "plan": {
            "files": [
                {"path": "test.py", "reason": "Testing without model"}
            ],
            "notes": "This is a mock response without model"
        },
        "raw_response": "Mock response"
    }

@app.post("/patch")
async def create_patch(request: dict):
    return {
        "patch_id": "test_patch_001", 
        "patch": {
            "version": "1",
            "edits": [
                {
                    "path": "test.py",
                    "loc": {"type": "regex", "pattern": "def test"},
                    "action": "insert_after",
                    "code": "    # Mock patch"
                }
            ]
        },
        "raw_response": "Mock patch response"
    }

if __name__ == "__main__":
    print("🚀 Starting Test Server (No Model)...")
    print("Health: http://127.0.0.1:8765/health")
    print("Plan: POST http://127.0.0.1:8765/plan")
    print("Patch: POST http://127.0.0.1:8765/patch")
    
    uvicorn.run(app, host="127.0.0.1", port=8766)  # 포트 8766 사용
