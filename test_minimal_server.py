#!/usr/bin/env python3
"""
최소한의 테스트 서버 - 모델 없이 실행
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Minimal Test Server")

@app.get("/")
async def root():
    return {"message": "Hello from minimal server"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Server is running"}

if __name__ == "__main__":
    print("🚀 Starting Minimal Test Server...")
    print("URL: http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765)
