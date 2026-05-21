import os
from pathlib import Path
from dotenv import load_dotenv

# 절대경로로 .env 로드 (어떤 디렉토리에서 실행해도 동작)
_root = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_root / ".env", override=True)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers import analysis, trading
import uvicorn

app = FastAPI(title="AlphaPick")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api")
app.include_router(trading.router, prefix="/api")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
