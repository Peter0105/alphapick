from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.scraper import get_stock_data
from services.claude_service import analyze_stock
from datetime import date
import json, os

router = APIRouter()
CACHE_FILE = "data/analysis_cache.json"


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(cache: dict):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)



class AnalysisRequest(BaseModel):
    ticker: str


@router.post("/analyze")
async def analyze(req: AnalysisRequest):
    ticker = req.ticker.strip().upper()
    today = date.today().isoformat()
    cache_key = f"{ticker}_{today}"

    cache = _load_cache()
    if cache_key in cache:
        return {**cache[cache_key], "from_cache": True}

    try:
        stock_data = get_stock_data(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"데이터 수집 실패: {e}")

    try:
        analysis = analyze_stock(stock_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {e}")

    result = {"stock_data": stock_data, "analysis": analysis, "cached_at": today}
    cache[cache_key] = result
    _save_cache(cache)

    return {**result, "from_cache": False}
