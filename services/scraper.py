import httpx
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

AV_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
AV_BASE = "https://www.alphavantage.co/query"


def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper().replace(" ", "")
    if ticker.isdigit() and len(ticker) == 6:
        return f"{ticker}.KS"
    return ticker


def _get(params: dict, retries: int = 3) -> dict:
    params["apikey"] = AV_KEY
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=20) as client:
                r = client.get(AV_BASE, params=params)
                r.raise_for_status()
                data = r.json()
                if "Note" in data or "Information" in data:
                    raise RuntimeError("API 한도 초과 — 잠시 후 다시 시도하세요")
                return data
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise e
    return {}


def get_stock_data(ticker: str) -> dict:
    normalized = normalize_ticker(ticker)
    av_symbol = normalized.replace(".KS", "").replace(".KQ", "")

    # ── 호출 1: OVERVIEW (펀더멘털 전체) ──────────────────
    overview = _get({"function": "OVERVIEW", "symbol": av_symbol})
    if not overview or not overview.get("Symbol"):
        raise ValueError(f"종목을 찾을 수 없습니다: {av_symbol}")

    time.sleep(15)  # 분당 5회 제한 — 15초 간격

    # ── 호출 2: 일봉 차트 + 현재가 ────────────────────────
    chart_data = []
    current_price = None
    try:
        hist = _get({
            "function": "TIME_SERIES_DAILY",
            "symbol": av_symbol,
            "outputsize": "compact",  # 최근 100일
        })
        series = hist.get("Time Series (Daily)", {})
        sorted_dates = sorted(series.keys())
        for date_str in sorted_dates[-100:]:
            d = series[date_str]
            chart_data.append({
                "date":   date_str,
                "open":   round(float(d["1. open"]), 2),
                "high":   round(float(d["2. high"]), 2),
                "low":    round(float(d["3. low"]), 2),
                "close":  round(float(d["4. close"]), 2),
                "volume": int(d["5. volume"]),
            })
        # 가장 최근 종가 = 현재가
        if sorted_dates:
            current_price = float(series[sorted_dates[-1]]["4. close"])
    except Exception:
        pass

    def flt(val):
        try:
            return float(val) if val and val not in ("None", "-") else None
        except Exception:
            return None

    return {
        "ticker":         normalized,
        "name":           overview.get("Name", av_symbol),
        "current_price":  current_price,
        "currency":       overview.get("Currency", "USD"),
        "market_cap":     flt(overview.get("MarketCapitalization")),
        "pe_ratio":       flt(overview.get("PERatio")),
        "forward_pe":     flt(overview.get("ForwardPE")),
        "eps":            flt(overview.get("EPS")),
        "revenue":        flt(overview.get("RevenueTTM")),
        "profit_margin":  flt(overview.get("ProfitMargin")),
        "week_52_high":   flt(overview.get("52WeekHigh")),
        "week_52_low":    flt(overview.get("52WeekLow")),
        "ma_50":          flt(overview.get("50DayMovingAverage")),
        "ma_200":         flt(overview.get("200DayMovingAverage")),
        "beta":           flt(overview.get("Beta")),
        "dividend_yield": flt(overview.get("DividendYield")),
        "volume":         None,
        "avg_volume":     None,
        "sector":         overview.get("Sector"),
        "industry":       overview.get("Industry"),
        "analyst_target": flt(overview.get("AnalystTargetPrice")),
        "analyst_low":    None,
        "analyst_high":   None,
        "news":           [],  # API 호출 절약 — 뉴스 제거
        "chart_data":     chart_data,
    }
