import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

FINNHUB_KEY  = os.getenv("FINNHUB_KEY", "")
AV_KEY       = os.getenv("ALPHA_VANTAGE_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"
AV_BASE      = "https://www.alphavantage.co/query"


# ──────────────────────────────────────────────────────────
#  헬퍼
# ──────────────────────────────────────────────────────────

def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper().replace(" ", "")
    if ticker.isdigit() and len(ticker) == 6:
        return f"{ticker}.KS"
    return ticker


def _flt(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def _fh(path: str, **params) -> dict | list:
    params["token"] = FINNHUB_KEY
    try:
        r = httpx.get(f"{FINNHUB_BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _av(params: dict) -> dict:
    params["apikey"] = AV_KEY
    try:
        r = httpx.get(AV_BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "Note" in data or "Information" in data:
            return {}
        return data
    except Exception:
        return {}


def _fetch_google_news(query: str, max_items: int = 10) -> list[dict]:
    try:
        url = (
            f"https://news.google.com/rss/search"
            f"?q={quote(query)}+stock&hl=en&gl=US&ceid=US:en"
        )
        r = httpx.get(
            url, timeout=10, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title") or ""
            link  = item.findtext("link") or "#"
            if title:
                items.append({"title": title, "link": link})
        return items
    except Exception:
        return []


# ──────────────────────────────────────────────────────────
#  경로 A: 미국 주식 (Finnhub)
# ──────────────────────────────────────────────────────────

def _get_us_stock(ticker: str) -> dict:
    quote_data = _fh("/quote", symbol=ticker)
    if not quote_data or quote_data.get("c", 0) == 0:
        raise ValueError(f"종목을 찾을 수 없습니다: {ticker}")

    profile = _fh("/stock/profile2", symbol=ticker)
    metrics_raw = _fh("/stock/metric", symbol=ticker, metric="all")
    m = metrics_raw.get("metric", {})
    rec = _fh("/stock/price-target", symbol=ticker)

    # 차트: Alpha Vantage TIME_SERIES_DAILY
    chart_data = []
    av_resp = _av({"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": "compact"})
    series = av_resp.get("Time Series (Daily)", {})
    for date_str in sorted(series.keys())[-120:]:
        d = series[date_str]
        chart_data.append({
            "date":   date_str,
            "open":   round(float(d["1. open"]),  2),
            "high":   round(float(d["2. high"]),  2),
            "low":    round(float(d["3. low"]),   2),
            "close":  round(float(d["4. close"]), 2),
            "volume": int(d["5. volume"]),
        })

    week_52_high = _flt(m.get("52WeekHigh"))
    week_52_low  = _flt(m.get("52WeekLow"))
    if chart_data and not week_52_high:
        week_52_high = max(d["high"] for d in chart_data)
        week_52_low  = min(d["low"]  for d in chart_data)

    name = profile.get("name") or ticker
    news = _fetch_google_news(f"{ticker} {name}", max_items=10)

    return {
        "ticker":         ticker,
        "name":           name,
        "current_price":  _flt(quote_data.get("c")),
        "currency":       profile.get("currency", "USD"),
        "market_cap":     _flt(profile.get("marketCapitalization")),
        "pe_ratio":       _flt(m.get("peBasicExclExtraTTM") or m.get("peTTM")),
        "forward_pe":     _flt(m.get("peNormalizedAnnual")),
        "eps":            _flt(m.get("epsBasicExclExtraItemsTTM") or m.get("epsTTM")),
        "revenue":        _flt(m.get("revenuePerShareTTM")),
        "profit_margin":  _flt(m.get("netProfitMarginTTM")),
        "week_52_high":   week_52_high,
        "week_52_low":    week_52_low,
        "ma_50":          _flt(m.get("50DayMA")),
        "ma_200":         _flt(m.get("200DayMA")),
        "beta":           _flt(m.get("beta") or profile.get("beta")),
        "dividend_yield": _flt(m.get("dividendYieldIndicatedAnnual")),
        "volume":         _flt(quote_data.get("v")),
        "avg_volume":     _flt(m.get("avgVolume")),
        "sector":         profile.get("finnhubIndustry"),
        "industry":       profile.get("finnhubIndustry"),
        "analyst_target": _flt(rec.get("targetMean")),
        "analyst_low":    _flt(rec.get("targetLow")),
        "analyst_high":   _flt(rec.get("targetHigh")),
        "news":           news,
        "chart_data":     chart_data,
    }


# ──────────────────────────────────────────────────────────
#  경로 B: 한국 주식 (Alpha Vantage 전용)
# ──────────────────────────────────────────────────────────

def _get_kr_stock(ticker: str) -> dict:
    """ticker = '005930.KS' 형식"""
    av_symbol = ticker.replace(".KS", "").replace(".KQ", "")

    overview = _av({"function": "OVERVIEW", "symbol": av_symbol})
    if not overview or not overview.get("Symbol"):
        raise ValueError(f"종목을 찾을 수 없습니다: {ticker}")

    chart_data = []
    av_resp = _av({"function": "TIME_SERIES_DAILY", "symbol": av_symbol, "outputsize": "compact"})
    series = av_resp.get("Time Series (Daily)", {})
    current_price = None
    for date_str in sorted(series.keys())[-120:]:
        d = series[date_str]
        chart_data.append({
            "date":   date_str,
            "open":   round(float(d["1. open"]),  2),
            "high":   round(float(d["2. high"]),  2),
            "low":    round(float(d["3. low"]),   2),
            "close":  round(float(d["4. close"]), 2),
            "volume": int(d["5. volume"]),
        })
    if chart_data:
        current_price = chart_data[-1]["close"]

    name = overview.get("Name", av_symbol)
    news = _fetch_google_news(f"{av_symbol} {name}", max_items=10)

    return {
        "ticker":         ticker,
        "name":           name,
        "current_price":  _flt(current_price),
        "currency":       overview.get("Currency", "KRW"),
        "market_cap":     _flt(overview.get("MarketCapitalization")),
        "pe_ratio":       _flt(overview.get("PERatio")),
        "forward_pe":     _flt(overview.get("ForwardPE")),
        "eps":            _flt(overview.get("EPS")),
        "revenue":        _flt(overview.get("RevenueTTM")),
        "profit_margin":  _flt(overview.get("ProfitMargin")),
        "week_52_high":   _flt(overview.get("52WeekHigh")),
        "week_52_low":    _flt(overview.get("52WeekLow")),
        "ma_50":          _flt(overview.get("50DayMovingAverage")),
        "ma_200":         _flt(overview.get("200DayMovingAverage")),
        "beta":           _flt(overview.get("Beta")),
        "dividend_yield": _flt(overview.get("DividendYield")),
        "volume":         None,
        "avg_volume":     None,
        "sector":         overview.get("Sector"),
        "industry":       overview.get("Industry"),
        "analyst_target": _flt(overview.get("AnalystTargetPrice")),
        "analyst_low":    None,
        "analyst_high":   None,
        "news":           news,
        "chart_data":     chart_data,
    }


# ──────────────────────────────────────────────────────────
#  공개 인터페이스
# ──────────────────────────────────────────────────────────

def get_stock_data(ticker: str) -> dict:
    normalized = normalize_ticker(ticker)
    if normalized.endswith(".KS") or normalized.endswith(".KQ"):
        return _get_kr_stock(normalized)
    return _get_us_stock(normalized)
