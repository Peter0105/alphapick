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

# curl_cffi Chrome 세션 — Yahoo Finance IP 차단 우회
try:
    from curl_cffi import requests as _curl_requests
    _SESSION = _curl_requests.Session(impersonate="chrome124")
except Exception:
    _SESSION = None


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


def _yf_chart(symbol: str, months: int = 6) -> list[dict]:
    """Yahoo Finance v8 차트 API — curl_cffi로 IP 차단 우회, API 키 불필요"""
    try:
        if _SESSION is None:
            return []
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        r = _SESSION.get(url, params={"interval": "1d", "range": f"{months}mo"}, timeout=15)
        r.raise_for_status()
        result = r.json().get("chart", {}).get("result", [])
        if not result:
            return []
        res       = result[0]
        timestamps = res.get("timestamp", [])
        ohlcv      = res.get("indicators", {}).get("quote", [{}])[0]
        opens  = ohlcv.get("open",  [])
        highs  = ohlcv.get("high",  [])
        lows   = ohlcv.get("low",   [])
        closes = ohlcv.get("close", [])
        vols   = ohlcv.get("volume",[])
        chart = []
        for i, ts in enumerate(timestamps):
            try:
                chart.append({
                    "date":   datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                    "open":   round(float(opens[i]),  2),
                    "high":   round(float(highs[i]),  2),
                    "low":    round(float(lows[i]),   2),
                    "close":  round(float(closes[i]), 2),
                    "volume": int(vols[i] or 0),
                })
            except Exception:
                continue
        return chart
    except Exception:
        return []


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

    # 차트: Yahoo v8(curl_cffi, 무제한) → 실패 시 Alpha Vantage 폴백
    chart_data = _yf_chart(ticker, months=6)
    series = {}  # MA 계산용 (AV 폴백 시 채워짐)
    if not chart_data:
        av_resp = _av({"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": "full"})
        series  = av_resp.get("Time Series (Daily)", {})
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

    # MA50 / MA200 — 200일 데이터 별도 요청 후 계산
    ma_50 = ma_200 = None
    try:
        long_chart = _yf_chart(ticker, months=14)  # ~290 거래일 → 200일 MA 충분
        if long_chart:
            long_closes = [d["close"] for d in long_chart]
            if len(long_closes) >= 50:
                ma_50 = round(sum(long_closes[-50:]) / 50, 2)
            if len(long_closes) >= 200:
                ma_200 = round(sum(long_closes[-200:]) / 200, 2)
        elif series:
            # AV 폴백
            all_series_closes = [float(series[d]["4. close"]) for d in sorted(series.keys())]
            if len(all_series_closes) >= 50:
                ma_50 = round(sum(all_series_closes[-50:]) / 50, 2)
            if len(all_series_closes) >= 200:
                ma_200 = round(sum(all_series_closes[-200:]) / 200, 2)
    except Exception:
        pass
        if not week_52_high:
            week_52_high = max(d["high"] for d in chart_data)
            week_52_low  = min(d["low"]  for d in chart_data)

    # 총 매출 추정: revenue_per_share × 발행주식수(= market_cap / current_price)
    revenue = None
    try:
        rps   = _flt(m.get("revenuePerShareTTM"))
        price = _flt(quote_data.get("c")) or 1
        mcap  = _flt(profile.get("marketCapitalization"))  # 단위: 백만 달러
        if rps and mcap and price:
            shares = (mcap * 1_000_000) / price
            revenue = rps * shares
    except Exception:
        pass

    # Finnhub 이익률/배당수익률은 % 단위 → 프론트가 ×100 하므로 /100 저장
    profit_margin  = (_flt(m.get("netProfitMarginTTM"))         or 0) / 100 or None
    dividend_yield = (_flt(m.get("dividendYieldIndicatedAnnual")) or 0) / 100 or None

    name = profile.get("name") or ticker
    news = _fetch_google_news(f"{ticker} {name}", max_items=10)

    return {
        "ticker":         ticker,
        "name":           name,
        "current_price":  _flt(quote_data.get("c")),
        "currency":       profile.get("currency", "USD"),
        "market_cap":     (_flt(profile.get("marketCapitalization")) or 0) * 1_000_000 or None,
        "pe_ratio":       _flt(m.get("peBasicExclExtraTTM") or m.get("peTTM")),
        "forward_pe":     _flt(m.get("forwardPE") or m.get("peNormalizedAnnual")),
        "eps":            _flt(m.get("epsBasicExclExtraItemsTTM") or m.get("epsTTM")),
        "revenue":        revenue,
        "profit_margin":  profit_margin,
        "week_52_high":   week_52_high,
        "week_52_low":    week_52_low,
        "ma_50":          ma_50,
        "ma_200":         ma_200,
        "beta":           _flt(m.get("beta") or profile.get("beta")),
        "dividend_yield": dividend_yield,
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
