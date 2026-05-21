import time
from datetime import datetime, timedelta

# curl_cffi로 Chrome 흉내 → Yahoo Finance IP 차단 우회
try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="chrome110")
except Exception:
    _SESSION = None

import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper().replace(" ", "")
    # 6자리 숫자면 한국 주식 (KRX)
    if ticker.isdigit() and len(ticker) == 6:
        return f"{ticker}.KS"
    return ticker


def get_stock_data(ticker: str) -> dict:
    normalized = normalize_ticker(ticker)

    # yfinance Ticker 객체 — curl_cffi 세션 주입으로 차단 우회
    if _SESSION:
        t = yf.Ticker(normalized, session=_SESSION)
    else:
        t = yf.Ticker(normalized)

    # ── 1. 기본 정보 (info) ───────────────────────────────
    try:
        info = t.fast_info  # 빠른 버전 (가격 위주)
        full_info = t.info  # 전체 (섹터, PE, EPS 등)
    except Exception as e:
        raise ValueError(f"종목을 찾을 수 없습니다: {normalized} ({e})")

    if not full_info or full_info.get("quoteType") is None:
        raise ValueError(f"종목을 찾을 수 없습니다: {normalized}")

    # 현재가
    current_price = (
        full_info.get("currentPrice")
        or full_info.get("regularMarketPrice")
        or full_info.get("previousClose")
    )

    # 애널리스트 목표가
    analyst_target = full_info.get("targetMeanPrice")
    analyst_low    = full_info.get("targetLowPrice")
    analyst_high   = full_info.get("targetHighPrice")

    # ── 2. 뉴스 ──────────────────────────────────────────
    news_items = []
    try:
        raw_news = t.news or []
        for n in raw_news[:8]:
            content = n.get("content", {})
            title = (
                content.get("title")
                or n.get("title", "")
            )
            # 링크: canonicalUrl > clickThroughUrl > 직접 link
            link = (
                (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url")
                or n.get("link", "#")
            )
            if title:
                news_items.append({"title": title, "link": link})
    except Exception:
        pass

    # ── 3. 차트 데이터 (최근 6개월 일봉) ─────────────────
    chart_data = []
    try:
        end   = datetime.today()
        start = end - timedelta(days=180)
        hist  = t.history(start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          interval="1d")
        for date_idx, row in hist.iterrows():
            chart_data.append({
                "date":   date_idx.strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]),   2),
                "high":   round(float(row["High"]),   2),
                "low":    round(float(row["Low"]),    2),
                "close":  round(float(row["Close"]),  2),
                "volume": int(row["Volume"]),
            })
        # 차트에서 가져온 최신 종가로 현재가 보완
        if not current_price and chart_data:
            current_price = chart_data[-1]["close"]
    except Exception:
        pass

    def flt(val):
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    return {
        "ticker":         normalized,
        "name":           full_info.get("longName") or full_info.get("shortName") or normalized,
        "current_price":  flt(current_price),
        "currency":       full_info.get("currency", "USD"),
        "market_cap":     flt(full_info.get("marketCap")),
        "pe_ratio":       flt(full_info.get("trailingPE")),
        "forward_pe":     flt(full_info.get("forwardPE")),
        "eps":            flt(full_info.get("trailingEps")),
        "revenue":        flt(full_info.get("totalRevenue")),
        "profit_margin":  flt(full_info.get("profitMargins")),
        "week_52_high":   flt(full_info.get("fiftyTwoWeekHigh")),
        "week_52_low":    flt(full_info.get("fiftyTwoWeekLow")),
        "ma_50":          flt(full_info.get("fiftyDayAverage")),
        "ma_200":         flt(full_info.get("twoHundredDayAverage")),
        "beta":           flt(full_info.get("beta")),
        "dividend_yield": flt(full_info.get("dividendYield")),
        "volume":         flt(full_info.get("volume")),
        "avg_volume":     flt(full_info.get("averageVolume")),
        "sector":         full_info.get("sector"),
        "industry":       full_info.get("industry"),
        "analyst_target": flt(analyst_target),
        "analyst_low":    flt(analyst_low),
        "analyst_high":   flt(analyst_high),
        "news":           news_items,
        "chart_data":     chart_data,
    }
