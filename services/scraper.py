import yfinance as yf
import requests
import time


def _make_session() -> requests.Session:
    """클라우드 서버 IP 차단 우회용 세션"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return s


def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper().replace(" ", "")
    if ticker.isdigit() and len(ticker) == 6:
        return f"{ticker}.KS"
    return ticker


def get_stock_data(ticker: str, retries: int = 3) -> dict:
    normalized = normalize_ticker(ticker)

    last_err = None
    for attempt in range(retries):
        try:
            session = _make_session()
            stock = yf.Ticker(normalized, session=session)
            info = stock.info

            # info가 비어있으면 실패로 간주
            if not info or info.get("trailingPegRatio") is None and not info.get("shortName"):
                raise ValueError("빈 응답 — 티커를 찾을 수 없거나 레이트 리밋")

            hist = stock.history(period="6mo")

            try:
                news = [
                    {"title": n.get("title", ""), "link": n.get("link", "")}
                    for n in (stock.news or [])[:5]
                ]
            except Exception:
                news = []

            chart_data = []
            if not hist.empty:
                for dt, row in hist.iterrows():
                    chart_data.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "open": round(float(row["Open"]), 2),
                        "high": round(float(row["High"]), 2),
                        "low": round(float(row["Low"]), 2),
                        "close": round(float(row["Close"]), 2),
                        "volume": int(row["Volume"]),
                    })

            current_price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or (float(hist["Close"].iloc[-1]) if not hist.empty else None)
            )

            return {
                "ticker": normalized,
                "name": info.get("longName") or info.get("shortName", ticker),
                "current_price": current_price,
                "currency": info.get("currency", "USD"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "revenue": info.get("totalRevenue"),
                "profit_margin": info.get("profitMargins"),
                "week_52_high": info.get("fiftyTwoWeekHigh"),
                "week_52_low": info.get("fiftyTwoWeekLow"),
                "ma_50": info.get("fiftyDayAverage"),
                "ma_200": info.get("twoHundredDayAverage"),
                "beta": info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "analyst_target": info.get("targetMeanPrice"),
                "analyst_low": info.get("targetLowPrice"),
                "analyst_high": info.get("targetHighPrice"),
                "news": news,
                "chart_data": chart_data,
            }

        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 + attempt * 2)  # 2s, 4s 대기 후 재시도
            continue

    raise RuntimeError(f"데이터 수집 실패 ({retries}회 시도): {last_err}")
