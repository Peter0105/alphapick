import yfinance as yf
import time


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
            # yfinance 1.3+ : session 직접 넘기지 않음 (curl_cffi 내부 사용)
            stock = yf.Ticker(normalized)
            info = stock.info

            if not info or not info.get("shortName"):
                raise ValueError("종목 정보를 찾을 수 없습니다")

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
                time.sleep(2)
            continue

    raise RuntimeError(f"데이터 수집 실패 ({retries}회 시도): {last_err}")
