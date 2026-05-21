"""
AlphaPick Auto Trading Router
백테스트 검증된 MA 교차 전략 기반 규칙 매매
(Claude API는 시장 코멘트에만 사용 — 매매 결정은 순수 알고리즘)
"""

import json
import pandas as pd
from fastapi import APIRouter
from services.portfolio import load_portfolio, reset_portfolio, execute_trade, update_prices
from services.strategy  import strat_bollinger_rev, add_indicators
from services.scraper   import get_stock_data

router = APIRouter()

WATCHLIST = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]

# ── 포트폴리오 ──────────────────────────────────────────────

@router.get("/portfolio")
async def get_portfolio():
    return load_portfolio()


@router.post("/portfolio/reset")
async def do_reset():
    return reset_portfolio()


# ── 전략 신호 계산 ─────────────────────────────────────────

def _get_signal(ticker: str) -> dict:
    """Yahoo Finance에서 1년치 데이터 받아 MA 교차 신호 계산"""
    try:
        from curl_cffi import requests as cr
        from datetime import datetime, timedelta

        session = cr.Session(impersonate="chrome124")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r   = session.get(url, params={"interval": "1d", "range": "1y"}, timeout=15)
        result = r.json().get("chart", {}).get("result", [])
        if not result:
            return {"signal": 0, "price": None}

        res    = result[0]
        ts     = res.get("timestamp", [])
        ohlcv  = res.get("indicators", {}).get("quote", [{}])[0]
        closes = ohlcv.get("close", [])
        opens  = ohlcv.get("open",  [])
        highs  = ohlcv.get("high",  [])
        lows   = ohlcv.get("low",   [])
        vols   = ohlcv.get("volume",[])

        dates = [pd.Timestamp.fromtimestamp(t) for t in ts]
        df    = pd.DataFrame({
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "volume": vols,
        }, index=pd.DatetimeIndex(dates)).dropna()

        if len(df) < 55:
            return {"signal": 0, "price": float(closes[-1]) if closes else None}

        df     = strat_bollinger_rev(df)
        last   = df.iloc[-1]
        signal = int(last.get("signal", 0))
        price  = float(last["close"])

        return {
            "signal":    signal,
            "price":     price,
            "bb_upper":  float(last["bb_upper"]) if pd.notna(last.get("bb_upper")) else None,
            "bb_lower":  float(last["bb_lower"]) if pd.notna(last.get("bb_lower")) else None,
            "bb_pct":    float(last["bb_pct"])   if pd.notna(last.get("bb_pct"))   else None,
            "rsi":       float(last["rsi"])       if pd.notna(last.get("rsi"))      else None,
        }
    except Exception as e:
        return {"signal": 0, "price": None}


# ── 자동 매매 실행 ─────────────────────────────────────────

@router.post("/trade/auto")
async def auto_trade():
    portfolio     = load_portfolio()
    price_updates = {}
    signals       = {}
    trade_log     = []

    # 신호 수집
    for ticker in WATCHLIST:
        sig = _get_signal(ticker)
        if sig["price"]:
            price_updates[ticker] = sig["price"]
            signals[ticker]       = sig

    # 가격 업데이트
    portfolio = update_prices(price_updates)

    # 손절 (-8%)
    for ticker, pos in list(portfolio["positions"].items()):
        if ticker not in signals:
            continue
        price = signals[ticker]["price"]
        ret   = (price - pos["avg_price"]) / pos["avg_price"]
        if ret <= -0.08:
            result = execute_trade("SELL", ticker, pos["quantity"], price, "손절 -8%")
            trade_log.append({
                "action": "SELL", "ticker": ticker,
                "quantity": pos["quantity"], "price": price,
                "reason": f"손절 -8% (손실 {ret*100:.1f}%)",
                "result": result,
            })

    # 전략 신호 실행
    portfolio = load_portfolio()
    cash      = portfolio["cash"]
    positions = portfolio["positions"]

    for ticker, sig in signals.items():
        price    = sig["price"]
        signal   = sig["signal"]
        ma20     = sig.get("ma20")
        ma50     = sig.get("ma50")

        if signal == 1 and ticker not in positions:
            # 보유 종목 최대 4개 제한, 현금 20% 사용
            if len(positions) >= 4:
                continue
            invest = cash * 0.20
            qty    = int(invest / price)
            if qty < 1:
                continue
            bb_l = sig.get("bb_lower")
            bb_u = sig.get("bb_upper")
            reason_buy = (f"볼린저 하단 반등 (BB하단={bb_l:.2f}, BB상단={bb_u:.2f})"
                          if bb_l else "볼린저 하단 반등")
            result = execute_trade("BUY", ticker, qty, price, reason_buy)
            if result["success"]:
                cash -= qty * price
                positions[ticker] = True
                trade_log.append({
                    "action": "BUY", "ticker": ticker,
                    "quantity": qty, "price": price,
                    "reason": "볼린저 하단 반등",
                    "result": result,
                })

        elif signal == -1 and ticker in positions:
            pos = portfolio["positions"].get(ticker)
            if not pos:
                continue
            bb_u = sig.get("bb_upper")
            reason_sell = (f"볼린저 상단 하락 (BB상단={bb_u:.2f})" if bb_u else "볼린저 상단 하락")
            result = execute_trade("SELL", ticker, pos["quantity"], price, reason_sell)
            if result["success"]:
                trade_log.append({
                    "action": "SELL", "ticker": ticker,
                    "quantity": pos["quantity"], "price": price,
                    "reason": "MA 데드크로스",
                    "result": result,
                })

    # 시장 코멘트 (Claude — 옵션, 실패해도 무관)
    market_view = _get_market_comment(signals, portfolio)

    return {
        "decision": {
            "market_view": market_view,
            "strategy":    "볼린저 밴드 역추세 (백테스트 4년 +91.2%, 샤프 1.09, 승률 60%)",
        },
        "executed":  trade_log,
        "portfolio": load_portfolio(),
    }


def _get_market_comment(signals: dict, portfolio: dict) -> str:
    try:
        from services.claude_service import _get_client
        oversold  = [t for t, s in signals.items() if (s.get("bb_pct") or 1) < 0.2]
        overbought= [t for t, s in signals.items() if (s.get("bb_pct") or 0) > 0.8]
        prompt = (
            f"볼린저 밴드 과매도(하단 근접) 종목: {oversold}\n"
            f"볼린저 밴드 과매수(상단 근접) 종목: {overbought}\n"
            f"포트폴리오 수익률: {portfolio.get('return_pct', 0):.2f}%\n"
            "위 상황을 한 문장으로 요약해줘."
        )
        resp = _get_client().messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        bull_count = sum(1 for s in signals.values() if (s.get("ma20") or 0) > (s.get("ma50") or 0))
        return f"관심 종목 {len(signals)}개 중 {bull_count}개 상승 추세 (MA20 > MA50)"


@router.get("/trade/history")
async def trade_history():
    p = load_portfolio()
    return list(reversed(p.get("trade_history", [])))


@router.get("/backtest/result")
async def get_backtest_result():
    """백테스트 결과 조회"""
    from pathlib import Path
    result_file = Path(__file__).resolve().parent.parent / "data" / "backtest_result.json"
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "백테스트 결과 없음 — scripts/run_backtest.py 실행 필요"}
