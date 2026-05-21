import json
import os
from datetime import datetime
from pathlib import Path

# 어떤 환경에서 실행해도 프로젝트 루트 기준으로 절대경로 사용
_ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO_FILE = str(_ROOT / "data" / "portfolio.json")
INITIAL_CASH = 10_000_000


def load_portfolio() -> dict:
    if not os.path.exists(PORTFOLIO_FILE):
        return _create_initial()
    with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_portfolio(portfolio: dict):
    os.makedirs(str(_ROOT / "data"), exist_ok=True)
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def _create_initial() -> dict:
    p = {
        "cash": INITIAL_CASH,
        "initial_cash": INITIAL_CASH,
        "positions": {},
        "trade_history": [],
        "created_at": datetime.now().isoformat(),
        "total_value": INITIAL_CASH,
        "return_pct": 0.0,
    }
    save_portfolio(p)
    return p


def reset_portfolio() -> dict:
    p = _create_initial()
    save_portfolio(p)
    return p


def execute_trade(action: str, ticker: str, quantity: int, price: float, reason: str) -> dict:
    p = load_portfolio()

    if action == "BUY":
        cost = quantity * price
        if cost > p["cash"]:
            return {"success": False, "message": "현금 부족"}
        p["cash"] -= cost
        if ticker in p["positions"]:
            pos = p["positions"][ticker]
            total_qty = pos["quantity"] + quantity
            avg = (pos["avg_price"] * pos["quantity"] + price * quantity) / total_qty
            p["positions"][ticker] = {"quantity": total_qty, "avg_price": round(avg, 2), "current_price": price}
        else:
            p["positions"][ticker] = {"quantity": quantity, "avg_price": price, "current_price": price}

    elif action == "SELL":
        if ticker not in p["positions"]:
            return {"success": False, "message": "보유 종목 없음"}
        pos = p["positions"][ticker]
        sell_qty = min(quantity, pos["quantity"])
        p["cash"] += sell_qty * price
        if sell_qty >= pos["quantity"]:
            del p["positions"][ticker]
        else:
            p["positions"][ticker]["quantity"] -= sell_qty

    p["trade_history"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": action,
        "ticker": ticker,
        "quantity": quantity,
        "price": price,
        "reason": reason,
    })

    save_portfolio(p)
    return {"success": True}


def update_prices(price_updates: dict) -> dict:
    p = load_portfolio()
    total_pos = 0
    for ticker, pos in p["positions"].items():
        if ticker in price_updates:
            p["positions"][ticker]["current_price"] = price_updates[ticker]
        total_pos += p["positions"][ticker]["quantity"] * p["positions"][ticker]["current_price"]
    p["total_value"] = p["cash"] + total_pos
    p["return_pct"] = (p["total_value"] - p["initial_cash"]) / p["initial_cash"] * 100
    save_portfolio(p)
    return p
