from fastapi import APIRouter
from services.portfolio import load_portfolio, reset_portfolio, execute_trade, update_prices
from services.scraper import get_stock_data
from services.claude_service import make_trading_decision

router = APIRouter()

WATCHLIST = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "005930.KS", "000660.KS"]


@router.get("/portfolio")
async def get_portfolio():
    return load_portfolio()


@router.post("/portfolio/reset")
async def do_reset():
    return reset_portfolio()


@router.post("/trade/auto")
async def auto_trade():
    portfolio = load_portfolio()

    watchlist_data = []
    price_updates = {}

    for ticker in WATCHLIST:
        try:
            d = get_stock_data(ticker)
            if d["current_price"]:
                price_updates[ticker] = d["current_price"]
                watchlist_data.append({
                    "ticker": ticker,
                    "name": d["name"],
                    "price": d["current_price"],
                    "pe_ratio": d["pe_ratio"],
                    "week_52_high": d["week_52_high"],
                    "week_52_low": d["week_52_low"],
                    "ma_50": d["ma_50"],
                    "analyst_target": d["analyst_target"],
                })
        except Exception:
            continue

    portfolio = update_prices(price_updates)
    decision = make_trading_decision(portfolio, watchlist_data)

    executed = []
    for a in decision.get("actions", []):
        if a["action"] in ("BUY", "SELL"):
            result = execute_trade(
                action=a["action"],
                ticker=a["ticker"],
                quantity=int(a.get("quantity", 0)),
                price=float(a.get("price", 0)),
                reason=a.get("reason", ""),
            )
            executed.append({**a, "result": result})

    return {
        "decision": decision,
        "executed": executed,
        "portfolio": load_portfolio(),
    }


@router.get("/trade/history")
async def trade_history():
    p = load_portfolio()
    return list(reversed(p.get("trade_history", [])))
