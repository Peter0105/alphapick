"""
AlphaPick Backtesting Engine v2
지표: 총수익률 / 연환산수익률 / 샤프 / 소르티노 / 칼마 / MDD / 승률 / 프로핏팩터
슬리피지 0.1%, 수수료 0.05% 편도
포지션 사이징: 가용 현금 20% (최대 5종목)
손절: -8% 하드 스톱
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


INITIAL_CASH  = 10_000_000
COMMISSION    = 0.0005
SLIPPAGE      = 0.001
MAX_POSITIONS = 5
POSITION_SIZE = 0.20
STOP_LOSS_PCT = -0.08
RISK_FREE     = 0.03 / 252   # 일별 무위험이자율


@dataclass
class Trade:
    date: str
    ticker: str
    action: str
    qty: int
    entry_price: float
    exit_price: float = 0.0
    pnl: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    strategy_name: str
    tickers: list
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_trade_days: float
    total_trades: int
    final_value: float
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)

    def row(self) -> dict:
        return {
            "전략":         self.strategy_name,
            "총수익률(%)":  f"{self.total_return_pct:+.1f}",
            "연환산(%)":    f"{self.annualized_return_pct:+.1f}",
            "샤프":         f"{self.sharpe_ratio:.2f}",
            "소르티노":     f"{self.sortino_ratio:.2f}",
            "칼마":         f"{self.calmar_ratio:.2f}",
            "MDD(%)":       f"{self.max_drawdown_pct:.1f}",
            "승률(%)":      f"{self.win_rate_pct:.1f}",
            "P.F":          f"{self.profit_factor:.2f}",
            "거래수":        self.total_trades,
            "최종자산(원)":  f"{self.final_value:,.0f}",
        }


def _exec_price(price: float, side: str) -> float:
    return price * (1 + SLIPPAGE) if side == "BUY" else price * (1 - SLIPPAGE)


def run_backtest(
    price_data: dict,
    strategy_fn: Callable,
    strategy_name: str,
) -> BacktestResult:

    # ── 신호 생성 ──────────────────────────────────────
    signals = {}
    for ticker, df in price_data.items():
        try:
            signals[ticker] = strategy_fn(df.copy())
        except Exception:
            continue

    if not signals:
        raise ValueError("신호 생성 실패")

    all_dates = sorted(set.union(*[set(df.index) for df in signals.values()]))

    cash       = float(INITIAL_CASH)
    positions  = {}   # {ticker: {"qty":int, "avg":float, "entry_date":date}}
    open_trades= {}   # {ticker: Trade (미체결)}
    closed     = []   # 완료된 Trade 목록
    equity     = []

    for date in all_dates:
        # ── 손절 체크 ──────────────────────────────────
        for ticker in list(positions.keys()):
            if ticker not in signals or date not in signals[ticker].index:
                continue
            price = signals[ticker].loc[date, "close"]
            pos   = positions[ticker]
            ret   = (price - pos["avg"]) / pos["avg"]
            if ret <= STOP_LOSS_PCT:
                ep       = _exec_price(price, "SELL")
                proceeds = pos["qty"] * ep * (1 - COMMISSION)
                cash    += proceeds
                if ticker in open_trades:
                    t = open_trades.pop(ticker)
                    t.exit_price = ep
                    t.pnl        = (ep - t.entry_price) * t.qty
                    closed.append(t)
                del positions[ticker]

        # ── 신호 처리 ──────────────────────────────────
        for ticker, df in signals.items():
            if date not in df.index:
                continue
            sig   = df.loc[date, "signal"]
            price = df.loc[date, "close"]
            if pd.isna(sig) or pd.isna(price):
                continue

            if sig == 1 and ticker not in positions:
                if len(positions) >= MAX_POSITIONS:
                    continue
                avail = cash * POSITION_SIZE
                ep    = _exec_price(price, "BUY")
                qty   = int(avail / (ep * (1 + COMMISSION)))
                if qty < 1:
                    continue
                cost  = qty * ep * (1 + COMMISSION)
                if cost > cash:
                    continue
                cash -= cost
                positions[ticker] = {"qty": qty, "avg": ep, "entry_date": date}
                open_trades[ticker] = Trade(
                    date=str(date.date()), ticker=ticker, action="BUY",
                    qty=qty, entry_price=ep, reason="매수 신호"
                )

            elif sig == -1 and ticker in positions:
                pos  = positions.pop(ticker)
                ep   = _exec_price(price, "SELL")
                cash += pos["qty"] * ep * (1 - COMMISSION)
                if ticker in open_trades:
                    t = open_trades.pop(ticker)
                    t.exit_price = ep
                    t.pnl        = (ep - t.entry_price) * t.qty
                    t.reason     = "매도 신호"
                    closed.append(t)

        # ── 자산 평가 ──────────────────────────────────
        pos_val = 0.0
        for t2, pos in positions.items():
            if t2 in signals and date in signals[t2].index:
                pos_val += pos["qty"] * signals[t2].loc[date, "close"]
        equity.append({"date": str(date.date()), "value": cash + pos_val})

    # ── 미체결 포지션 청산 ─────────────────────────────
    last_date = all_dates[-1]
    for ticker, pos in list(positions.items()):
        if ticker in signals and last_date in signals[ticker].index:
            price = signals[ticker].loc[last_date, "close"]
            ep    = _exec_price(price, "SELL")
            cash += pos["qty"] * ep * (1 - COMMISSION)
            if ticker in open_trades:
                t = open_trades.pop(ticker)
                t.exit_price = ep
                t.pnl        = (ep - t.entry_price) * t.qty
                t.reason     = "기간 종료 청산"
                closed.append(t)

    final_value = cash

    # ── 성과 지표 계산 ─────────────────────────────────
    curve = pd.Series([e["value"] for e in equity], dtype=float)
    total_ret = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
    n_years   = len(all_dates) / 252
    ann_ret   = ((final_value / INITIAL_CASH) ** (1 / max(n_years, 0.01)) - 1) * 100

    daily_ret = curve.pct_change().dropna()
    excess    = daily_ret - RISK_FREE
    sharpe    = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    downside  = excess.clip(upper=0)
    sortino   = (excess.mean() / downside.std() * np.sqrt(252)) if downside.std() > 0 else 0.0

    roll_max  = curve.cummax()
    dd        = (curve - roll_max) / roll_max * 100
    max_dd    = float(dd.min())
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # 승률 / 프로핏팩터
    wins  = [t for t in closed if t.pnl > 0]
    loses = [t for t in closed if t.pnl <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0.0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in loses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    # 평균 보유 기간 (추정: 거래 수 / 2 → 라운드트립)
    avg_days = len(all_dates) / max(len(closed), 1)

    return BacktestResult(
        strategy_name          = strategy_name,
        tickers                = list(price_data.keys()),
        total_return_pct       = round(total_ret, 2),
        annualized_return_pct  = round(ann_ret, 2),
        sharpe_ratio           = round(float(sharpe), 3),
        sortino_ratio          = round(float(sortino), 3),
        calmar_ratio           = round(calmar, 3),
        max_drawdown_pct       = round(max_dd, 2),
        win_rate_pct           = round(win_rate, 1),
        profit_factor          = round(pf, 2),
        avg_trade_days         = round(avg_days, 1),
        total_trades           = len(closed),
        final_value            = round(final_value, 0),
        equity_curve           = equity,
        trades                 = closed,
    )
