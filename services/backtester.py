"""
AlphaPick Backtesting Engine
- 초기 자본: 10,000,000원 (또는 USD 기준으로 단일 통화 시뮬레이션)
- 슬리피지: 0.1%, 수수료: 0.05% (편도)
- 포지션 사이징: 가용 현금의 20% (최대 5종목 동시 보유)
- 손절: -8% 하드 스톱
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


INITIAL_CASH  = 10_000_000   # 원
COMMISSION    = 0.0005       # 0.05% 편도
SLIPPAGE      = 0.001        # 0.1%
MAX_POSITIONS = 5
POSITION_SIZE = 0.20         # 가용 현금의 20%
STOP_LOSS     = -0.08        # -8% 손절


@dataclass
class Trade:
    date: str
    ticker: str
    action: str       # BUY / SELL
    qty: int
    price: float
    value: float
    reason: str = ""


@dataclass
class BacktestResult:
    strategy_name: str
    tickers: list[str]
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    final_value: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.strategy_name}]\n"
            f"  총수익률:     {self.total_return_pct:+.1f}%\n"
            f"  연환산수익률: {self.annualized_return_pct:+.1f}%\n"
            f"  샤프지수:     {self.sharpe_ratio:.2f}\n"
            f"  최대낙폭(MDD):{self.max_drawdown_pct:.1f}%\n"
            f"  승률:         {self.win_rate_pct:.1f}%\n"
            f"  거래 횟수:    {self.total_trades}회\n"
            f"  최종 자산:    {self.final_value:,.0f}원\n"
        )


def _exec_price(price: float, side: str) -> float:
    """슬리피지 적용 체결가"""
    return price * (1 + SLIPPAGE) if side == "BUY" else price * (1 - SLIPPAGE)


def run_backtest(
    price_data: dict[str, pd.DataFrame],   # {ticker: OHLCV DataFrame}
    strategy_fn: Callable,
    strategy_name: str,
) -> BacktestResult:
    """
    price_data: 각 DataFrame은 index=DatetimeIndex, columns=[open,high,low,close,volume]
    strategy_fn: DataFrame → signal 컬럼 추가된 DataFrame 반환
    """
    # 신호 생성
    signals: dict[str, pd.DataFrame] = {}
    for ticker, df in price_data.items():
        try:
            signals[ticker] = strategy_fn(df.copy())
        except Exception:
            continue

    # 공통 날짜 범위 (모든 종목이 갖는 날짜만)
    all_dates = sorted(set.intersection(*[set(df.index) for df in signals.values()]))
    if not all_dates:
        raise ValueError("공통 날짜가 없습니다.")

    cash      = float(INITIAL_CASH)
    positions = {}   # {ticker: {"qty": int, "avg_price": float}}
    trades    = []
    equity    = []

    for date in all_dates:
        # ── 손절 체크 ─────────────────────────────
        for ticker in list(positions.keys()):
            pos   = positions[ticker]
            price = signals[ticker].loc[date, "close"]
            ret   = (price - pos["avg_price"]) / pos["avg_price"]
            if ret <= STOP_LOSS:
                ep  = _exec_price(price, "SELL")
                proceeds = pos["qty"] * ep * (1 - COMMISSION)
                cash    += proceeds
                trades.append(Trade(
                    date=str(date.date()), ticker=ticker, action="SELL",
                    qty=pos["qty"], price=ep, value=proceeds, reason="손절"
                ))
                del positions[ticker]

        # ── 신호 처리 ─────────────────────────────
        for ticker, df in signals.items():
            sig   = df.loc[date, "signal"]
            price = df.loc[date, "close"]

            if sig == 1 and ticker not in positions:
                if len(positions) >= MAX_POSITIONS:
                    continue
                avail  = cash * POSITION_SIZE
                ep     = _exec_price(price, "BUY")
                qty    = int(avail / (ep * (1 + COMMISSION)))
                if qty < 1:
                    continue
                cost   = qty * ep * (1 + COMMISSION)
                if cost > cash:
                    continue
                cash -= cost
                positions[ticker] = {"qty": qty, "avg_price": ep}
                trades.append(Trade(
                    date=str(date.date()), ticker=ticker, action="BUY",
                    qty=qty, price=ep, value=cost, reason="매수 신호"
                ))

            elif sig == -1 and ticker in positions:
                pos    = positions[ticker]
                ep     = _exec_price(price, "SELL")
                proceeds = pos["qty"] * ep * (1 - COMMISSION)
                cash  += proceeds
                trades.append(Trade(
                    date=str(date.date()), ticker=ticker, action="SELL",
                    qty=pos["qty"], price=ep, value=proceeds, reason="매도 신호"
                ))
                del positions[ticker]

        # ── 자산 평가 ─────────────────────────────
        pos_value = sum(
            positions[t]["qty"] * signals[t].loc[date, "close"]
            for t in positions if date in signals[t].index
        )
        equity.append({"date": str(date.date()), "value": cash + pos_value})

    # 남은 포지션 청산 (마지막 날)
    last_date = all_dates[-1]
    for ticker, pos in positions.items():
        price    = signals[ticker].loc[last_date, "close"]
        ep       = _exec_price(price, "SELL")
        proceeds = pos["qty"] * ep * (1 - COMMISSION)
        cash    += proceeds

    final_value = cash

    # ── 성과 지표 계산 ────────────────────────────
    curve_vals = [e["value"] for e in equity]
    total_ret  = (final_value - INITIAL_CASH) / INITIAL_CASH * 100

    # 연환산 수익률
    n_years = len(all_dates) / 252
    ann_ret = ((final_value / INITIAL_CASH) ** (1 / max(n_years, 0.01)) - 1) * 100

    # 샤프지수 (일간 수익률 기준, 무위험이자율 3%)
    daily_rets = pd.Series(curve_vals).pct_change().dropna()
    excess     = daily_rets - 0.03 / 252
    sharpe     = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    # MDD
    roll_max  = pd.Series(curve_vals).cummax()
    drawdown  = (pd.Series(curve_vals) - roll_max) / roll_max * 100
    max_dd    = drawdown.min()

    # 승률
    sell_trades = [t for t in trades if t.action == "SELL" and t.reason != "손절"]
    buy_map     = {}
    for t in trades:
        if t.action == "BUY":
            buy_map[t.ticker] = t.price
    wins = 0
    for t in sell_trades:
        if t.ticker in buy_map and t.price > buy_map[t.ticker]:
            wins += 1
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0

    return BacktestResult(
        strategy_name          = strategy_name,
        tickers                = list(price_data.keys()),
        total_return_pct       = round(total_ret, 2),
        annualized_return_pct  = round(ann_ret, 2),
        sharpe_ratio           = round(float(sharpe), 2),
        max_drawdown_pct       = round(float(max_dd), 2),
        win_rate_pct           = round(win_rate, 1),
        total_trades           = len(trades),
        final_value            = round(final_value, 0),
        trades                 = trades,
        equity_curve           = equity,
    )
