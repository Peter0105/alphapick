"""
AlphaPick Trading Strategies
각 전략은 OHLCV DataFrame을 받아 'signal' 컬럼을 반환.
signal: +1 = 매수, -1 = 매도, 0 = 홀드
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  공통 지표 계산
# ─────────────────────────────────────────────

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma20"]  = df["close"].rolling(20).mean()
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    # RSI(14)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # MACD(12,26,9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ATR(14) — 변동성
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    # 볼린저 밴드(20)
    df["bb_mid"]   = df["ma20"]
    df["bb_upper"] = df["ma20"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["ma20"] - 2 * df["close"].rolling(20).std()

    return df


# ─────────────────────────────────────────────
#  전략 1: MA Crossover (단순 이동평균 교차)
#   - 20일 MA가 50일 MA 위로 교차 → 매수
#   - 20일 MA가 50일 MA 아래로 교차 → 매도
# ─────────────────────────────────────────────

def strategy_ma_crossover(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_indicators(df)
    df["signal"] = 0
    prev_above = df["ma20"].shift(1) >= df["ma50"].shift(1)
    curr_above = df["ma20"] >= df["ma50"]
    df.loc[~prev_above & curr_above, "signal"] = 1   # 골든크로스
    df.loc[ prev_above & ~curr_above, "signal"] = -1  # 데드크로스
    return df


# ─────────────────────────────────────────────
#  전략 2: RSI Mean Reversion (RSI 평균회귀)
#   - RSI < 32 AND 가격 > 200일 MA (상승 추세) → 매수
#   - RSI > 68 OR 가격 < 200일 MA (하락 추세) → 매도
# ─────────────────────────────────────────────

def strategy_rsi_reversion(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_indicators(df)
    df["signal"] = 0

    buy_cond  = (df["rsi"] < 32) & (df["close"] > df["ma200"])
    sell_cond = (df["rsi"] > 68) | (df["close"] < df["ma200"])

    df.loc[buy_cond,  "signal"] = 1
    df.loc[sell_cond, "signal"] = -1
    return df


# ─────────────────────────────────────────────
#  전략 3: MACD + Bollinger 복합
#   - MACD 히스토그램 양전환 AND 볼린저 하단 반등 → 매수
#   - MACD 히스토그램 음전환 OR 볼린저 상단 돌파 후 하락 → 매도
# ─────────────────────────────────────────────

def strategy_macd_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_indicators(df)
    df["signal"] = 0

    macd_cross_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    macd_cross_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)
    near_lower_band = df["close"] <= df["bb_lower"] * 1.02

    buy_cond  = macd_cross_up & near_lower_band
    sell_cond = macd_cross_down

    df.loc[buy_cond,  "signal"] = 1
    df.loc[sell_cond, "signal"] = -1
    return df


STRATEGIES = {
    "MA 교차":        strategy_ma_crossover,
    "RSI 평균회귀":   strategy_rsi_reversion,
    "MACD+볼린저":    strategy_macd_bollinger,
}
