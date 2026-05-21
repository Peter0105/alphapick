"""
AlphaPick — 다중 전략 정의 (10가지)
각 전략은 OHLCV DataFrame → signal 컬럼 추가된 DataFrame 반환
signal: +1 매수 / -1 매도 / 0 홀드
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════
#  공통 지표 계산
# ══════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    # ── 이동평균 ─────────────────────────────────────
    for n in [10, 20, 50, 100, 200]:
        df[f"ma{n}"] = c.rolling(n).mean()

    # ── EMA ──────────────────────────────────────────
    df["ema12"] = c.ewm(span=12, adjust=False).mean()
    df["ema26"] = c.ewm(span=26, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────
    df["macd"]      = df["ema12"] - df["ema26"]
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # ── RSI(14) ───────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ── 스토캐스틱(14,3) ─────────────────────────────
    low14  = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = (c - low14) / (high14 - low14 + 1e-9) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── 볼린저 밴드(20,2) ─────────────────────────────
    std20         = c.rolling(20).std()
    df["bb_mid"]   = df["ma20"]
    df["bb_upper"] = df["ma20"] + 2 * std20
    df["bb_lower"] = df["ma20"] - 2 * std20
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ── ATR(14) ───────────────────────────────────────
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift()).abs(),
        (df["low"]  - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ── OBV (On-Balance Volume) ───────────────────────
    obv = (np.sign(c.diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"]    = obv
    df["obv_ma"] = obv.rolling(20).mean()

    # ── CCI(20) ───────────────────────────────────────
    tp = (df["high"] + df["low"] + c) / 3
    df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean()))

    # ── 모멘텀(Rate of Change, 20일) ──────────────────
    df["roc20"] = c.pct_change(20) * 100

    # ── 52주 고/저 대비 위치 ──────────────────────────
    df["high52"] = df["high"].rolling(252).max()
    df["low52"]  = df["low"].rolling(252).min()
    df["pos52"]  = (c - df["low52"]) / (df["high52"] - df["low52"] + 1e-9)

    return df


# ══════════════════════════════════════════════════════
#  전략 1: MA 교차 (20/50)
#  골든크로스→매수, 데드크로스→매도
# ══════════════════════════════════════════════════════
def strat_ma_cross(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0
    prev = df["ma20"].shift(1) >= df["ma50"].shift(1)
    curr = df["ma20"] >= df["ma50"]
    df.loc[~prev &  curr, "signal"] =  1
    df.loc[ prev & ~curr, "signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 2: 삼중 MA (10/50/200)
#  MA10>MA50>MA200 동시 만족 → 매수
#  MA10<MA50 → 매도
# ══════════════════════════════════════════════════════
def strat_triple_ma(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0
    bull = (df["ma10"] > df["ma50"]) & (df["ma50"] > df["ma200"])
    prev_bull = bull.shift(1).fillna(False)
    df.loc[ bull & ~prev_bull, "signal"] =  1
    df.loc[~bull &  prev_bull, "signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 3: RSI 역추세
#  RSI<30 AND 200MA 위 → 매수 / RSI>70 → 매도
# ══════════════════════════════════════════════════════
def strat_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0
    buy  = (df["rsi"] < 30) & (df["close"] > df["ma200"])
    sell = (df["rsi"] > 70)
    df.loc[buy,  "signal"] =  1
    df.loc[sell, "signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 4: RSI + MA 복합
#  RSI 30~50 회복 AND MA20>MA50 → 매수
#  RSI>65 OR MA20<MA50 → 매도
# ══════════════════════════════════════════════════════
def strat_rsi_ma(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0
    above = df["ma20"] > df["ma50"]
    rsi_rec = (df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)
    buy  = rsi_rec & above
    sell = (df["rsi"] > 65) | (df["ma20"] < df["ma50"])
    df.loc[buy,  "signal"] =  1
    df.loc[sell, "signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 5: MACD 교차
#  MACD가 시그널 위로 교차 → 매수 / 아래로 → 매도
# ══════════════════════════════════════════════════════
def strat_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0
    prev = df["macd"].shift(1) >= df["macd_sig"].shift(1)
    curr = df["macd"] >= df["macd_sig"]
    df.loc[~prev &  curr, "signal"] =  1
    df.loc[ prev & ~curr, "signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 6: MACD + RSI 복합
#  MACD 골든크로스 AND RSI 40~60 → 매수
#  MACD 데드크로스 AND RSI>65 → 매도
# ══════════════════════════════════════════════════════
def strat_macd_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0
    macd_up   = (df["macd"] > df["macd_sig"]) & (df["macd"].shift(1) <= df["macd_sig"].shift(1))
    macd_down = (df["macd"] < df["macd_sig"]) & (df["macd"].shift(1) >= df["macd_sig"].shift(1))
    buy  = macd_up   & (df["rsi"] > 40) & (df["rsi"] < 65)
    sell = macd_down & (df["rsi"] > 55)
    df.loc[buy,  "signal"] =  1
    df.loc[sell, "signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 7: 볼린저 밴드 역추세
#  하단 이탈 후 회복 → 매수 / 상단 돌파 후 하락 → 매도
# ══════════════════════════════════════════════════════
def strat_bollinger_rev(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0
    prev_below = df["close"].shift(1) < df["bb_lower"].shift(1)
    curr_above = df["close"] >= df["bb_lower"]
    prev_above = df["close"].shift(1) > df["bb_upper"].shift(1)
    curr_below = df["close"] <= df["bb_upper"]
    buy  = prev_below & curr_above
    sell = prev_above & curr_below
    df.loc[buy,  "signal"] =  1
    df.loc[sell, "signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 8: 스토캐스틱 교차
#  %K가 %D 위로 교차 AND 과매도(<20) 구간 탈출 → 매수
#  %K가 %D 아래로 교차 AND 과매수(>80) → 매도
# ══════════════════════════════════════════════════════
def strat_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0
    k_cross_up   = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
    k_cross_down = (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
    buy  = k_cross_up   & (df["stoch_k"] < 30)
    sell = k_cross_down & (df["stoch_k"] > 70)
    df.loc[buy,  "signal"] =  1
    df.loc[sell, "signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 9: OBV 모멘텀
#  OBV가 20일 MA 위로 교차 AND 가격 상승 → 매수
#  OBV가 20일 MA 아래로 → 매도
# ══════════════════════════════════════════════════════
def strat_obv(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0
    obv_above = df["obv"] > df["obv_ma"]
    prev_above = obv_above.shift(1).fillna(False)
    price_up   = df["close"] > df["close"].shift(1)
    df.loc[ obv_above & ~prev_above & price_up,  "signal"] =  1
    df.loc[~obv_above &  prev_above,             "signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 10: 종합 다중 지표 (앙상블)
#  RSI, MACD, MA, 볼린저 4개 지표 중 3개 이상 매수 신호 → 매수
#  3개 이상 매도 신호 → 매도
# ══════════════════════════════════════════════════════
def strat_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df["signal"] = 0

    # 각 지표별 점수
    scores = pd.DataFrame(index=df.index)
    scores["ma"]    = np.where(df["ma20"] > df["ma50"],  1, -1)
    scores["rsi"]   = np.where(df["rsi"]  < 45,          1, np.where(df["rsi"] > 60, -1, 0))
    scores["macd"]  = np.where(df["macd"] > df["macd_sig"], 1, -1)
    scores["bb"]    = np.where(df["bb_pct"] < 0.25,      1, np.where(df["bb_pct"] > 0.80, -1, 0))
    scores["stoch"] = np.where(df["stoch_k"] < 30,       1, np.where(df["stoch_k"] > 70, -1, 0))

    total = scores.sum(axis=1)
    prev_total = total.shift(1).fillna(0)

    # 3개 이상 동조 → 신호
    df.loc[(total >= 3) & (prev_total < 3),  "signal"] =  1
    df.loc[(total <= -3) & (prev_total > -3),"signal"] = -1
    return df


# ══════════════════════════════════════════════════════
#  전략 딕셔너리
# ══════════════════════════════════════════════════════
STRATEGIES = {
    "①MA 교차(20/50)":    strat_ma_cross,
    "②삼중MA(10/50/200)": strat_triple_ma,
    "③RSI 역추세":         strat_rsi,
    "④RSI+MA 복합":        strat_rsi_ma,
    "⑤MACD 교차":          strat_macd,
    "⑥MACD+RSI 복합":      strat_macd_rsi,
    "⑦볼린저 역추세":      strat_bollinger_rev,
    "⑧스토캐스틱":         strat_stochastic,
    "⑨OBV 모멘텀":         strat_obv,
    "⑩앙상블(5지표)":      strat_ensemble,
}
