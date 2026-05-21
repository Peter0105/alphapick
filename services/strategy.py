"""
AlphaPick — 20가지 트레이딩 전략
signal: +1 매수 / -1 매도 / 0 홀드
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════
#  공통 지표 계산
# ══════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # 이동평균
    for n in [5, 10, 20, 50, 100, 200]:
        df[f"ma{n}"] = c.rolling(n).mean()
    for n in [12, 26]:
        df[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()

    # MACD
    df["macd"]      = df["ema12"] - df["ema26"]
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # RSI(14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # 스토캐스틱(14,3)
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = (c - low14) / (high14 - low14 + 1e-9) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # 볼린저 밴드(20)
    std20          = c.rolling(20).std()
    df["bb_mid"]   = df["ma20"]
    df["bb_upper"] = df["ma20"] + 2 * std20
    df["bb_lower"] = df["ma20"] - 2 * std20
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ATR(14)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ADX(14)
    plus_dm  = (h.diff().clip(lower=0)).where(h.diff() > (-l.diff()).clip(lower=0), 0)
    minus_dm = ((-l.diff()).clip(lower=0)).where((-l.diff()) > h.diff().clip(lower=0), 0)
    atr14    = tr.rolling(14).mean()
    df["adx_plus"]  = (plus_dm.rolling(14).mean()  / atr14) * 100
    df["adx_minus"] = (minus_dm.rolling(14).mean() / atr14) * 100
    dx = (df["adx_plus"] - df["adx_minus"]).abs() / (df["adx_plus"] + df["adx_minus"] + 1e-9) * 100
    df["adx"] = dx.rolling(14).mean()

    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv"]    = obv
    df["obv_ma"] = obv.rolling(20).mean()

    # CCI(20)
    tp = (h + l + c) / 3
    df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean()))

    # Williams %R(14)
    df["willr"] = (high14 - c) / (high14 - low14 + 1e-9) * -100

    # Z-Score(20)
    df["zscore"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)

    # 켈트너 채널(20, ATR×2)
    df["kc_mid"]   = df["ma20"]
    df["kc_upper"] = df["ma20"] + 2 * df["atr"]
    df["kc_lower"] = df["ma20"] - 2 * df["atr"]

    # 돈치안 채널(20일 고/저)
    df["don_high"] = h.rolling(20).max()
    df["don_low"]  = l.rolling(20).min()

    # 파라볼릭 SAR (간략 구현)
    df["sar"] = _calc_sar(df)

    # 이치모쿠
    nine_high  = h.rolling(9).max();  nine_low  = l.rolling(9).min()
    twenty_six_high = h.rolling(26).max(); twenty_six_low = l.rolling(26).min()
    df["ich_tenkan"] = (nine_high + nine_low) / 2
    df["ich_kijun"]  = (twenty_six_high + twenty_six_low) / 2
    df["ich_spanA"]  = ((df["ich_tenkan"] + df["ich_kijun"]) / 2).shift(26)
    fiftytwo_high = h.rolling(52).max(); fiftytwo_low = l.rolling(52).min()
    df["ich_spanB"]  = ((fiftytwo_high + fiftytwo_low) / 2).shift(26)

    # 모멘텀(ROC)
    df["roc20"]  = c.pct_change(20) * 100
    df["roc60"]  = c.pct_change(60) * 100
    df["roc120"] = c.pct_change(120) * 100

    # 변동성 돌파 (래리 윌리엄스 K=0.5)
    df["vb_target"] = df["ma20"] + df["atr"] * 0.5

    return df


def _calc_sar(df: pd.DataFrame) -> pd.Series:
    """Parabolic SAR 간략 계산"""
    sar   = df["close"].copy()
    trend = pd.Series(1, index=df.index)
    ep    = df["high"].copy()
    af    = pd.Series(0.02, index=df.index)
    AF_STEP, AF_MAX = 0.02, 0.20

    for i in range(2, len(df)):
        prev_sar = sar.iloc[i-1]
        prev_ep  = ep.iloc[i-1]
        prev_af  = af.iloc[i-1]
        prev_tr  = trend.iloc[i-1]

        if prev_tr == 1:
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            new_sar = min(new_sar, df["low"].iloc[i-1], df["low"].iloc[i-2])
            if df["low"].iloc[i] < new_sar:
                trend.iloc[i] = -1
                sar.iloc[i]   = prev_ep
                ep.iloc[i]    = df["low"].iloc[i]
                af.iloc[i]    = AF_STEP
            else:
                trend.iloc[i] = 1
                sar.iloc[i]   = new_sar
                if df["high"].iloc[i] > prev_ep:
                    ep.iloc[i] = df["high"].iloc[i]
                    af.iloc[i] = min(prev_af + AF_STEP, AF_MAX)
                else:
                    ep.iloc[i] = prev_ep
                    af.iloc[i] = prev_af
        else:
            new_sar = prev_sar - prev_af * (prev_sar - prev_ep)
            new_sar = max(new_sar, df["high"].iloc[i-1], df["high"].iloc[i-2])
            if df["high"].iloc[i] > new_sar:
                trend.iloc[i] = 1
                sar.iloc[i]   = prev_ep
                ep.iloc[i]    = df["high"].iloc[i]
                af.iloc[i]    = AF_STEP
            else:
                trend.iloc[i] = -1
                sar.iloc[i]   = new_sar
                if df["low"].iloc[i] < prev_ep:
                    ep.iloc[i] = df["low"].iloc[i]
                    af.iloc[i] = min(prev_af + AF_STEP, AF_MAX)
                else:
                    ep.iloc[i] = prev_ep
                    af.iloc[i] = prev_af
    return sar


# ══════════════════════════════════════════════════════════════
#  기존 전략 (1~10)
# ══════════════════════════════════════════════════════════════

def strat_ma_cross(df):
    df = add_indicators(df); df["signal"] = 0
    prev = df["ma20"].shift(1) >= df["ma50"].shift(1)
    curr = df["ma20"] >= df["ma50"]
    df.loc[~prev &  curr, "signal"] =  1
    df.loc[ prev & ~curr, "signal"] = -1
    return df

def strat_triple_ma(df):
    df = add_indicators(df); df["signal"] = 0
    bull = (df["ma10"] > df["ma50"]) & (df["ma50"] > df["ma200"])
    prev = bull.shift(1).fillna(False)
    df.loc[ bull & ~prev, "signal"] =  1
    df.loc[~bull &  prev, "signal"] = -1
    return df

def strat_rsi(df):
    df = add_indicators(df); df["signal"] = 0
    df.loc[(df["rsi"] < 30) & (df["close"] > df["ma200"]), "signal"] =  1
    df.loc[(df["rsi"] > 70), "signal"] = -1
    return df

def strat_rsi_ma(df):
    df = add_indicators(df); df["signal"] = 0
    above = df["ma20"] > df["ma50"]
    rsi_rec = (df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)
    df.loc[rsi_rec & above, "signal"] = 1
    df.loc[(df["rsi"] > 65) | (df["ma20"] < df["ma50"]), "signal"] = -1
    return df

def strat_macd(df):
    df = add_indicators(df); df["signal"] = 0
    prev = df["macd"].shift(1) >= df["macd_sig"].shift(1)
    curr = df["macd"] >= df["macd_sig"]
    df.loc[~prev &  curr, "signal"] =  1
    df.loc[ prev & ~curr, "signal"] = -1
    return df

def strat_macd_rsi(df):
    df = add_indicators(df); df["signal"] = 0
    mu = (df["macd"] > df["macd_sig"]) & (df["macd"].shift(1) <= df["macd_sig"].shift(1))
    md = (df["macd"] < df["macd_sig"]) & (df["macd"].shift(1) >= df["macd_sig"].shift(1))
    df.loc[mu & (df["rsi"] > 40) & (df["rsi"] < 65), "signal"] =  1
    df.loc[md & (df["rsi"] > 55), "signal"] = -1
    return df

def strat_bollinger_rev(df):
    df = add_indicators(df); df["signal"] = 0
    prev_below = df["close"].shift(1) < df["bb_lower"].shift(1)
    curr_above = df["close"] >= df["bb_lower"]
    prev_above = df["close"].shift(1) > df["bb_upper"].shift(1)
    curr_below = df["close"] <= df["bb_upper"]
    df.loc[prev_below & curr_above, "signal"] =  1
    df.loc[prev_above & curr_below, "signal"] = -1
    return df

def strat_stochastic(df):
    df = add_indicators(df); df["signal"] = 0
    ku = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
    kd = (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
    df.loc[ku & (df["stoch_k"] < 30), "signal"] =  1
    df.loc[kd & (df["stoch_k"] > 70), "signal"] = -1
    return df

def strat_obv(df):
    df = add_indicators(df); df["signal"] = 0
    above = df["obv"] > df["obv_ma"]
    prev  = above.shift(1).fillna(False)
    df.loc[ above & ~prev & (df["close"] > df["close"].shift(1)), "signal"] =  1
    df.loc[~above &  prev, "signal"] = -1
    return df

def strat_ensemble(df):
    df = add_indicators(df); df["signal"] = 0
    s = pd.DataFrame(index=df.index)
    s["ma"]    = np.where(df["ma20"] > df["ma50"],   1, -1)
    s["rsi"]   = np.where(df["rsi"]  < 45,           1, np.where(df["rsi"] > 60, -1, 0))
    s["macd"]  = np.where(df["macd"] > df["macd_sig"], 1, -1)
    s["bb"]    = np.where(df["bb_pct"] < 0.25,       1, np.where(df["bb_pct"] > 0.80, -1, 0))
    s["stoch"] = np.where(df["stoch_k"] < 30,        1, np.where(df["stoch_k"] > 70, -1, 0))
    tot = s.sum(axis=1)
    df.loc[(tot >= 3) & (tot.shift(1).fillna(0) < 3),  "signal"] =  1
    df.loc[(tot <= -3) & (tot.shift(1).fillna(0) > -3), "signal"] = -1
    return df


# ══════════════════════════════════════════════════════════════
#  신규 전략 (11~20)
# ══════════════════════════════════════════════════════════════

# ── 전략 11: 터틀 트레이딩 (돈치안 채널 20일) ─────────────────
# 리처드 데니스가 만든 추세추종 전략
# 20일 최고가 돌파 → 매수 / 20일 최저가 이탈 → 매도
def strat_turtle(df):
    df = add_indicators(df); df["signal"] = 0
    prev_don_h = df["don_high"].shift(1)
    prev_don_l = df["don_low"].shift(1)
    df.loc[df["close"] > prev_don_h, "signal"] =  1   # 상단 돌파
    df.loc[df["close"] < prev_don_l, "signal"] = -1   # 하단 이탈
    return df


# ── 전략 12: 파라볼릭 SAR ────────────────────────────────────
# 추세추종: 가격이 SAR 위 → 매수, SAR 아래 → 매도
def strat_parabolic_sar(df):
    df = add_indicators(df); df["signal"] = 0
    above     = df["close"] > df["sar"]
    prev_above = above.shift(1).fillna(False)
    df.loc[ above & ~prev_above, "signal"] =  1   # SAR 위로 교차
    df.loc[~above &  prev_above, "signal"] = -1   # SAR 아래로 교차
    return df


# ── 전략 13: ADX 추세강도 필터 + MA ─────────────────────────
# ADX > 25 (강한 추세) 일 때만 MA 교차 매매
def strat_adx_ma(df):
    df = add_indicators(df); df["signal"] = 0
    strong = df["adx"] > 25
    prev   = (df["ma20"].shift(1) >= df["ma50"].shift(1)) & strong.shift(1).fillna(False)
    curr   = (df["ma20"] >= df["ma50"]) & strong
    df.loc[~prev &  curr, "signal"] =  1
    df.loc[ prev & ~curr, "signal"] = -1
    return df


# ── 전략 14: 이치모쿠 클라우드 ───────────────────────────────
# 전환선 > 기준선 AND 가격 > 구름 위 → 매수
def strat_ichimoku(df):
    df = add_indicators(df); df["signal"] = 0
    cloud_top = df[["ich_spanA","ich_spanB"]].max(axis=1)
    cloud_bot = df[["ich_spanA","ich_spanB"]].min(axis=1)
    bullish = (
        (df["ich_tenkan"] > df["ich_kijun"]) &
        (df["close"] > cloud_top)
    )
    bearish = (
        (df["ich_tenkan"] < df["ich_kijun"]) |
        (df["close"] < cloud_bot)
    )
    prev_bull = bullish.shift(1).fillna(False)
    df.loc[ bullish & ~prev_bull, "signal"] =  1
    df.loc[ bearish & ~bearish.shift(1).fillna(False), "signal"] = -1
    return df


# ── 전략 15: Williams %R ─────────────────────────────────────
# %R < -80 (과매도) → 매수 / %R > -20 (과매수) → 매도
def strat_williams_r(df):
    df = add_indicators(df); df["signal"] = 0
    oversold  = df["willr"] < -80
    overbought= df["willr"] > -20
    prev_os   = oversold.shift(1).fillna(False)
    prev_ob   = overbought.shift(1).fillna(False)
    # 과매도 → 회복 진입
    df.loc[prev_os  & ~oversold,   "signal"] =  1
    # 과매수 → 하락 진입
    df.loc[prev_ob  & ~overbought, "signal"] = -1
    return df


# ── 전략 16: Z-Score 평균회귀 ────────────────────────────────
# 통계적 평균회귀: Z < -1.5 → 매수 / Z > 1.5 → 매도
def strat_zscore(df):
    df = add_indicators(df); df["signal"] = 0
    prev_z = df["zscore"].shift(1)
    df.loc[(prev_z < -1.5) & (df["zscore"] >= -1.5), "signal"] =  1   # 과매도 탈출
    df.loc[(prev_z >  1.5) & (df["zscore"] <=  1.5), "signal"] = -1   # 과매수 탈출
    return df


# ── 전략 17: 켈트너 채널 ─────────────────────────────────────
# ATR 기반 채널 — 볼린저보다 뻥튀기 적음
def strat_keltner(df):
    df = add_indicators(df); df["signal"] = 0
    prev_below = df["close"].shift(1) < df["kc_lower"].shift(1)
    curr_rec   = df["close"] >= df["kc_lower"]
    prev_above = df["close"].shift(1) > df["kc_upper"].shift(1)
    curr_drop  = df["close"] <= df["kc_upper"]
    df.loc[prev_below & curr_rec,  "signal"] =  1
    df.loc[prev_above & curr_drop, "signal"] = -1
    return df


# ── 전략 18: 듀얼 모멘텀 (게리 안토나치) ────────────────────
# 절대 모멘텀(12개월 수익률 > 0) + 상대 모멘텀(종목 중 상위권)
def strat_dual_momentum(df):
    df = add_indicators(df); df["signal"] = 0
    abs_mom   = df["roc120"] > 0        # 절대 모멘텀: 6개월 양수
    rel_rank  = df["roc60"].rolling(5).mean() > 0  # 상대 모멘텀 대리 (60일 ROC 양수)
    in_market  = abs_mom & rel_rank
    prev_in   = in_market.shift(1).fillna(False)
    df.loc[ in_market & ~prev_in, "signal"] =  1
    df.loc[~in_market &  prev_in, "signal"] = -1
    return df


# ── 전략 19: 트리플 스크린 (알렉산더 엘더) ───────────────────
# 1스크린: 주간 추세(MA50 기울기) 상승
# 2스크린: 일간 RSI 과매도(<40) 반등
# 3스크린: 당일 가격 전일 고가 돌파
def strat_triple_screen(df):
    df = add_indicators(df); df["signal"] = 0
    weekly_up  = df["ma50"] > df["ma50"].shift(5)          # 주간 추세 상승
    rsi_recov  = (df["rsi"] > 40) & (df["rsi"].shift(1) <= 40)  # 일간 RSI 반등
    breakout   = df["close"] > df["high"].shift(1)         # 전일 고가 돌파
    buy        = weekly_up & rsi_recov & breakout
    sell       = (df["ma50"] < df["ma50"].shift(5)) | (df["rsi"] > 70)
    df.loc[buy,  "signal"] =  1
    df.loc[sell, "signal"] = -1
    return df


# ── 전략 20: 변동성 돌파 (래리 윌리엄스 K=0.5) ───────────────
# 당일 시가 + (전일 고가-저가) × K 돌파 → 매수
# 다음날 시가 청산 (여기서는 당일 종가 기준으로 단순화)
def strat_volatility_breakout(df):
    df = add_indicators(df); df["signal"] = 0
    target = df["close"].shift(1) + (df["high"].shift(1) - df["low"].shift(1)) * 0.5
    buy    = df["close"] > target
    prev_b = buy.shift(1).fillna(False)
    df.loc[ buy & ~prev_b, "signal"] =  1
    df.loc[~buy &  prev_b, "signal"] = -1
    return df


# ══════════════════════════════════════════════════════════════
#  전략 딕셔너리 (20개)
# ══════════════════════════════════════════════════════════════
STRATEGIES = {
    # ── 기존 10 ──
    " 1.MA교차(20/50)":         strat_ma_cross,
    " 2.삼중MA(10/50/200)":     strat_triple_ma,
    " 3.RSI역추세":              strat_rsi,
    " 4.RSI+MA복합":             strat_rsi_ma,
    " 5.MACD교차":               strat_macd,
    " 6.MACD+RSI복합":           strat_macd_rsi,
    " 7.볼린저역추세":            strat_bollinger_rev,
    " 8.스토캐스틱":              strat_stochastic,
    " 9.OBV모멘텀":              strat_obv,
    "10.앙상블(5지표)":          strat_ensemble,
    # ── 신규 10 ──
    "11.터틀(돈치안20)":         strat_turtle,
    "12.파라볼릭SAR":            strat_parabolic_sar,
    "13.ADX+MA":                strat_adx_ma,
    "14.이치모쿠":               strat_ichimoku,
    "15.Williams%R":            strat_williams_r,
    "16.Z-Score평균회귀":        strat_zscore,
    "17.켈트너채널":             strat_keltner,
    "18.듀얼모멘텀":             strat_dual_momentum,
    "19.트리플스크린(엘더)":     strat_triple_screen,
    "20.변동성돌파(LW)":         strat_volatility_breakout,
}
