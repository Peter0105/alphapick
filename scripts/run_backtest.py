"""
AlphaPick 종합 백테스트 v3
- 20개 전략 전수 비교
- 워크포워드 검증 (4회 OOS)
- 마켓 레짐별 분석 (약세장 / 회복 / 강세장)
- 종합 랭킹 및 최우수 전략 선정

사용법: py scripts/run_backtest.py
"""

import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from services.strategy  import STRATEGIES
from services.backtester import run_backtest, BacktestResult, INITIAL_CASH

# ══════════════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════════════
WATCHLIST  = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"]
FULL_START = "2020-01-01"   # 5년 (충분한 롤링 윈도우)
FULL_END   = datetime.today().strftime("%Y-%m-%d")

# 워크포워드 설정: IS=18개월, OOS=6개월, 4회 반복
WF_IS_MONTHS  = 18
WF_OOS_MONTHS = 6
WF_ROUNDS     = 4

W = 110
DIV = "─" * W

def hdr(txt): print(f"\n{'═'*W}\n  {txt}\n{'═'*W}")
def sub(txt): print(f"\n{DIV}\n  {txt}\n{DIV}")

# ══════════════════════════════════════════════════════════════
#  데이터 다운로드
# ══════════════════════════════════════════════════════════════
hdr("AlphaPick 종합 백테스트 v3  —  20개 전략 × 워크포워드 × 레짐 분석")
print(f"  기간: {FULL_START} ~ {FULL_END}  │  초기자본: {INITIAL_CASH:,}원  │  전략: {len(STRATEGIES)}개")

print("\n📥 데이터 다운로드 중...\n")
raw: dict[str, pd.DataFrame] = {}
for ticker in WATCHLIST:
    try:
        df = yf.download(ticker, start=FULL_START, end=FULL_END,
                         auto_adjust=True, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        raw[ticker] = df
        print(f"  ✓ {ticker}: {len(df):,}일  {df.index[0].date()} ~ {df.index[-1].date()}")
    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

assert raw, "데이터 없음"

# Buy & Hold 벤치마크
bnh = {t: (df["close"].iloc[-1]/df["close"].iloc[0]-1)*100 for t, df in raw.items()}
bnh_avg = np.mean(list(bnh.values()))
print(f"\n  Buy&Hold 평균: {bnh_avg:+.1f}%  │  " +
      "  ".join(f"{t}: {r:+.0f}%" for t, r in bnh.items()))


# ══════════════════════════════════════════════════════════════
#  헬퍼: 기간 슬라이스
# ══════════════════════════════════════════════════════════════
def slice_data(data: dict, start: str, end: str) -> dict:
    out = {}
    for t, df in data.items():
        s = df.loc[start:end]
        if len(s) >= 60:
            out[t] = s
    return out


def composite_score(r: BacktestResult) -> float:
    return (r.sharpe_ratio    * 0.35
          + r.sortino_ratio   * 0.20
          + r.annualized_return_pct * 0.20
          + r.win_rate_pct    * 0.10
          + r.profit_factor   * 0.05
          - abs(r.max_drawdown_pct) * 0.10)


# ══════════════════════════════════════════════════════════════
#  파트 1: 전체 기간 풀 백테스트
# ══════════════════════════════════════════════════════════════
sub("PART 1 — 전체 기간 풀 백테스트")

# 전체 기간 (2020~now, 단 지표 계산용 워밍업 제외 → 2021-01 이후)
full_data = slice_data(raw, "2021-01-01", FULL_END)
print(f"\n  분석 기간: 2021-01-01 ~ {FULL_END}  ({len(next(iter(full_data.values()))):,}일)\n")
print(f"  {'전략':<28} {'총수익(%)':>10} {'연환산(%)':>9} {'샤프':>7} {'소르티노':>9} "
      f"{'칼마':>7} {'MDD(%)':>8} {'승률(%)':>8} {'PF':>6} {'거래':>6} {'최종자산':>16}")
print(f"  {DIV}")

full_results: list[BacktestResult] = []
t0 = time.time()
for name, fn in STRATEGIES.items():
    try:
        r = run_backtest(full_data, fn, name)
        full_results.append(r)
        row = r.row()
        print(f"  {name:<28} {row['총수익률(%)']:>10} {row['연환산(%)']:>9} "
              f"{row['샤프']:>7} {row['소르티노']:>9} {row['칼마']:>7} "
              f"{row['MDD(%)']:>8} {row['승률(%)']:>8} {row['P.F']:>6} "
              f"{row['거래수']:>6} {row['최종자산(원)']:>16}")
    except Exception as e:
        print(f"  {name:<28}  오류: {e}")

print(f"  {DIV}")
print(f"  {'Buy&Hold 평균':<28} {bnh_avg:>+9.1f}%  (벤치마크)")
print(f"\n  ⏱ {time.time()-t0:.1f}초")


# ══════════════════════════════════════════════════════════════
#  파트 2: 마켓 레짐별 성과
# ══════════════════════════════════════════════════════════════
sub("PART 2 — 마켓 레짐별 성과 분석")

REGIMES = {
    "약세장(2022)":   ("2022-01-01", "2022-12-31"),
    "회복장(2023)":   ("2023-01-01", "2023-12-31"),
    "강세장(2024)":   ("2024-01-01", "2024-12-31"),
    "최근6개월":      ((datetime.today()-timedelta(days=180)).strftime("%Y-%m-%d"), FULL_END),
}

regime_scores: dict[str, dict] = {n: {} for n in STRATEGIES}

for regime_name, (rs, re) in REGIMES.items():
    rdata = slice_data(raw, rs, re)
    if not rdata:
        print(f"\n  {regime_name}: 데이터 없음")
        continue

    print(f"\n  ┌─ {regime_name} ({rs} ~ {re}) ─────")
    print(f"  │  {'전략':<28} {'총수익(%)':>10} {'샤프':>7} {'MDD(%)':>8} {'승률(%)':>8}")
    print(f"  │  {'─'*65}")
    for name, fn in STRATEGIES.items():
        try:
            r = run_backtest(rdata, fn, name)
            regime_scores[name][regime_name] = r.total_return_pct
            flag = "🟢" if r.total_return_pct > 0 else "🔴"
            print(f"  │  {name:<28} {r.total_return_pct:>+9.1f}% {r.sharpe_ratio:>7.2f} "
                  f"{r.max_drawdown_pct:>8.1f}% {r.win_rate_pct:>7.1f}%  {flag}")
        except Exception:
            pass
    print(f"  └{'─'*70}")


# ══════════════════════════════════════════════════════════════
#  파트 3: 워크포워드 검증 (IS=18개월, OOS=6개월, 4회)
# ══════════════════════════════════════════════════════════════
sub("PART 3 — 워크포워드 검증 (In-Sample → Out-of-Sample 4회)")

# OOS 시작점: 2022-07, 2023-01, 2023-07, 2024-01
wf_windows = []
oos_start = datetime(2022, 7, 1)
for _ in range(WF_ROUNDS):
    is_end  = oos_start - timedelta(days=1)
    is_start= oos_start - timedelta(days=WF_IS_MONTHS * 30)
    oos_end = oos_start + timedelta(days=WF_OOS_MONTHS * 30)
    wf_windows.append((is_start.strftime("%Y-%m-%d"),
                       is_end.strftime("%Y-%m-%d"),
                       oos_start.strftime("%Y-%m-%d"),
                       oos_end.strftime("%Y-%m-%d")))
    oos_start += timedelta(days=WF_OOS_MONTHS * 30)

print(f"\n  IS={WF_IS_MONTHS}개월 학습 → OOS={WF_OOS_MONTHS}개월 검증  ×  {WF_ROUNDS}회\n")

wf_oos_returns: dict[str, list] = {n: [] for n in STRATEGIES}

for i, (is_s, is_e, oos_s, oos_e) in enumerate(wf_windows, 1):
    print(f"  ── Round {i}: IS {is_s}~{is_e} │ OOS {oos_s}~{oos_e}")
    is_data  = slice_data(raw, is_s, is_e)
    oos_data = slice_data(raw, oos_s, oos_e)
    if not is_data or not oos_data:
        print("     데이터 부족 — 스킵")
        continue

    # IS에서 최우수 전략 찾기
    is_results = {}
    for name, fn in STRATEGIES.items():
        try:
            r = run_backtest(is_data, fn, name)
            is_results[name] = r
        except Exception:
            pass

    # OOS 성과 검증
    oos_results = {}
    for name, fn in STRATEGIES.items():
        try:
            r = run_backtest(oos_data, fn, name)
            oos_results[name] = r
            wf_oos_returns[name].append(r.total_return_pct)
        except Exception:
            wf_oos_returns[name].append(0.0)

    # 라운드별 상위 5개 IS vs OOS
    top5_is = sorted(is_results.items(), key=lambda x: composite_score(x[1]), reverse=True)[:5]
    print(f"  {'전략':<28} {'IS수익(%)':>10} {'OOS수익(%)':>11} {'검증':>6}")
    print(f"  {'─'*60}")
    for name, is_r in top5_is:
        oos_r = oos_results.get(name)
        if oos_r:
            ok = "✅" if oos_r.total_return_pct > 0 else "❌"
            print(f"  {name:<28} {is_r.total_return_pct:>+9.1f}% {oos_r.total_return_pct:>+10.1f}%  {ok}")
    print()


# ══════════════════════════════════════════════════════════════
#  파트 4: 종합 랭킹
# ══════════════════════════════════════════════════════════════
sub("PART 4 — 종합 최종 랭킹")

# 종합점수 = 풀백테스트 70% + WF OOS 평균 30%
ranking = []
for r in full_results:
    fs  = composite_score(r)
    wf_avg = np.mean(wf_oos_returns.get(r.strategy_name, [0])) if wf_oos_returns.get(r.strategy_name) else 0
    # 레짐 일관성 점수 (약세장에서도 양수면 보너스)
    regime_vals = list(regime_scores.get(r.strategy_name, {}).values())
    regime_pos  = sum(1 for v in regime_vals if v > 0) / len(regime_vals) if regime_vals else 0
    final_score = fs * 0.60 + wf_avg * 0.30 + regime_pos * 5
    ranking.append((r.strategy_name, final_score, fs, wf_avg, regime_pos, r))

ranking.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {'순위':<5} {'전략':<28} {'종합점수':>9} {'풀백테스트':>10} {'WF-OOS평균(%)':>14} "
      f"{'레짐일관성':>11} {'총수익(%)':>10} {'샤프':>7} {'승률(%)':>8}")
print(f"  {'─'*105}")
for i, (name, fscore, fs, wf_avg, rc, r) in enumerate(ranking, 1):
    bar = "★" * min(i, 1) if i == 1 else ("☆" if i <= 3 else " ")
    print(f"  {i:2}위 {bar}  {name:<28} {fscore:>8.2f}  {fs:>9.2f}  {wf_avg:>+12.1f}%  "
          f"{'%.0f%%'%( rc*100):>11}  {r.total_return_pct:>+9.1f}%  {r.sharpe_ratio:>6.2f}  {r.win_rate_pct:>7.1f}%")

best_name, best_score, _, wf_best, _, best_r = ranking[0]

print(f"\n{'═'*W}")
print(f"  🏆 최우수 전략: {best_name}")
print(f"  ─────────────────────────────────────────────────────────")
print(f"  총수익률:     {best_r.total_return_pct:+.1f}%")
print(f"  연환산수익률: {best_r.annualized_return_pct:+.1f}%")
print(f"  샤프지수:     {best_r.sharpe_ratio:.2f}")
print(f"  소르티노:     {best_r.sortino_ratio:.2f}")
print(f"  칼마비율:     {best_r.calmar_ratio:.2f}")
print(f"  최대낙폭(MDD):{best_r.max_drawdown_pct:.1f}%")
print(f"  승률:         {best_r.win_rate_pct:.1f}%")
print(f"  프로핏팩터:   {best_r.profit_factor:.2f}")
print(f"  거래 횟수:    {best_r.total_trades}회")
print(f"  WF OOS 평균:  {wf_best:+.1f}%")
print(f"  최종 자산:    {best_r.final_value:,.0f}원  (초기 {INITIAL_CASH:,}원)")
print(f"{'═'*W}\n")

# 2위~5위도 출력
print("  ── 차점 전략 (2~5위) ──")
for i, (name, fscore, fs, wf_avg, rc, r) in enumerate(ranking[1:5], 2):
    print(f"  {i}위 {name:<28}  총수익 {r.total_return_pct:+.1f}%  샤프 {r.sharpe_ratio:.2f}  "
          f"OOS평균 {wf_avg:+.1f}%  점수 {fscore:.2f}")


# ══════════════════════════════════════════════════════════════
#  결과 저장
# ══════════════════════════════════════════════════════════════
out_dir = Path(__file__).parent.parent / "data"
out_dir.mkdir(exist_ok=True)

save = {
    "run_at":         datetime.now().isoformat(),
    "period":         {"start": "2021-01-01", "end": FULL_END},
    "tickers":        WATCHLIST,
    "buy_hold_avg":   round(bnh_avg, 2),
    "best_strategy":  best_name,
    "best_score":     round(best_score, 3),
    "best_metrics": {
        "total_return_pct":      best_r.total_return_pct,
        "annualized_return_pct": best_r.annualized_return_pct,
        "sharpe_ratio":          best_r.sharpe_ratio,
        "sortino_ratio":         best_r.sortino_ratio,
        "calmar_ratio":          best_r.calmar_ratio,
        "max_drawdown_pct":      best_r.max_drawdown_pct,
        "win_rate_pct":          best_r.win_rate_pct,
        "profit_factor":         best_r.profit_factor,
        "total_trades":          best_r.total_trades,
        "final_value":           best_r.final_value,
        "wf_oos_avg":            round(wf_best, 2),
    },
    "ranking": [
        {
            "rank":        i + 1,
            "strategy":    name,
            "final_score": round(fscore, 3),
            "full_bt_score": round(fs, 3),
            "wf_oos_avg":  round(wf_avg, 2),
            "regime_consistency": round(rc, 2),
            "total_return_pct":      r.total_return_pct,
            "annualized_return_pct": r.annualized_return_pct,
            "sharpe_ratio":          r.sharpe_ratio,
            "sortino_ratio":         r.sortino_ratio,
            "calmar_ratio":          r.calmar_ratio,
            "max_drawdown_pct":      r.max_drawdown_pct,
            "win_rate_pct":          r.win_rate_pct,
            "profit_factor":         r.profit_factor,
            "total_trades":          r.total_trades,
        }
        for i, (name, fscore, fs, wf_avg, rc, r) in enumerate(ranking)
    ],
    "equity_curve": best_r.equity_curve,
}

result_file = out_dir / "backtest_result.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(save, f, ensure_ascii=False, indent=2)

print(f"\n💾 결과 저장: {result_file}")
print(f"✅ 완료!\n")
