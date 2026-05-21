"""
AlphaPick 종합 백테스트
사용법: py scripts/run_backtest.py
10가지 전략 × 5~7개 종목 전수 비교분석
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta
import json
import time
import pandas as pd
import numpy as np
import yfinance as yf

from services.strategy  import STRATEGIES
from services.backtester import run_backtest, BacktestResult, INITIAL_CASH

# ── 설정 ──────────────────────────────────────────────────
WATCHLIST  = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"]
START_DATE = (datetime.today() - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
END_DATE   = datetime.today().strftime("%Y-%m-%d")
DIVIDER    = "─" * 110

# ── 헤더 ──────────────────────────────────────────────────
print()
print("═" * 110)
print("  AlphaPick 종합 백테스트 엔진 v2")
print(f"  기간: {START_DATE} ~ {END_DATE}  (약 4년)")
print(f"  종목: {', '.join(WATCHLIST)}")
print(f"  초기자본: {INITIAL_CASH:,}원  │  슬리피지: 0.1%  │  수수료: 0.05%(편도)  │  손절: -8%")
print(f"  전략 수: {len(STRATEGIES)}개")
print("═" * 110)

# ── 데이터 다운로드 ────────────────────────────────────────
print("\n📥 과거 데이터 다운로드 중...\n")
price_data: dict[str, pd.DataFrame] = {}

for ticker in WATCHLIST:
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"  ⚠  {ticker}: 데이터 없음")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        price_data[ticker] = df
        print(f"  ✓  {ticker}: {len(df):,}일  │  {df.index[0].date()} ~ {df.index[-1].date()}")
    except Exception as e:
        print(f"  ✗  {ticker}: {e}")

if not price_data:
    print("❌ 데이터 없음")
    sys.exit(1)

# Buy & Hold 벤치마크 (SPY 대신 전체 종목 동일비중)
print("\n📊 Buy & Hold 벤치마크 계산 중...")
bnh_returns = []
for ticker, df in price_data.items():
    ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    bnh_returns.append(ret)
    print(f"  {ticker}: {ret:+.1f}%")
bnh_avg = np.mean(bnh_returns)
print(f"  평균 Buy&Hold 수익률: {bnh_avg:+.1f}%")

# ── 백테스트 실행 ──────────────────────────────────────────
print(f"\n{'─'*110}")
print(f"  {'전략':<22} {'총수익률':>9} {'연환산':>8} {'샤프':>7} {'소르티노':>9} {'칼마':>7} {'MDD':>8} {'승률':>7} {'P.F':>6} {'거래수':>7} {'최종자산':>16}")
print(f"{'─'*110}")

results: list[BacktestResult] = []
t0 = time.time()

for name, fn in STRATEGIES.items():
    try:
        r = run_backtest(price_data, fn, name)
        results.append(r)
        row = r.row()
        print(
            f"  {name:<22} "
            f"{row['총수익률(%)']:>9} "
            f"{row['연환산(%)']:>8} "
            f"{row['샤프']:>7} "
            f"{row['소르티노']:>9} "
            f"{row['칼마']:>7} "
            f"{row['MDD(%)']:>8} "
            f"{row['승률(%)']:>7} "
            f"{row['P.F']:>6} "
            f"{row['거래수']:>7} "
            f"{row['최종자산(원)']:>16}"
        )
    except Exception as e:
        print(f"  {name:<22}  오류: {e}")

elapsed = time.time() - t0
print(f"{'─'*110}")
print(f"  Buy&Hold 평균                   {bnh_avg:>+8.1f}%  (벤치마크)")
print(f"\n  ⏱  소요시간: {elapsed:.1f}초")

# ── 랭킹 ──────────────────────────────────────────────────
if not results:
    print("❌ 모든 전략 실패")
    sys.exit(1)

def composite_score(r: BacktestResult) -> float:
    """종합점수: 샤프 40% + 소르티노 20% + 연환산수익률 20% + 승률 10% - |MDD| 10%"""
    return (r.sharpe_ratio * 0.40
            + r.sortino_ratio * 0.20
            + r.annualized_return_pct * 0.20
            + r.win_rate_pct * 0.10
            - abs(r.max_drawdown_pct) * 0.10)

results.sort(key=composite_score, reverse=True)

print(f"\n{'═'*110}")
print("  📊 종합 순위 (샤프40% + 소르티노20% + 연환산20% + 승률10% - |MDD|10%)\n")
for i, r in enumerate(results, 1):
    sc = composite_score(r)
    bar = "█" * min(int(max(sc, 0) * 3), 40)
    print(f"  {i:2}위  {r.strategy_name:<22}  점수: {sc:6.2f}  {bar}")

best = results[0]
print(f"\n{'═'*110}")
print(f"  🏆 최우수 전략: {best.strategy_name}")
print(f"     총수익률: {best.total_return_pct:+.1f}%  │  연환산: {best.annualized_return_pct:+.1f}%  │  샤프: {best.sharpe_ratio:.2f}  │  소르티노: {best.sortino_ratio:.2f}")
print(f"     칼마: {best.calmar_ratio:.2f}  │  MDD: {best.max_drawdown_pct:.1f}%  │  승률: {best.win_rate_pct:.1f}%  │  P.F: {best.profit_factor:.2f}  │  거래수: {best.total_trades}")
print(f"{'═'*110}")

# 최근 체결 내역
if best.trades:
    print(f"\n  📋 {best.strategy_name} 최근 매매 10건:")
    print(f"  {'날짜':>12}  {'종목':>6}  {'구분':>4}  {'수량':>6}  {'진입가':>10}  {'청산가':>10}  {'손익':>12}  이유")
    print(f"  {'─'*95}")
    for t in best.trades[-10:]:
        pnl_str = f"{t.pnl:+,.0f}원" if t.pnl != 0 else "—"
        print(f"  {t.date:>12}  {t.ticker:>6}  {'매도':>4}  {t.qty:>6}  {t.entry_price:>10.2f}  {t.exit_price:>10.2f}  {pnl_str:>12}  {t.reason}")

# ── 결과 저장 ──────────────────────────────────────────────
from pathlib import Path
out = Path(__file__).parent.parent / "data"
out.mkdir(exist_ok=True)

save_data = {
    "run_at":          datetime.now().isoformat(),
    "period":          {"start": START_DATE, "end": END_DATE},
    "tickers":         WATCHLIST,
    "buy_hold_avg_pct": round(bnh_avg, 2),
    "best_strategy":   best.strategy_name,
    "best_score":      round(composite_score(best), 3),
    "best_metrics": {
        "total_return_pct":      best.total_return_pct,
        "annualized_return_pct": best.annualized_return_pct,
        "sharpe_ratio":          best.sharpe_ratio,
        "sortino_ratio":         best.sortino_ratio,
        "calmar_ratio":          best.calmar_ratio,
        "max_drawdown_pct":      best.max_drawdown_pct,
        "win_rate_pct":          best.win_rate_pct,
        "profit_factor":         best.profit_factor,
        "total_trades":          best.total_trades,
        "final_value":           best.final_value,
    },
    "all_results": [
        {
            "rank": i + 1,
            "strategy": r.strategy_name,
            "score": round(composite_score(r), 3),
            **{k: getattr(r, k) for k in [
                "total_return_pct", "annualized_return_pct",
                "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                "max_drawdown_pct", "win_rate_pct", "profit_factor",
                "total_trades", "final_value",
            ]},
        }
        for i, r in enumerate(results)
    ],
    "equity_curve": best.equity_curve,
    "recent_trades": [
        {"date": t.date, "ticker": t.ticker, "qty": t.qty,
         "entry": round(t.entry_price, 2), "exit": round(t.exit_price, 2),
         "pnl": round(t.pnl, 0), "reason": t.reason}
        for t in best.trades[-50:]
    ],
}

result_file = out / "backtest_result.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2)

print(f"\n💾 결과 저장 완료: {result_file}")
print(f"✅ {len(STRATEGIES)}개 전략 백테스트 완료 — '{best.strategy_name}' 채택\n")
