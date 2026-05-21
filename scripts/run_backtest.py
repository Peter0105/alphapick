"""
AlphaPick 백테스트 실행 스크립트
사용법: py scripts/run_backtest.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from services.strategy import STRATEGIES
from services.backtester import run_backtest, BacktestResult

# ── 설정 ──────────────────────────────────────────────────
WATCHLIST  = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
START_DATE = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
END_DATE   = datetime.today().strftime("%Y-%m-%d")

print("=" * 60)
print("  AlphaPick 백테스트 엔진")
print(f"  기간: {START_DATE} ~ {END_DATE}  (약 3년)")
print(f"  종목: {', '.join(WATCHLIST)}")
print("=" * 60)

# ── 데이터 다운로드 ────────────────────────────────────────
print("\n📥 과거 데이터 다운로드 중...")
price_data: dict[str, pd.DataFrame] = {}

for ticker in WATCHLIST:
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"  ⚠ {ticker}: 데이터 없음")
            continue
        # yfinance 최신 버전: MultiIndex 컬럼 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        price_data[ticker] = df
        print(f"  ✓ {ticker}: {len(df)}일치")
    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

if not price_data:
    print("❌ 데이터 없음 — 인터넷 연결 확인")
    sys.exit(1)

# ── 백테스트 실행 ──────────────────────────────────────────
print("\n🔄 전략별 백테스트 실행 중...\n")
results: list[BacktestResult] = []

for name, fn in STRATEGIES.items():
    try:
        r = run_backtest(price_data, fn, name)
        results.append(r)
        print(r.summary())
    except Exception as e:
        print(f"[{name}] 오류: {e}\n")

if not results:
    print("❌ 모든 전략 실패")
    sys.exit(1)

# ── 최우수 전략 선정 ───────────────────────────────────────
# 샤프지수 최우선, MDD 패널티 적용한 종합 점수
def score(r: BacktestResult) -> float:
    return r.sharpe_ratio * 0.5 + r.total_return_pct * 0.3 + r.win_rate_pct * 0.2 - abs(r.max_drawdown_pct) * 0.1

best = max(results, key=score)

print("=" * 60)
print(f"🏆 최우수 전략: {best.strategy_name}")
print(f"   → 종합점수: {score(best):.2f}")
print("=" * 60)

# ── 최우수 전략 결과 저장 ──────────────────────────────────
import json
from pathlib import Path

out_dir = Path(__file__).parent.parent / "data"
out_dir.mkdir(exist_ok=True)

result_file = out_dir / "backtest_result.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump({
        "best_strategy": best.strategy_name,
        "score": round(score(best), 2),
        "total_return_pct":       best.total_return_pct,
        "annualized_return_pct":  best.annualized_return_pct,
        "sharpe_ratio":           best.sharpe_ratio,
        "max_drawdown_pct":       best.max_drawdown_pct,
        "win_rate_pct":           best.win_rate_pct,
        "total_trades":           best.total_trades,
        "final_value":            best.final_value,
        "run_at":                 datetime.now().isoformat(),
        "all_results": [
            {
                "strategy": r.strategy_name,
                "total_return_pct":      r.total_return_pct,
                "annualized_return_pct": r.annualized_return_pct,
                "sharpe_ratio":          r.sharpe_ratio,
                "max_drawdown_pct":      r.max_drawdown_pct,
                "win_rate_pct":          r.win_rate_pct,
                "total_trades":          r.total_trades,
            }
            for r in results
        ],
        "equity_curve": best.equity_curve,
        "recent_trades": [
            {"date": t.date, "ticker": t.ticker, "action": t.action,
             "qty": t.qty, "price": round(t.price, 2), "reason": t.reason}
            for t in best.trades[-30:]
        ],
    }, f, ensure_ascii=False, indent=2)

print(f"\n💾 결과 저장: {result_file}")
print("\n✅ 백테스트 완료! 이 전략을 모의 트레이딩에 적용합니다.")
