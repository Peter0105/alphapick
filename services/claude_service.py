import anthropic
import os
import json
from pathlib import Path

def _read_api_key() -> str:
    # 1) 환경변수 우선
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    # 2) .env 파일 직접 파싱
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""

def _get_client():
    return anthropic.Anthropic(api_key=_read_api_key())


def analyze_stock(stock_data: dict) -> dict:
    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:,.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    news_text = "\n".join(
        f"- {n['title']}" for n in stock_data.get("news", [])
    ) or "없음"

    prompt = f"""당신은 월가 출신 수석 애널리스트입니다. 아래 데이터를 바탕으로 철저한 투자 분석을 해주세요.

=== 종목 정보 ===
종목명: {stock_data.get('name')}  |  티커: {stock_data.get('ticker')}
현재가: {fmt(stock_data.get('current_price'))} {stock_data.get('currency')}
시가총액: {fmt(stock_data.get('market_cap'))}  |  섹터: {stock_data.get('sector')}  |  업종: {stock_data.get('industry')}

=== 밸류에이션 ===
PER(TTM): {fmt(stock_data.get('pe_ratio'))}  |  Forward PER: {fmt(stock_data.get('forward_pe'))}
EPS: {fmt(stock_data.get('eps'))}  |  매출: {fmt(stock_data.get('revenue'))}
이익률: {fmt(stock_data.get('profit_margin'))}  |  배당수익률: {fmt(stock_data.get('dividend_yield'))}

=== 기술적 지표 ===
52주 최고: {fmt(stock_data.get('week_52_high'))}  |  52주 최저: {fmt(stock_data.get('week_52_low'))}
50일 이동평균: {fmt(stock_data.get('ma_50'))}  |  200일 이동평균: {fmt(stock_data.get('ma_200'))}
베타: {fmt(stock_data.get('beta'))}  |  거래량: {fmt(stock_data.get('volume'))}  |  평균거래량: {fmt(stock_data.get('avg_volume'))}

=== 애널리스트 컨센서스 ===
목표가(평균): {fmt(stock_data.get('analyst_target'))}  |  범위: {fmt(stock_data.get('analyst_low'))} ~ {fmt(stock_data.get('analyst_high'))}

=== 최신 뉴스 ===
{news_text}

위 데이터를 종합하여 반드시 아래 JSON 형식으로만 답변하세요. JSON 외 다른 텍스트 없이:
{{
  "verdict": "강력매수 | 매수 | 중립 | 매도 | 강력매도 중 하나",
  "confidence": 정수 1~100,
  "fair_price_low": 적정가 하단 숫자,
  "fair_price_high": 적정가 상단 숫자,
  "target_price": 목표가 숫자,
  "upside_pct": 현재가 대비 목표가 상승여력 퍼센트 숫자,
  "summary": "핵심 요약 2~3문장",
  "bull_case": ["매수 근거 1", "매수 근거 2", "매수 근거 3"],
  "bear_case": ["리스크 1", "리스크 2", "리스크 3"],
  "valuation_analysis": "밸류에이션 상세 분석 2~3문장",
  "technical_analysis": "기술적 분석 2~3문장 (이동평균, 52주 고저 대비 포지션 등)",
  "news_sentiment": "positive | neutral | negative",
  "news_summary": "최신 뉴스 기반 시장 분위기 1~2문장"
}}"""

    response = _get_client().messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def make_trading_decision(portfolio: dict, watchlist_data: list) -> dict:
    prompt = f"""당신은 독립적으로 운용되는 AI 퀀트 트레이더입니다.
현재 포트폴리오 상황과 관심 종목 데이터를 보고 구체적인 매매 결정을 내려주세요.

=== 현재 포트폴리오 ===
보유 현금: {portfolio.get('cash', 0):,.0f}원
총 평가금액: {portfolio.get('total_value', 0):,.0f}원
누적 수익률: {portfolio.get('return_pct', 0):.2f}%
보유 종목:
{json.dumps(portfolio.get('positions', {}), ensure_ascii=False, indent=2)}

=== 관심 종목 현황 ===
{json.dumps(watchlist_data, ensure_ascii=False, indent=2)}

매매 결정을 아래 JSON 형식으로만 반환하세요:
{{
  "actions": [
    {{
      "action": "BUY 또는 SELL 또는 HOLD",
      "ticker": "종목코드",
      "quantity": 수량(정수),
      "price": 현재가(숫자),
      "reason": "결정 이유 한 문장"
    }}
  ],
  "market_view": "현재 시장 전반에 대한 한 문장 의견",
  "strategy": "현재 운용 전략 한 문장"
}}

규칙: 현금 비율 최소 20% 유지. HOLD는 actions에 포함하지 않아도 됨."""

    response = _get_client().messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
