// ── TAB SWITCHING ──────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
  event.target.classList.add('active');
  if (name === 'trading') loadPortfolio();
}

function quickSearch(ticker) {
  document.getElementById('tickerInput').value = ticker;
  runAnalysis();
}

// ── ANALYSIS ───────────────────────────────────────
async function runAnalysis() {
  const ticker = document.getElementById('tickerInput').value.trim();
  if (!ticker) return;

  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('analysis-result').classList.add('hidden');

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker }),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(`오류: ${err.detail}`);
      return;
    }
    const data = await res.json();
    renderAnalysis(data);
  } catch (e) {
    alert('서버 연결 실패: ' + e.message);
  } finally {
    document.getElementById('loading').classList.add('hidden');
  }
}

document.getElementById('tickerInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') runAnalysis();
});

function fmt(v, decimals = 2) {
  if (v == null) return '—';
  if (typeof v === 'number') {
    if (Math.abs(v) >= 1e12) return (v / 1e12).toFixed(1) + 'T';
    if (Math.abs(v) >= 1e9)  return (v / 1e9).toFixed(1) + 'B';
    if (Math.abs(v) >= 1e6)  return (v / 1e6).toFixed(1) + 'M';
    return v.toLocaleString('ko-KR', { maximumFractionDigits: decimals });
  }
  return v;
}

function renderAnalysis({ stock_data: s, analysis: a, from_cache }) {
  // header
  document.getElementById('stockName').textContent = s.name || s.ticker;
  document.getElementById('stockTicker').textContent = s.ticker;
  document.getElementById('currentPrice').textContent = fmt(s.current_price);
  document.getElementById('currency').textContent = s.currency || '';
  const ct = document.getElementById('cacheTag');
  from_cache ? ct.classList.remove('hidden') : ct.classList.add('hidden');

  // verdict
  const vb = document.getElementById('verdictBadge');
  vb.textContent = a.verdict;
  vb.className = `verdict-badge v-${a.verdict}`;
  document.getElementById('confidenceBar').style.width = `${a.confidence}%`;
  document.getElementById('confidenceVal').textContent = `${a.confidence}%`;

  // summary
  document.getElementById('summary').textContent = a.summary;
  const nsb = document.getElementById('newsSentiment');
  const sentMap = { positive: '긍정', neutral: '중립', negative: '부정' };
  nsb.textContent = sentMap[a.news_sentiment] || a.news_sentiment;
  nsb.className = `sentiment-badge s-${a.news_sentiment}`;
  document.getElementById('newsSummary').textContent = a.news_summary;

  // price range
  document.getElementById('fairLow').textContent = fmt(a.fair_price_low);
  document.getElementById('fairHigh').textContent = fmt(a.fair_price_high);
  document.getElementById('targetPrice').textContent = fmt(a.target_price);
  const up = a.upside_pct;
  const upEl = document.getElementById('upsidePct');
  upEl.textContent = up != null ? `${up > 0 ? '+' : ''}${up.toFixed(1)}%` : '—';
  upEl.style.color = up > 0 ? 'var(--green)' : up < 0 ? 'var(--red)' : 'var(--yellow)';
  document.getElementById('analystTarget').textContent =
    s.analyst_target ? `${fmt(s.analyst_target)} (${fmt(s.analyst_low)} ~ ${fmt(s.analyst_high)})` : '—';

  // bull / bear
  const bull = document.getElementById('bullCase');
  bull.innerHTML = (a.bull_case || []).map(x => `<li>${x}</li>`).join('');
  const bear = document.getElementById('bearCase');
  bear.innerHTML = (a.bear_case || []).map(x => `<li>${x}</li>`).join('');

  // analysis texts
  document.getElementById('valuationAnalysis').textContent = a.valuation_analysis;
  document.getElementById('technicalAnalysis').textContent = a.technical_analysis;

  // metrics
  const metrics = [
    ['현재가',       fmt(s.current_price)],
    ['PER (TTM)',   fmt(s.pe_ratio)],
    ['Forward PER', fmt(s.forward_pe)],
    ['EPS',         fmt(s.eps)],
    ['시가총액',    fmt(s.market_cap)],
    ['매출',        fmt(s.revenue)],
    ['이익률',      s.profit_margin ? (s.profit_margin * 100).toFixed(1) + '%' : '—'],
    ['52주 최고',   fmt(s.week_52_high)],
    ['52주 최저',   fmt(s.week_52_low)],
    ['50일 MA',    fmt(s.ma_50)],
    ['200일 MA',   fmt(s.ma_200)],
    ['베타',        fmt(s.beta)],
    ['배당수익률',  s.dividend_yield ? (s.dividend_yield * 100).toFixed(2) + '%' : '—'],
    ['섹터',        s.sector || '—'],
  ];
  document.getElementById('metricsGrid').innerHTML = metrics.map(([l, v]) =>
    `<div class="metric-item"><div class="metric-label">${l}</div><div class="metric-value">${v}</div></div>`
  ).join('');

  // chart
  renderChart(s.chart_data || []);

  // news
  const nl = document.getElementById('newsList');
  nl.innerHTML = (s.news || []).map(n =>
    `<li><a href="${n.link || '#'}" target="_blank">${n.title || '뉴스 없음'}</a></li>`
  ).join('') || '<li style="color:var(--text-muted)">뉴스 없음</li>';

  document.getElementById('analysis-result').classList.remove('hidden');
  document.getElementById('analysis-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderChart(chartData) {
  const el = document.getElementById('chart');
  const chart = echarts.init(el, 'dark');

  const dates = chartData.map(d => d.date);
  const ohlc  = chartData.map(d => [d.open, d.close, d.low, d.high]);
  const vols  = chartData.map(d => d.volume);

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    grid: [
      { left: 60, right: 20, top: 20, bottom: 120 },
      { left: 60, right: 20, top: '72%', bottom: 40 },
    ],
    xAxis: [
      { type: 'category', data: dates, gridIndex: 0, axisLabel: { show: false } },
      { type: 'category', data: dates, gridIndex: 1, axisLabel: { color: '#6b7a99', fontSize: 11 } },
    ],
    yAxis: [
      { gridIndex: 0, axisLabel: { color: '#6b7a99', fontSize: 11 }, splitLine: { lineStyle: { color: '#252c3d' } } },
      { gridIndex: 1, axisLabel: { color: '#6b7a99', fontSize: 10 }, splitLine: { lineStyle: { color: '#252c3d' } } },
    ],
    series: [
      {
        type: 'candlestick', data: ohlc, xAxisIndex: 0, yAxisIndex: 0,
        itemStyle: { color: '#00e676', color0: '#ff5252', borderColor: '#00e676', borderColor0: '#ff5252' },
      },
      {
        type: 'bar', data: vols, xAxisIndex: 1, yAxisIndex: 1,
        itemStyle: { color: '#448aff40' },
      },
    ],
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1], start: 0, end: 100 }],
  });

  window.addEventListener('resize', () => chart.resize());
}

// ── PORTFOLIO ──────────────────────────────────────
async function loadPortfolio() {
  try {
    const res = await fetch('/api/portfolio');
    if (!res.ok) throw new Error(res.status);
    const p = await res.json();
    renderPortfolio(p);
  } catch (e) {
    // 실패해도 UI는 유지, 에러 메시지만 업데이트
    const el = document.getElementById('portStats');
    if (el) el.innerHTML =
      '<div class="port-stat"><div class="port-stat-label">상태</div>'
      + '<div class="port-stat-value" style="color:var(--yellow);font-size:13px">서버 연결 중...</div></div>';
    // 5초 후 재시도
    setTimeout(loadPortfolio, 5000);
  }
}

function renderPortfolio(p) {
  const retColor = p.return_pct >= 0 ? 'var(--green)' : 'var(--red)';
  document.getElementById('portStats').innerHTML = `
    <div class="port-stat">
      <div class="port-stat-label">총 평가금액</div>
      <div class="port-stat-value">${(p.total_value || 0).toLocaleString('ko-KR')}원</div>
    </div>
    <div class="port-stat">
      <div class="port-stat-label">보유 현금</div>
      <div class="port-stat-value">${(p.cash || 0).toLocaleString('ko-KR')}원</div>
    </div>
    <div class="port-stat">
      <div class="port-stat-label">누적 수익률</div>
      <div class="port-stat-value" style="color:${retColor}">${p.return_pct >= 0 ? '+' : ''}${(p.return_pct || 0).toFixed(2)}%</div>
    </div>
  `;

  // positions
  const pos = p.positions || {};
  const pb = document.getElementById('positionBody');
  if (Object.keys(pos).length === 0) {
    pb.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted);text-align:center;padding:20px">보유 종목 없음</td></tr>';
  } else {
    pb.innerHTML = Object.entries(pos).map(([ticker, d]) => {
      const pnl = ((d.current_price - d.avg_price) / d.avg_price * 100).toFixed(2);
      const cls = pnl >= 0 ? 'pos-profit' : 'pos-loss';
      return `<tr>
        <td><strong>${ticker}</strong></td>
        <td>${d.quantity}</td>
        <td>${fmt(d.avg_price)}</td>
        <td>${fmt(d.current_price)}</td>
        <td class="${cls}">${pnl >= 0 ? '+' : ''}${pnl}%</td>
      </tr>`;
    }).join('');
  }

  // history
  const hist = p.trade_history || [];
  const hb = document.getElementById('historyBody');
  if (hist.length === 0) {
    hb.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted);text-align:center;padding:20px">매매 내역 없음</td></tr>';
  } else {
    hb.innerHTML = [...hist].reverse().slice(0, 20).map(t => `
      <tr>
        <td style="color:var(--text-muted);font-size:12px">${t.timestamp}</td>
        <td><strong>${t.ticker}</strong></td>
        <td><span class="tag-${t.action === 'BUY' ? 'buy' : 'sell'}">${t.action === 'BUY' ? '매수' : '매도'}</span></td>
        <td>${t.quantity}</td>
        <td>${fmt(t.price)}</td>
      </tr>`).join('');
  }
}

async function runAutoTrade() {
  const btn = document.getElementById('autoBtn');
  btn.disabled = true;
  document.getElementById('trade-loading').classList.remove('hidden');
  document.getElementById('last-decision').classList.add('hidden');

  try {
    const res = await fetch('/api/trade/auto', { method: 'POST' });
    const data = await res.json();
    renderDecision(data);
    renderPortfolio(data.portfolio);
  } catch (e) {
    alert('자동매매 오류: ' + e.message);
  } finally {
    btn.disabled = false;
    document.getElementById('trade-loading').classList.add('hidden');
  }
}

function renderDecision({ decision, executed }) {
  document.getElementById('marketView').textContent = decision.market_view || '';
  document.getElementById('strategyView').textContent = '전략: ' + (decision.strategy || '');

  const et = document.getElementById('executedTrades');
  if (executed.length === 0) {
    et.innerHTML = '<div class="executed-item" style="color:var(--text-muted)">이번 분석에서는 매매 없음 (HOLD)</div>';
  } else {
    et.innerHTML = executed.map(t => `
      <div class="executed-item">
        <span class="tag-${t.action === 'BUY' ? 'buy' : 'sell'}">${t.action === 'BUY' ? '매수' : '매도'}</span>
        <strong style="margin:0 8px">${t.ticker}</strong>
        ${t.quantity}주 @ ${fmt(t.price)}
        <span style="color:var(--text-muted);margin-left:10px;font-size:12px">${t.reason}</span>
        ${t.result.success === false ? `<span style="color:var(--red)"> ✗ ${t.result.message}</span>` : ''}
      </div>`).join('');
  }
  document.getElementById('last-decision').classList.remove('hidden');
}

async function resetPortfolio() {
  if (!confirm('포트폴리오를 초기화 하시겠습니까? (1,000만원으로 리셋)')) return;
  await fetch('/api/portfolio/reset', { method: 'POST' });
  loadPortfolio();
  document.getElementById('last-decision').classList.add('hidden');
}

// 초기 로드 — 트레이딩 탭 기본 뼈대 먼저 그리고 데이터 로드
renderPortfolio({
  cash: 0, initial_cash: 10000000, total_value: 0,
  return_pct: 0, positions: {}, trade_history: []
});
loadPortfolio();
