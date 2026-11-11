/* ---------------------------------------------------------------------
 * desk.js: the Verifiable Research Desk site.
 * Everything shown is read from data/desk-data.js, which the desk pipeline
 * writes. The one thing computed here is the ledger check: the page fetches
 * the raw ledger files and re-walks their SHA-256 chains in the browser.
 * ------------------------------------------------------------------- */
(function () {
  'use strict';
  const D = window.DESK_DATA;
  const $ = (id) => document.getElementById(id);
  const TABS = ['overview', 'portfolio', 'valuation', 'ledger', 'method'];
  const REPO = 'https://github.com/Nnd1106/verifiable-research-desk';
  const state = { tab: 'overview', frontier: 'policy', subject: 'policy_orp', chain: null };
  const charts = {};

  /* ---------------- palette (validated dark categorical slots, surface #0f131c) ---------------- */
  const COL = { s1: '#3987e5', s2: '#d95926', s3: '#199e70', s4: '#c98500', s5: '#d55181', neg: '#e66767',
    neutral: '#6b7a8f', ink: '#e6edf7', ink2: '#9fb0c3', muted: '#6b7a8f', grid: 'rgba(255,255,255,0.06)', axis: 'rgba(255,255,255,0.16)' };
  const MONO = '"SFMono-Regular", Consolas, "Cascadia Mono", "Liberation Mono", monospace';
  const SECTOR = { it: ['IT Services', COL.s1], fmcg: ['Consumer Staples', COL.s2], pharma: ['Pharmaceuticals', COL.s3], auto: ['Automobiles', COL.s4], cement: ['Cement', COL.s5] };
  const SUBJECT = { policy_orp: 'Policy tangency portfolio', textbook_orp: 'Textbook tangency portfolio', benchmark: 'Nifty 50 benchmark' };
  const MODEL = { hist: 'Historical simulation', normal: 'Parametric normal', ewma: 'EWMA (RiskMetrics)', rolling60: '60-day rolling', garch: 'GARCH(1,1)', dcf_baseline: 'DCF baseline' };
  const MODEL_COL = { hist: COL.s1, normal: COL.s2, ewma: COL.s3 };

  /* ---------------- formatting ---------------- */
  const ok = (x) => x !== null && x !== undefined && Number.isFinite(x);
  const nf = (dp) => new Intl.NumberFormat('en-IN', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  const NF = [nf(0), nf(1), nf(2), nf(3), nf(4)];
  const F = {
    num: (x, dp = 0) => (ok(x) ? NF[dp].format(x) : '–'),
    pct: (x, dp = 1) => (ok(x) ? (x * 100).toFixed(dp) + '%' : '–'),
    spct: (x, dp = 1) => (ok(x) ? (x >= 0 ? '+' : '−') + Math.abs(x * 100).toFixed(dp) + '%' : '–'),
    px: (x) => (ok(x) ? '₹' + NF[Math.abs(x) >= 1000 ? 0 : 1].format(x) : '–'),
    cr: (x) => (ok(x) ? (Math.abs(x) >= 1e5 ? `₹${NF[2].format(x / 1e5)} L cr` : `₹${NF[0].format(x)} cr`) : '–'),
    sci: (x) => (!ok(x) ? '–' : x === 0 ? '0' : x < 1e-3 ? x.toExponential(1) : x.toPrecision(2)),
    p: (x) => (!ok(x) ? 'n/a' : x < 0.001 ? '<0.001' : x.toFixed(3)),
    date: (iso) => (iso ? new Date(iso.length === 10 ? iso + 'T00:00:00Z' : iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }) : '–'),
    stamp: (iso) => (iso ? new Date(iso).toLocaleString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }) + ' UTC' : '–'),
    hash: (h) => (h ? h.slice(0, 12) + '…' + h.slice(-8) : '–'),
  };
  const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  const status = (kind, label) => `<span class="status ${kind}"><span class="ico" aria-hidden="true">${{ pass: '✓', warn: '!', fail: '×' }[kind]}</span>${label}</span>`;
  const statBox = (label, value, sub, cls = '') => `<div class="stat-box ${cls}"><div class="stat-label">${label}</div><div class="stat-value">${value}</div>${sub ? `<div class="stat-sub">${sub}</div>` : ''}</div>`;
  const dclass = (x) => (!ok(x) ? '' : x >= 0 ? 'delta-up' : 'delta-down');

  /* ---------------- charts ---------------- */
  function chart(id, config) {
    const el = $(id);
    if (!el || !window.Chart) return null;
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(el, config);
    return charts[id];
  }
  const axis = (o = {}) => Object.assign({ grid: { color: COL.grid, drawTicks: false }, border: { color: COL.axis }, ticks: { color: COL.muted, padding: 6, font: { family: MONO, size: 10 } } }, o);
  if (window.Chart) {
    Chart.defaults.color = COL.ink2;
    Chart.defaults.font.family = '-apple-system, "Segoe UI", system-ui, sans-serif';
    Chart.defaults.font.size = 11;
    Chart.defaults.animation = false;
    Chart.defaults.plugins.legend.display = false;
    Object.assign(Chart.defaults.plugins.tooltip, { backgroundColor: '#0b0e15', borderColor: 'rgba(0,240,255,0.25)', borderWidth: 1, titleColor: COL.ink, bodyColor: COL.ink2, padding: 10, displayColors: false, bodyFont: { family: MONO, size: 11 } });
    Chart.register({
      id: 'valueLabels',
      afterDatasetsDraw(c, _a, opts) {
        if (!opts || !opts.fmt) return;
        const ctx = c.ctx;
        ctx.save();
        ctx.font = `600 10.5px ${MONO}`;
        ctx.fillStyle = COL.ink;
        const horizontal = c.options.indexAxis === 'y';
        c.data.datasets.forEach((ds, di) => {
          const meta = c.getDatasetMeta(di);
          if (meta.type !== 'bar') return;
          meta.data.forEach((bar, i) => {
            const txt = opts.fmt(ds.data[i], i, ds);
            if (!txt) return;
            if (horizontal) {
              const right = ds.data[i] >= 0;
              ctx.textAlign = right ? 'left' : 'right';
              ctx.textBaseline = 'middle';
              ctx.fillText(txt, bar.x + (right ? 6 : -6), bar.y);
            } else { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(txt, bar.x, Math.min(bar.y, bar.base) - 4); }
          });
        });
        ctx.restore();
      },
    });
    Chart.register({
      id: 'refLine',
      afterDatasetsDraw(c, _a, opts) {
        if (!opts || !ok(opts.value)) return;
        const { ctx, chartArea: a, scales } = c;
        const y = scales.y.getPixelForValue(opts.value);
        if (y < a.top || y > a.bottom) return;
        ctx.save();
        ctx.strokeStyle = COL.ink; ctx.fillStyle = COL.ink; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(a.left, y); ctx.lineTo(a.right, y); ctx.stroke();
        ctx.setLineDash([]); ctx.font = `600 10.5px ${MONO}`; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
        ctx.fillText(opts.label || '', a.right - 4, y - 4);
        ctx.restore();
      },
    });
  }

  /* ---------------- shared pieces ---------------- */
  function checksTable(checks) {
    const groups = [];
    checks.forEach((c) => { if (!groups.includes(c.group)) groups.push(c.group); });
    return `<div class="table-scroll"><table class="data-table"><thead><tr><th>Check</th><th style="text-align:left">Result</th><th>Difference</th><th>Tolerance</th><th class="wrap">What was compared</th></tr></thead><tbody>
      ${groups.map((g) => `<tr class="group"><td colspan="5">${esc(g)}</td></tr>` + checks.filter((c) => c.group === g).map((c) =>
        `<tr><td style="white-space:normal">${esc(c.title)}</td><td style="text-align:left">${c.ok ? status('pass', 'Agrees') : status('fail', 'Differs')}</td><td>${F.sci(c.diff)}</td><td>${F.sci(c.tolerance)}</td><td class="wrap">${esc(c.detail)}</td></tr>`).join('')).join('')}
    </tbody></table></div>`;
  }
  const passCount = (checks) => `${checks.filter((c) => c.ok).length} / ${checks.length}`;

  /* ---------------- ledger verification in the browser ---------------- */
  async function sha256hex(text) {
    const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
    return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, '0')).join('');
  }
  async function verifyChain(file) {
    const res = await fetch(file, { cache: 'no-store' });
    if (!res.ok) throw new Error(`could not load ${file}`);
    const lines = (await res.text()).split('\n').filter((l) => l.trim());
    let prev = '0'.repeat(64);
    for (let i = 0; i < lines.length; i++) {
      const e = JSON.parse(lines[i]);
      if (e.prev !== prev || e.hash !== await sha256hex(prev + '\n' + e.payload) || JSON.parse(e.payload).seq !== i) return { ok: false, at: i, n: lines.length, head: prev };
      prev = e.hash;
    }
    return { ok: true, n: lines.length, head: prev };
  }
  async function verifyLedgers() {
    state.chain = { pending: true };
    paintChain();
    try {
      const [f, o] = await Promise.all([verifyChain(D.ledger.forecasts.file), verifyChain(D.ledger.outcomes.file)]);
      state.chain = { f, o, ok: f.ok && o.ok && f.head === D.ledger.forecasts.head && o.head === D.ledger.outcomes.head };
    } catch (err) {
      state.chain = { error: err.message };
    }
    paintChain();
    renderKpis();
  }
  function chainStatus() {
    const c = state.chain;
    if (!c || c.pending) return status('warn', 'Verifying…');
    if (c.error) return status('warn', 'Not checked here');
    return c.ok ? status('pass', 'Chain intact') : status('fail', 'Chain broken');
  }
  function paintChain() {
    const el = $('chain-box');
    if (!el) return;
    const c = state.chain, L = D.ledger;
    const detail = !c || c.pending ? 'Fetching the raw ledger files and recomputing every SHA-256 link…'
      : c.error ? `The ledger files could not be fetched from this page (${esc(c.error)}). The pipeline's own check is shown.`
      : c.ok ? `Your browser just recomputed ${c.f.n + c.o.n} SHA-256 links from the raw ledger files. Every link holds, and the chain heads match the ones this run published.`
      : `The chain does not verify in your browser (first break at entry ${c.f.ok ? c.o.at : c.f.at}). Do not trust the record until this is explained.`;
    el.innerHTML = `
      <div class="stat-row">
        ${statBox('Checked in your browser', chainStatus(), 'SHA-256 via the Web Crypto API', 'lead')}
        ${statBox('Forecasts sealed', F.num(L.forecasts.entries), `since ${F.date(L.started)}`)}
        ${statBox('Outcomes scored', F.num(L.outcomes.entries), `${L.pending_risk_forecasts} risk forecasts awaiting their session`)}
        ${statBox('Checked by the pipeline', L.forecasts.verified && L.outcomes.verified ? status('pass', 'Chain intact') : status('fail', 'Chain broken'), F.stamp(D.meta.generated_utc))}
      </div>
      <p class="card-desc">${detail}</p>
      <div class="table-scroll"><table class="data-table loose"><tbody>
        <tr><td>Forecast chain head</td><td class="hash">${esc(L.forecasts.head)}</td></tr>
        <tr><td>Outcome chain head</td><td class="hash">${esc(L.outcomes.head)}</td></tr>
      </tbody></table></div>`;
  }

  /* ---------------- KPI strip ---------------- */
  function renderKpis() {
    const s = D.summary, L = D.ledger;
    const tiles = [
      ['Independent checks', `${s.checks_passed} / ${s.checks_total}`, s.checks_failed ? `${s.checks_failed} differ` : 'all agree'],
      ['Portfolio lab', passCount(D.portfolio.checks), `data as of ${F.date(D.portfolio.as_of)}`],
      ['Valuation lab', passCount(D.valuation.checks), `data as of ${F.date(D.valuation.valuation_date)}`],
      ['Forecasts sealed', F.num(L.forecasts.entries), `since ${F.date(L.started)}`],
      ['Outcomes scored', F.num(L.outcomes.entries), `${L.pending_risk_forecasts} pending`],
      ['Ledger chain', chainStatus(), 'verified in this browser'],
    ];
    $('kpi-strip').innerHTML = tiles.map(([l, v, sub]) => `<div class="kpi-tile"><div class="kpi-label">${l}</div><div class="kpi-value" style="font-size:17px">${v}</div><div class="kpi-sub">${sub}</div></div>`).join('');
  }

  /* ---------------- 01 overview ---------------- */
  function buildOverview(host) {
    const s = D.summary, L = D.ledger, P = D.sources.portfolio, V = D.sources.valuation;
    const bt = D.portfolio.var_backtest.policy_orp.models.hist['0.99'];
    const project = (src, checks, what) => `<a class="project-card" href="${esc(src.site)}" target="_blank" rel="noopener">
      <div class="name">${esc(src.name)} ↗</div><div class="what">${what}</div>
      <div class="state">${checks.every((c) => c.ok) ? status('pass', `${checks.length} of ${checks.length} checks agree`) : status('fail', `${checks.filter((c) => !c.ok).length} of ${checks.length} checks differ`)}</div></a>`;
    host.innerHTML = `
      <div class="hero">
        <div class="card">
          <h2 class="card-title">What this desk is for</h2>
          <div class="hero-figure">${s.checks_passed}<small> / ${s.checks_total} independent checks agree</small></div>
          <p class="hero-caption">
            Anyone can build a model that prints numbers. This desk asks the next question, the one a bank's model-validation team asks:
            <strong style="color:var(--text-primary)">are the numbers right, and do the forecasts hold up?</strong> It re-derives the results of my two analytics platforms from
            their published data alone, with separately written code and usually a different method, and it seals forecasts in a hash-chained ledger
            <em>before</em> their outcomes are known, then scores them.
          </p>
          <div class="btn-group" style="margin-top:14px">
            <button type="button" class="btn" data-go="portfolio">Portfolio validation →</button>
            <button type="button" class="btn" data-go="valuation">Valuation validation →</button>
            <button type="button" class="btn" data-go="ledger">Forecast ledger →</button>
          </div>
        </div>
        <div class="card">
          <h2 class="card-title">The projects under review</h2>
          <div class="stack">
            ${project(P, D.portfolio.checks, 'Multi-asset portfolio construction for an Indian investor: efficient frontier, VaR, attribution, rebalancing, goals.')}
            ${project(V, D.valuation.checks, 'Investment-banking valuation on Indian large caps: DCF, comps, LBO, M&amp;A, football field.')}
          </div>
        </div>
      </div>
      <div class="grid-3" style="margin-top:18px">
        <div class="card">
          <h2 class="card-title">1 · Replicate</h2>
          <p class="card-desc" style="margin:0">The portfolio lab's estimation is rebuilt from its published daily returns: weekly covariance matrix, sample means and the Bayes-Stein shrinkage. All agree within the rounding of the published inputs.</p>
        </div>
        <div class="card">
          <h2 class="card-title">2 · Challenge</h2>
          <p class="card-desc" style="margin:0">Results are reached again by a different route. The frontier is re-solved exactly by enumerating active constraints instead of iterating; every DCF is re-valued as invested capital plus discounted economic profit. Two methods, one answer.</p>
        </div>
        <div class="card">
          <h2 class="card-title">3 · Score outcomes</h2>
          <p class="card-desc" style="margin:0">Agreement proves the arithmetic, not the forecast. So forecasts are sealed daily and scored when reality arrives. The reconstructed backtest already shows a real weakness: the 99% VaR was breached ${bt.exceedances} times where ${F.num(bt.expected, 1)} were expected.</p>
        </div>
      </div>
      <div class="card" style="margin-top:18px">
        <h2 class="card-title">Exactly what was checked in this run</h2>
        <div class="table-scroll"><table class="data-table loose"><thead><tr><th>Source</th><th>Read from</th><th>File generated</th><th>SHA-256 of the bytes that were validated</th></tr></thead><tbody>
          ${[P, V].map((x) => `<tr><td>${esc(x.name)}</td><td>${x.origin === 'published' ? status('pass', 'Published file') : status('warn', 'Local checkout')}</td><td>${F.stamp(x.generated_utc)}</td><td class="hash">${esc(x.sha256)}</td></tr>`).join('')}
        </tbody></table></div>
        <p class="legend-note">The desk reads what the public sees: the JSON each project serves. The hash pins the exact bytes, so a later change to a project's file cannot be confused with what was validated here. "Local checkout" means that project had not been deployed yet when the desk ran.</p>
      </div>`;
    host.querySelectorAll('[data-go]').forEach((b) => b.addEventListener('click', () => go(b.dataset.go)));
  }

  /* ---------------- 02 portfolio ---------------- */
  function drawFrontier() {
    const f = D.portfolio.frontiers[state.frontier];
    const pts = (rows) => rows.map(([x, y]) => ({ x, y }));
    chart('pf-frontier', {
      type: 'scatter',
      data: { datasets: [
        { label: 'Random feasible portfolios', data: pts(f.cloud), backgroundColor: 'rgba(107,122,143,0.45)', pointRadius: 2, pointHoverRadius: 3 },
        { label: 'Published frontier', data: pts(f.published), showLine: true, borderColor: COL.s1, borderWidth: 2, pointRadius: 0, pointHitRadius: 6 },
        { label: 'Exact re-solution', data: pts(f.challenger.filter((_p, i) => i % 6 === 0)), backgroundColor: 'transparent', borderColor: COL.s2, borderWidth: 2, pointRadius: 5, pointStyle: 'circle' },
        { label: 'Minimum variance and tangency (published)', data: pts([f.mvp.published, f.orp.published]), backgroundColor: COL.ink, borderColor: '#0f131c', borderWidth: 2, pointRadius: 6, pointStyle: 'rectRot' },
      ] },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: axis({ title: { display: true, text: 'Volatility (annual)', color: COL.muted }, ticks: { color: COL.muted, callback: (v) => (v * 100).toFixed(0) + '%' } }),
          y: axis({ title: { display: true, text: 'Expected return (annual)', color: COL.muted }, ticks: { color: COL.muted, callback: (v) => (v * 100).toFixed(0) + '%' } }),
        },
        plugins: { tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: vol ${F.pct(ctx.parsed.x, 2)}, return ${F.pct(ctx.parsed.y, 2)}` } } },
      },
    });
    const names = D.portfolio.asset_names, keys = Object.keys(names);
    const wrow = (label, pub, chal) => `<tr><td>${label}</td>${keys.map((_k, i) => `<td>${F.pct(pub[i], 2)}<br><span style="color:var(--text-muted)">${F.pct(chal[i], 2)}</span></td>`).join('')}</tr>`;
    $('pf-weights').innerHTML = `<table class="data-table"><thead><tr><th>Weights: published / <span style="color:var(--text-muted)">exact</span></th>${keys.map((k) => `<th>${esc(names[k])}</th>`).join('')}</tr></thead><tbody>
      ${wrow('Minimum-variance portfolio', f.mvp.w_published, f.mvp.w_challenger)}${wrow('Tangency portfolio', f.orp.w_published, f.orp.w_challenger)}</tbody></table>
      <div class="legend-note">Tangency Sharpe ratio: published ${f.orp.sharpe_published.toFixed(5)}, exact ${f.orp.sharpe_challenger.toFixed(5)}. Model: ${esc(f.label)}.</div>`;
  }

  function buildPortfolio(host) {
    const P = D.portfolio, fnd = P.findings, sh = fnd.orp_realised_sharpe;
    const moving = fnd.mvp_weight_ranges.filter((r) => r.p95 - r.p05 > 0.005);
    host.innerHTML = `
      <p class="section-intro">
        The <a href="${esc(D.sources.portfolio.site)}" target="_blank" rel="noopener">Portfolio Construction Lab</a> publishes its inputs and its results. This page takes only that file and asks whether the
        results follow from the inputs. The estimation is <strong>replicated</strong> from daily returns; the optimiser is <strong>challenged</strong> by an exact method that shares no code and no algorithm
        with the lab's; the risk metrics are <strong>recomputed</strong> with separately written code. Tolerances sit just above the rounding of the published inputs (six decimals).
      </p>
      <div class="card">
        <h2 class="card-title">Checks: ${passCount(P.checks)} agree</h2>
        ${checksTable(P.checks)}
      </div>
      <div class="grid-2" style="margin-top:18px">
        <div class="card">
          <h2 class="card-title">The frontier, solved twice</h2>
          <div class="btn-group" style="margin-bottom:10px" role="group" aria-label="Model">
            <button type="button" class="btn" data-frontier="policy">Policy model</button>
            <button type="button" class="btn" data-frontier="textbook">Textbook model</button>
          </div>
          <div class="legend-row">
            <span><span class="swatch" style="background:${COL.s1}"></span>Published (iterative optimiser)</span>
            <span><span class="swatch" style="background:transparent;border:2px solid ${COL.s2};border-radius:50%"></span>Exact re-solution</span>
            <span><span class="swatch" style="background:${COL.neutral}"></span>Random feasible portfolios</span>
            <span><span class="swatch" style="background:${COL.ink};transform:rotate(45deg)"></span>Minimum variance, tangency</span>
          </div>
          <div class="chart-wrap tall"><canvas id="pf-frontier" role="img" aria-label="Efficient frontier as published and as re-solved exactly, over a cloud of random feasible portfolios"></canvas></div>
          <div class="legend-note">The orange rings are a different algorithm's answer to the same problem and sit on the blue line. No random portfolio lies above and to the left of it.</div>
        </div>
        <div class="stack">
          <div class="card">
            <h2 class="card-title">Optimal weights, both methods</h2>
            <div class="table-scroll" id="pf-weights"></div>
          </div>
          <div class="card">
            <h2 class="card-title">Finding: how much to trust the outputs</h2>
            <ul class="note-list">
              <li><strong>The realised Sharpe ratio is far less certain than it looks.</strong> The policy tangency portfolio earned a Sharpe of ${sh.point.toFixed(2)} over ${sh.years.toFixed(1)} years, but resampling those days gives a 90% range of ${sh.p05.toFixed(2)} to ${sh.p95.toFixed(2)}. Under seven years of data cannot distinguish a good strategy from a lucky one.</li>
              <li><strong>The minimum-variance weights are pinned by the allocation limits, not by the data.</strong> Across ${fnd.bootstrap_draws} resampled histories ${moving.length ? `only ${moving.map((r) => esc(P.asset_names[r.key])).join(' and ')} move${moving.length === 1 ? 's' : ''} at all` : 'no weight moves'}; the rest sit on a bound in every draw. The 5–50% ranges are doing the work the optimiser appears to do.</li>
              <li>These are observations about model risk, not errors: the lab's arithmetic is right. See the <a href="#ledger">forecast ledger</a> for how its VaR has held up out of sample.</li>
            </ul>
          </div>
        </div>
      </div>`;
    host.querySelectorAll('[data-frontier]').forEach((b) => {
      b.classList.toggle('active', b.dataset.frontier === state.frontier);
      b.addEventListener('click', () => { state.frontier = b.dataset.frontier; host.querySelectorAll('[data-frontier]').forEach((x) => x.classList.toggle('active', x === b)); drawFrontier(); });
    });
    drawFrontier();
  }

  /* ---------------- 03 valuation ---------------- */
  function buildValuation(host) {
    const V = D.valuation;
    const rows = V.rows.slice().sort((a, b) => b.upside - a.upside);
    const worst = Math.max(...V.rows.map((r) => r.ev_rel_diff));
    host.innerHTML = `
      <p class="section-intro">
        The <a href="${esc(D.sources.valuation.site)}" target="_blank" rel="noopener">Valuation &amp; Deal Lab</a> values 24 companies by discounted cash flow. Here each one is valued again by a different method:
        <strong>invested capital plus the present value of economic profit</strong> (operating profit after tax, less a charge for the capital employed). The two methods are algebraically equivalent, so two independently
        written implementations must land on the same enterprise value. They do, to ${F.sci(worst)} relative. The LBO schedules are audited line by line and the M&amp;A results are recomputed from a closed-form expression.
      </p>
      <div class="card">
        <h2 class="card-title">Checks: ${passCount(V.checks)} agree</h2>
        ${checksTable(V.checks)}
      </div>
      <div class="grid-2" style="margin-top:18px">
        <div class="card">
          <h2 class="card-title">DCF baseline against market price</h2>
          <div class="legend-row">${Object.entries(SECTOR).map(([, [n, c]]) => `<span><span class="swatch" style="background:${c}"></span>${n}</span>`).join('')}</div>
          <div class="chart-wrap" style="height:${60 + rows.length * 21}px"><canvas id="val-gap" role="img" aria-label="Gap between the DCF baseline value and the market price for each company"></canvas></div>
          <div class="legend-note">Gap = DCF value ÷ price − 1, as of ${F.date(V.valuation_date)}. These are the valuation calls the ledger seals each week and will score against later returns.</div>
        </div>
        <div class="stack">
          <div class="card">
            <h2 class="card-title">Model-risk observations</h2>
            <ul class="note-list">${V.observations.map((o) => `<li><strong>${esc(o.title)}.</strong> ${esc(o.text)}</li>`).join('')}</ul>
          </div>
          <div class="card">
            <h2 class="card-title">Two methods, one enterprise value</h2>
            <div class="table-scroll" style="max-height:420px;overflow-y:auto"><table class="data-table"><thead><tr><th>Company</th><th>DCF (lab)</th><th>Economic profit (desk)</th><th>Rel. difference</th><th>TV share</th></tr></thead><tbody>
              ${V.rows.map((r) => `<tr><td>${esc(r.ticker)}</td><td>${F.cr(r.ev_published)}</td><td>${F.cr(r.ev_challenger)}</td><td>${F.sci(r.ev_rel_diff)}</td><td>${F.pct(r.terminal_share, 0)}</td></tr>`).join('')}
            </tbody></table></div>
          </div>
        </div>
      </div>`;
    chart('val-gap', {
      type: 'bar',
      data: { labels: rows.map((r) => r.ticker), datasets: [{ data: rows.map((r) => r.upside), backgroundColor: rows.map((r) => SECTOR[r.sector][1]), borderRadius: 3, barThickness: 13 }] },
      options: {
        indexAxis: 'y', maintainAspectRatio: false, layout: { padding: { right: 44, left: 6 } },
        scales: { x: axis({ suggestedMin: -1.1, suggestedMax: 0.3, ticks: { color: COL.muted, callback: (v) => (v * 100).toFixed(0) + '%' } }), y: axis({ grid: { display: false }, ticks: { color: COL.ink2, font: { size: 10 }, autoSkip: false } }) },
        plugins: {
          valueLabels: { fmt: (v) => F.spct(v, 0) },
          tooltip: { callbacks: { title: (items) => rows[items[0].dataIndex].name, label: (ctx) => { const r = rows[ctx.dataIndex]; return [`${SECTOR[r.sector][0]}`, `Price ${F.px(r.price)} · DCF ${F.px(r.per_share_published)}`, `Gap ${F.spct(r.upside, 0)}`, `Price-implied growth ${r.implied_growth === null ? 'out of range' : F.pct(r.implied_growth)} (model ${F.pct(r.model_growth)})`]; } } },
        },
      },
    });
  }

  /* ---------------- 04 ledger ---------------- */
  function drawBacktest() {
    const bt = D.portfolio.var_backtest[state.subject];
    const models = Object.keys(bt.models);
    ['0.95', '0.99'].forEach((conf) => {
      const e = models.map((m) => bt.models[m][conf]);
      chart('bt-' + conf.replace('.', ''), {
        type: 'bar',
        data: { labels: models.map((m) => MODEL[m]), datasets: [{ data: e.map((x) => x.exceedances), backgroundColor: models.map((m) => MODEL_COL[m]), borderRadius: 3, maxBarThickness: 44 }] },
        options: {
          maintainAspectRatio: false, layout: { padding: { top: 18 } },
          scales: { x: axis({ grid: { display: false }, ticks: { color: COL.ink2, font: { size: 10 } } }), y: axis({ beginAtZero: true, suggestedMax: Math.max(...e.map((x) => x.exceedances), e[0].expected) * 1.2 }) },
          plugins: { valueLabels: { fmt: (v) => String(v) }, refLine: { value: e[0].expected, label: `Expected ${F.num(e[0].expected, 1)}` },
            tooltip: { callbacks: { label: (ctx) => { const x = e[ctx.dataIndex]; return [`${x.exceedances} breaches vs ${F.num(x.expected, 1)} expected`, `Kupiec p ${F.p(x.kupiec_p)}`, `Clustering p ${F.p(x.christoffersen_p)}`]; } } } },
        },
      });
    });
    const flag = (p) => (ok(p) && p < 0.05 ? status('warn', F.p(p)) : `<span class="status pass">${F.p(p)}</span>`);
    const best = (conf) => models.slice().sort((a, b) => bt.models[a][conf].pinball - bt.models[b][conf].pinball)[0];
    $('bt-table').innerHTML = `<table class="data-table"><thead><tr><th>Model</th><th>Level</th><th>Breaches</th><th>Expected</th><th>Rate</th><th style="text-align:left">Coverage (Kupiec p)</th><th style="text-align:left">Clustering (Christoffersen p)</th><th>Quantile loss ×10⁴</th><th style="text-align:left">Basel zone</th></tr></thead><tbody>
      ${models.map((m) => ['0.95', '0.99'].map((conf) => { const x = bt.models[m][conf]; const zone = x.basel_zone;
        return `<tr><td>${MODEL[m]}</td><td>${F.pct(x.confidence, 0)}</td><td>${x.exceedances}</td><td>${F.num(x.expected, 1)}</td><td>${F.pct(x.rate, 2)}</td><td style="text-align:left">${flag(x.kupiec_p)}</td><td style="text-align:left">${flag(x.christoffersen_p)}</td><td>${(x.pinball * 1e4).toFixed(3)}${best(conf) === m ? ' ★' : ''}</td><td style="text-align:left">${zone ? status(zone === 'green' ? 'pass' : zone === 'yellow' ? 'warn' : 'fail', zone[0].toUpperCase() + zone.slice(1)) : '–'}</td></tr>`; }).join('')).join('')}
    </tbody></table>
    <div class="legend-note">${F.num(bt.n)} one-day forecasts, ${F.date(bt.start)} to ${F.date(bt.end)}, each made with data up to the previous day only. A p-value below 0.05 (marked “!”) rejects the model on that test. ★ marks the lowest quantile loss, a proper scoring rule: the true quantile minimises it. Basel zone: breaches of the 99% VaR in the last 250 sessions (green up to 4, yellow 5 to 9, red 10 or more).</div>`;
  }

  function buildLedger(host) {
    const L = D.ledger, latest = L.latest;
    const riskAsOf = (latest.find((r) => r.kind === 'var') || {}).as_of;
    const made = (latest.find((r) => r.kind === 'var') || {}).made_utc;
    const subjects = Object.keys(D.portfolio.var_backtest);
    const cell = (s, m, c) => { const r = latest.find((x) => x.kind === 'var' && x.subject === s && x.model === m && x.confidence === c); return r ? F.pct(r.value, 2) : '–'; };
    const vol = latest.filter((r) => r.kind === 'vol');
    const calls = latest.filter((r) => r.kind === 'valuation').sort((a, b) => b.value - a.value);
    const sc = L.scorecards;
    const live = sc.var.length || sc.vol.length || sc.valuation.length;
    const vb = D.portfolio.vol_backtest;
    host.innerHTML = `
      <p class="section-intro">
        A model can agree with itself perfectly and still be wrong about the world. The only fix is to write the forecast down <strong>before</strong> the outcome exists, in a form that cannot be quietly edited, and then keep score.
        Every weekday this desk seals one-day risk forecasts and, once a week, a valuation call on each company. Each record carries the hash of the one before it, and each run is committed by GitHub with a server-side timestamp.
      </p>
      <div class="card">
        <h2 class="card-title">Chain integrity</h2>
        <div id="chain-box"></div>
        <div class="btn-group" style="margin-top:14px">
          <button type="button" class="btn" id="chain-rerun">Verify again in this browser</button>
          <a class="btn" href="${esc(L.forecasts.file)}" target="_blank" rel="noopener">Raw forecast ledger ↗</a>
          <a class="btn" href="${esc(L.outcomes.file)}" target="_blank" rel="noopener">Raw outcome ledger ↗</a>
          <a class="btn" href="${REPO}/commits/main/docs/ledger" target="_blank" rel="noopener">Commit history ↗</a>
        </div>
      </div>

      <div class="card" style="margin-top:18px">
        <h2 class="card-title">Live track record <span class="tag live">sealed before the outcome</span></h2>
        ${live ? `
          ${sc.var.length ? `<div class="table-scroll"><table class="data-table"><thead><tr><th>Portfolio</th><th>Model</th><th>Level</th><th>Scored</th><th>Breaches</th><th>Expected</th><th>Kupiec p</th><th>Quantile loss ×10⁴</th></tr></thead><tbody>
            ${sc.var.map((r) => `<tr><td>${SUBJECT[r.subject] || esc(r.subject)}</td><td>${MODEL[r.model]}</td><td>${F.pct(r.confidence, 0)}</td><td>${r.n}</td><td>${r.exceedances}</td><td>${F.num(r.expected, 2)}</td><td>${r.n >= 60 ? F.p(r.kupiec_p) : 'too few'}</td><td>${(r.pinball * 1e4).toFixed(3)}</td></tr>`).join('')}</tbody></table></div>` : ''}
          ${sc.vol.length ? `<div class="table-scroll" style="margin-top:14px"><table class="data-table"><thead><tr><th>Volatility model</th><th>Scored</th><th>QLIKE (lower is better)</th></tr></thead><tbody>${sc.vol.map((r) => `<tr><td>${MODEL[r.model]}</td><td>${r.n}</td><td>${r.qlike.toFixed(4)}</td></tr>`).join('')}</tbody></table></div>` : ''}
          ${sc.valuation.length ? `<div class="table-scroll" style="margin-top:14px"><table class="data-table"><thead><tr><th>Valuation cohort</th><th>Horizon</th><th>Companies</th><th>Rank correlation</th><th>Hit rate</th></tr></thead><tbody>${sc.valuation.map((r) => `<tr><td>${esc(r.cohort)}</td><td>${r.horizon_weeks} weeks</td><td>${r.n}</td><td>${ok(r.rank_ic) ? r.rank_ic.toFixed(2) : 'n/a'}</td><td>${F.pct(r.hit_rate, 0)}</td></tr>`).join('')}</tbody></table></div>` : ''}
          <p class="legend-note">Only forecasts whose outcomes are in the ledger are counted. Coverage tests are shown once a model has at least 60 scored forecasts; before that the sample says nothing.</p>`
        : `<div class="empty-state"><strong>No outcomes yet, by design.</strong> The ledger started on ${F.date(L.started)}. The ${L.pending_risk_forecasts} risk forecasts sealed so far are for the trading session after ${F.date(riskAsOf)}; they are scored on the first run after that session's data is published.
            The first valuation calls are scored on ${F.date(L.first_valuation_score_due)} (${L.valuation_horizons_weeks[0]} weeks), then at ${L.valuation_horizons_weeks.slice(1).join(', ')} weeks. A track record cannot be backdated, which is the point; until it accumulates, the reconstructed backtest below is the evidence.</div>`}
      </div>

      <div class="grid-2" style="margin-top:18px">
        <div class="card">
          <h2 class="card-title">Latest sealed risk forecasts</h2>
          <p class="card-desc">One-day value-at-risk for the session after ${F.date(riskAsOf)}, as a share of portfolio value. Sealed ${F.stamp(made)}.</p>
          <div class="table-scroll"><table class="data-table"><thead><tr><th>Portfolio</th><th>Model</th><th>95% VaR</th><th>99% VaR</th></tr></thead><tbody>
            ${subjects.map((s) => ['hist', 'normal', 'ewma'].map((m, i) => `<tr><td>${i === 0 ? SUBJECT[s] : ''}</td><td>${MODEL[m]}</td><td>${cell(s, m, 0.95)}</td><td>${cell(s, m, 0.99)}</td></tr>`).join('')).join('')}
          </tbody></table></div>
          <div class="table-scroll" style="margin-top:14px"><table class="data-table"><thead><tr><th>Nifty 50 volatility forecast</th><th>One day</th><th>Annualised</th></tr></thead><tbody>
            ${vol.map((r) => `<tr><td>${MODEL[r.model]}</td><td>${F.pct(Math.sqrt(r.value), 2)}</td><td>${F.pct(Math.sqrt(r.value * 252), 1)}</td></tr>`).join('')}
          </tbody></table></div>
        </div>
        <div class="card">
          <h2 class="card-title">Latest sealed valuation calls (cohort ${esc(L.valuation_cohorts[L.valuation_cohorts.length - 1] || '–')})</h2>
          <p class="card-desc">The claim being tested: companies with a smaller gap to their DCF baseline outperform those with a larger one. Scored on return relative to the equal-weighted universe.</p>
          <div class="table-scroll" style="max-height:430px;overflow-y:auto"><table class="data-table"><thead><tr><th>Company</th><th>Sector</th><th>Price</th><th>DCF value</th><th>Gap</th></tr></thead><tbody>
            ${calls.map((r) => `<tr><td>${esc(r.subject)}</td><td>${SECTOR[r.sector] ? SECTOR[r.sector][0] : esc(r.sector)}</td><td>${F.px(r.price)}</td><td>${F.px(r.dcf_value)}</td><td class="${dclass(r.value)}">${F.spct(r.value, 0)}</td></tr>`).join('')}
          </tbody></table></div>
        </div>
      </div>

      <div class="card" style="margin-top:18px">
        <h2 class="card-title">Reconstructed backtest <span class="tag recon">not timestamped</span></h2>
        <p class="card-desc">What the same models would have forecast on the portfolio lab's history, walking forward one day at a time. It is honest about information (no forecast sees its own day) but it was computed after the fact, so it is shown separately from the live record and carries less weight.</p>
        <div class="btn-group" style="margin-bottom:12px" role="group" aria-label="Portfolio">${subjects.map((s) => `<button type="button" class="btn" data-subject="${s}">${SUBJECT[s]}</button>`).join('')}</div>
        <div class="grid-2">
          <div><div class="axis-caption">95% VaR: days the loss exceeded the forecast</div><div class="chart-wrap short"><canvas id="bt-095" role="img" aria-label="Breaches of the 95 percent VaR by model against the expected number"></canvas></div></div>
          <div><div class="axis-caption">99% VaR: days the loss exceeded the forecast</div><div class="chart-wrap short"><canvas id="bt-099" role="img" aria-label="Breaches of the 99 percent VaR by model against the expected number"></canvas></div></div>
        </div>
        <div class="legend-row" style="margin-top:10px">${['hist', 'normal', 'ewma'].map((m) => `<span><span class="swatch" style="background:${MODEL_COL[m]}"></span>${MODEL[m]}</span>`).join('')}</div>
        <div class="table-scroll" id="bt-table"></div>
        <div class="table-scroll" style="margin-top:16px"><table class="data-table"><thead><tr><th>Nifty 50 one-day volatility model</th><th>QLIKE (lower is better)</th><th>MSE ×10⁸</th><th>Average forecast, annualised</th></tr></thead><tbody>
          ${Object.entries(vb.models).map(([m, x]) => `<tr><td>${MODEL[m]}</td><td>${x.qlike.toFixed(4)}</td><td>${(x.mse * 1e8).toFixed(3)}</td><td>${F.pct(x.mean_vol_annual, 1)}</td></tr>`).join('')}
        </tbody></table></div>
        <div class="legend-note">Realised volatility over the same ${F.num(vb.n)} days: ${F.pct(vb.realised_vol_annual, 1)} annualised. GARCH parameters are re-estimated every 63 sessions on the history to date.</div>
      </div>`;
    $('chain-rerun').addEventListener('click', verifyLedgers);
    host.querySelectorAll('[data-subject]').forEach((b) => {
      b.classList.toggle('active', b.dataset.subject === state.subject);
      b.addEventListener('click', () => { state.subject = b.dataset.subject; host.querySelectorAll('[data-subject]').forEach((x) => x.classList.toggle('active', x === b)); drawBacktest(); });
    });
    paintChain();
    drawBacktest();
  }

  /* ---------------- 05 method ---------------- */
  function buildMethod(host) {
    host.innerHTML = `
      <p class="section-intro">How the desk works, what its results do and do not prove, and how to check it yourself without trusting this page.</p>
      <div class="grid-2">
        <div class="card">
          <h2 class="card-title">Principles</h2>
          <ul class="note-list">
            <li><strong>Outside-in.</strong> The desk reads only the JSON each project publishes. It does not import their code, so it cannot inherit their bugs.</li>
            <li><strong>A different method where one exists.</strong> Iterative optimiser against exact enumeration. Discounted cash flow against economic profit. Agreement between two routes is evidence; agreement of a routine with itself is not.</li>
            <li><strong>Failures are published.</strong> A failed check is a finding. The run commits its results first and only then fails the workflow, so a problem shows up on this page and as a red build.</li>
            <li><strong>Forecasts are sealed, then scored.</strong> Only outcomes that arrive after a forecast was committed count toward the live record.</li>
            <li><strong>The validator is tested too.</strong> The exact optimiser is checked against an iterative solver and closed forms; the test statistics against hand calculations and SciPy; the ledger against four kinds of tampering.</li>
          </ul>
        </div>
        <div class="card">
          <h2 class="card-title">What agreement does not prove</h2>
          <ul class="note-list">
            <li>That the inputs are right. The desk validates results against published inputs, not the inputs against audited filings.</li>
            <li>That the models are the right models. Economic profit and DCF agree because they are the same model in two forms; both inherit the same assumptions.</li>
            <li>That forecasts will hold. That is what the ledger is for, and it needs time: with a 1% tail, a model must be watched for months before a coverage test means anything.</li>
            <li>That the chain proves timing by itself. A hash chain proves the record was not altered relative to its head. The timing evidence is the public commit history, which GitHub timestamps.</li>
          </ul>
        </div>
      </div>
      <div class="card" style="margin-top:18px">
        <h2 class="card-title">Check it yourself</h2>
        <p class="card-desc">Everything is in <a href="${REPO}" target="_blank" rel="noopener">the repository</a>. To re-run the whole desk and its tests:</p>
        <pre class="hash" style="background:var(--surface-3);border:1px solid var(--border-soft);border-radius:6px;padding:12px 14px;overflow-x:auto;white-space:pre">pip install -r requirements.txt
python -m pytest tests -q        # the desk's own tools
python run.py                    # validate both projects, update the ledger
python run.py --assert-clean     # exit 1 if any check failed or a chain is broken</pre>
        <p class="card-desc" style="margin-top:14px">To verify the ledger with nothing but a Python prompt:</p>
        <pre class="hash" style="background:var(--surface-3);border:1px solid var(--border-soft);border-radius:6px;padding:12px 14px;overflow-x:auto;white-space:pre">import hashlib, json
prev = "0" * 64
for line in open("docs/ledger/forecasts.jsonl"):
    e = json.loads(line)
    assert e["prev"] == prev
    assert e["hash"] == hashlib.sha256((prev + "\\n" + e["payload"]).encode()).hexdigest()
    prev = e["hash"]
print("chain intact, head", prev)</pre>
      </div>
      <div class="card" style="margin-top:18px">
        <h2 class="card-title">References</h2>
        <ul class="note-list">
          <li>Board of Governors of the Federal Reserve System and OCC, <em>Supervisory Guidance on Model Risk Management</em> (SR 11-7), 2011: conceptual soundness, ongoing monitoring and outcomes analysis.</li>
          <li>Kupiec (1995), proportion-of-failures test; Christoffersen (1998), independence of VaR exceedances; Basel Committee (1996), traffic-light backtesting.</li>
          <li>Gneiting and Raftery (2007), strictly proper scoring rules; Patton (2011), QLIKE as a robust loss for volatility forecasts.</li>
          <li>Jorion (1986), Bayes-Stein estimation for portfolio analysis; Koller, Goedhart and Wessels, <em>Valuation</em>: equivalence of DCF and economic profit.</li>
        </ul>
      </div>`;
  }

  /* ---------------- tear sheet ---------------- */
  function openSheet() {
    const s = D.summary, L = D.ledger, P = D.portfolio, V = D.valuation;
    const groups = (checks) => { const g = {}; checks.forEach((c) => { g[c.group] = g[c.group] || [0, 0]; g[c.group][1]++; if (c.ok) g[c.group][0]++; }); return Object.entries(g).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v[0]} / ${v[1]}</td></tr>`).join(''); };
    const bt = P.var_backtest.policy_orp.models;
    $('memo-page').innerHTML = `
      <div class="memo-head"><div><div style="font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:#4a5568">Validation run tear sheet</div><h2>Verifiable Research Desk</h2></div>
        <div class="meta">Run ${F.stamp(D.meta.generated_utc)}<br>Niranjan Desai</div></div>
      <div class="memo-kpis">
        <div class="memo-kpi"><div class="l">Independent checks</div><div class="v">${s.checks_passed} / ${s.checks_total}</div><div class="s">${s.checks_failed ? s.checks_failed + ' differ' : 'all agree'}</div></div>
        <div class="memo-kpi"><div class="l">Forecasts sealed</div><div class="v">${L.forecasts.entries}</div><div class="s">since ${F.date(L.started)}</div></div>
        <div class="memo-kpi"><div class="l">Outcomes scored</div><div class="v">${L.outcomes.entries}</div><div class="s">${L.pending_risk_forecasts} pending</div></div>
        <div class="memo-kpi"><div class="l">Ledger chain</div><div class="v">${L.forecasts.verified && L.outcomes.verified ? 'Intact' : 'BROKEN'}</div><div class="s">head ${esc(L.forecasts.head.slice(0, 10))}…</div></div>
      </div>
      <div class="memo-cols">
        <div><h3>Portfolio Construction Lab (as of ${F.date(P.as_of)})</h3><table><tbody>${groups(P.checks)}</tbody></table>
          <h3>99% VaR backtest, policy portfolio</h3><table><thead><tr><th>Model</th><th>Breaches</th><th>Expected</th><th>Kupiec p</th></tr></thead><tbody>
          ${Object.keys(bt).map((m) => `<tr><td>${MODEL[m]}</td><td>${bt[m]['0.99'].exceedances}</td><td>${F.num(bt[m]['0.99'].expected, 1)}</td><td>${F.p(bt[m]['0.99'].kupiec_p)}</td></tr>`).join('')}</tbody></table></div>
        <div><h3>Valuation &amp; Deal Lab (as of ${F.date(V.valuation_date)})</h3><table><tbody>${groups(V.checks)}</tbody></table>
          <h3>Observations</h3>${V.observations.slice(0, 3).map((o) => `<p><strong>${esc(o.title)}.</strong> ${esc(o.text)}</p>`).join('')}</div>
      </div>
      <h3>Inputs validated</h3>
      <table><tbody>${Object.values(D.sources).map((x) => `<tr><td>${esc(x.name)} (${esc(x.origin)})</td><td style="font-size:9px">${esc(x.sha256)}</td></tr>`).join('')}</tbody></table>
      <div class="fine">Checks compare each project's published results with an independent recomputation, by a different method where one exists. Agreement validates the arithmetic, not the assumptions or the forecasts; the ledger exists to score the forecasts. Educational project; not investment advice.</div>`;
    $('memo-overlay').classList.add('open');
  }

  /* ---------------- routing ---------------- */
  const BUILD = { overview: buildOverview, portfolio: buildPortfolio, valuation: buildValuation, ledger: buildLedger, method: buildMethod };
  function show() {
    document.querySelectorAll('.tab-btn').forEach((b) => { const on = b.dataset.tab === state.tab; b.classList.toggle('active', on); b.setAttribute('aria-selected', on ? 'true' : 'false'); b.tabIndex = on ? 0 : -1; });
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === 'panel-' + state.tab));
    BUILD[state.tab]($('panel-' + state.tab));
    history.replaceState(null, '', '#' + state.tab);
  }
  function go(tab) { if (TABS.includes(tab)) { state.tab = tab; show(); window.scrollTo({ top: 0 }); } }
  function readHash() { const t = location.hash.replace(/^#/, ''); if (TABS.includes(t)) state.tab = t; }

  function init() {
    readHash();
    $('as-of').textContent = F.stamp(D.meta.generated_utc);
    const clean = D.summary.checks_failed === 0;
    const badge = $('head-badge');
    badge.textContent = clean ? `ALL ${D.summary.checks_total} CHECKS AGREE` : `${D.summary.checks_failed} CHECKS DIFFER`;
    if (!clean) { badge.style.borderColor = 'var(--critical)'; badge.style.color = '#ff8a8a'; }
    renderKpis();
    $('tab-nav').addEventListener('click', (ev) => { const b = ev.target.closest('.tab-btn'); if (b) go(b.dataset.tab); });
    $('tab-nav').addEventListener('keydown', (ev) => {
      if (ev.key !== 'ArrowRight' && ev.key !== 'ArrowLeft') return;
      const i = TABS.indexOf(state.tab);
      go(TABS[(i + (ev.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length]);
      $('tab-' + state.tab).focus();
    });
    window.addEventListener('hashchange', () => { readHash(); show(); });
    $('open-sheet').addEventListener('click', openSheet);
    $('memo-close').addEventListener('click', () => $('memo-overlay').classList.remove('open'));
    $('memo-print').addEventListener('click', () => window.print());
    document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') $('memo-overlay').classList.remove('open'); });
    $('footer-note').innerHTML = `<strong>Verifiable Research Desk</strong> · built by Niranjan Desai. Run ${F.stamp(D.meta.generated_utc)}. ` +
      `It validates the <a href="${esc(D.sources.portfolio.site)}" target="_blank" rel="noopener">Portfolio Construction Lab</a> and the <a href="${esc(D.sources.valuation.site)}" target="_blank" rel="noopener">Valuation &amp; Deal Lab</a> from their published data and keeps a public, hash-chained record of forecasts. ` +
      `<strong>Educational project; nothing here is investment advice.</strong>`;
    show();
    $('boot').classList.add('done');
    verifyLedgers();
  }

  if (!D || !window.Chart) $('boot-label').textContent = 'Could not load the results file or the chart library. Check your connection and reload.';
  else init();
})();
