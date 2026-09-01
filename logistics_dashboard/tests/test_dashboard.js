/* ══════════════════════════════════════════════════════════════════════════
   Dashboard validation harness.

   Loads dashboard.html into a headless DOM with the charting libraries stubbed,
   then drives every module the way a user would. This catches the class of bug
   that actually breaks a data dashboard - a renderer reaching for a payload key
   that isn't there, a filter that silently produces NaN, a table column bound to
   a field the aggregator never emits - without needing a real browser.

   It also reconciles the figures the page displays against the payload, so a
   chart cannot quietly disagree with its own data.

   Run:  node tests/test_dashboard.js
   ══════════════════════════════════════════════════════════════════════════ */

const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('/tmp/node_modules/jsdom');

const ROOT = path.resolve(__dirname, '..');
const HTML = path.join(ROOT, 'dashboard.html');

let pass = 0, fail = 0;
const failures = [];

function check(name, fn) {
  try {
    const msg = fn();
    console.log(`  PASS  ${name}${msg ? ' — ' + msg : ''}`);
    pass++;
  } catch (e) {
    console.log(`  FAIL  ${name}: ${e.message}`);
    failures.push(name);
    fail++;
  }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }
function near(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) throw new Error(`${msg} (${a} vs ${b})`);
}

/* ------------------------------------------------------------------ setup -- */
const consoleErrors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => consoleErrors.push('jsdomError: ' + e.message));
vc.on('error', (...a) => consoleErrors.push('console.error: ' + a.join(' ')));

// Record every chart, so we can assert charts were actually produced.
const charts = {};
const plotlyStub = {
  react: (el, traces, layout) => {
    charts[el.id] = { traces, layout, n: traces.length };
    return Promise.resolve();
  },
  newPlot: (...a) => plotlyStub.react(...a),
  purge: () => {}
};

// Leaflet stub that records layers and markers.
const mapState = { layers: [], markers: 0, polylines: 0, tiles: 0, view: null };
function layerObj(kind) {
  return { _kind: kind, addTo(){ return this; }, bindPopup(){ return this; },
           remove(){}, setStyle(){ return this; } };
}
const leafletStub = {
  map: () => ({
    setView(v){ mapState.view = v; return this; },
    addLayer(){}, removeLayer(){}, invalidateSize(){}, on(){}, remove(){}
  }),
  tileLayer: () => { mapState.tiles++; return layerObj('tile'); },
  circleMarker: () => { mapState.markers++; return layerObj('marker'); },
  marker:       () => { mapState.markers++; return layerObj('marker'); },
  polyline:     () => { mapState.polylines++; return layerObj('polyline'); },
  layerGroup:   (arr) => { mapState.layers.push((arr||[]).length); return layerObj('group'); },
  divIcon: () => ({}), latLngBounds: () => ({})
};

const dom = new JSDOM(fs.readFileSync(HTML, 'utf8'), {
  runScripts: 'outside-only',
  pretendToBeVisual: true,
  virtualConsole: vc,
  url: 'file://' + HTML
});
const { window } = dom;

// Inject stubs before the app script runs.
window.Plotly = plotlyStub;
window.L = leafletStub;
window.d3 = { select: () => ({ append(){return this;}, attr(){return this;},
                               style(){return this;}, text(){return this;} }) };
window.XLSX = { utils: { book_new: () => ({}), json_to_sheet: () => ({}),
                         book_append_sheet: () => {} }, writeFile: () => {} };
window.jspdf = { jsPDF: function () {
  return { setFontSize(){}, setTextColor(){}, text(){}, addPage(){}, save(){},
           splitTextToSize: (t) => [t] };
} };
window.tailwind = { config: {} };
window.matchMedia = window.matchMedia || (() => ({ matches:false, addListener(){}, removeListener(){} }));
// jsdom does not implement scrolling; without a stub every navigation emits a
// spurious "Not implemented" error that drowns out real ones.
window.scrollTo = () => {};
window.HTMLElement.prototype.scrollIntoView = () => {};

// Execute the application script that the build inlined.
const appScript = [...window.document.querySelectorAll('script')]
  .map(s => s.textContent)
  .filter(t => t && t.includes('RENDERERS'))
  .pop();
if (!appScript) { console.log('FATAL: could not locate the app script in dashboard.html'); process.exit(1); }

try {
  window.eval(appScript);
} catch (e) {
  console.log('FATAL: app script threw on load: ' + e.message);
  console.log(e.stack);
  process.exit(1);
}

// With runScripts:'outside-only' jsdom leaves readyState as 'loading', so the
// app's own DOMContentLoaded bootstrap never fires. Drive it explicitly rather
// than racing an async event — otherwise assertions run against a half-built DOM.
if (window.document.getElementById('nav').children.length === 0) window.init();

const doc = window.document;
const DATA = JSON.parse(doc.getElementById('payload').textContent);

console.log('\n── structure ───────────────────────────────────────────────');

check('payload parses and carries every section', () => {
  ['meta','cube','executive','operations','fleet','warehouse','forecast',
   'map','ml','optimization','simulator','recommendations']
    .forEach(k => assert(DATA[k] != null, `missing section: ${k}`));
  return `${Object.keys(DATA).length} sections`;
});

check('all nine modules exist in the DOM', () => {
  const ids = ['executive','operations','fleet','warehouse','delivery',
               'forecast','simulator','recommendations','xai'];
  ids.forEach(id => assert(doc.getElementById('m-' + id), `missing section m-${id}`));
  const nav = doc.querySelectorAll('[data-m]');
  assert(nav.length === 9, `nav has ${nav.length} entries, expected 9`);
  return '9 modules, 9 nav entries';
});

check('no errors raised while loading', () => {
  assert(consoleErrors.length === 0, consoleErrors.slice(0,3).join(' | '));
  return 'clean load';
});

check('every element the app looks up exists in the markup', () => {
  // getElementById returning null is the quietest bug in a dashboard: the
  // renderer guards, skips, and the panel is simply blank with no error.
  const ids = new Set();
  for (const m of appScript.matchAll(/getElementById\(\s*'([^']+)'\s*\)/g)) ids.add(m[1]);
  const missing = [...ids].filter(id => !doc.getElementById(id));
  assert(missing.length === 0, 'referenced but absent: ' + missing.join(', '));
  return `${ids.size} ids all present`;
});

console.log('\n── rendering ───────────────────────────────────────────────');

const MODULES = ['executive','operations','fleet','warehouse','delivery',
                 'forecast','simulator','recommendations','xai'];

MODULES.forEach(id => {
  check(`module renders: ${id}`, () => {
    const before = consoleErrors.length;
    window.go(id);
    const errs = consoleErrors.slice(before);
    assert(errs.length === 0, errs.slice(0,2).join(' | '));
    const active = doc.querySelector('.module.active');
    assert(active && active.id === 'm-' + id, 'module did not become active');
    return `${Object.keys(charts).length} charts drawn so far`;
  });
});

check('every chart container in the markup gets drawn', () => {
  // The mirror of the check above: a div sized for a chart that nothing ever
  // renders into leaves a hole in the page.
  const containers = [...doc.querySelectorAll('div[id][style*="height"]')]
    .map(d => d.id)
    .filter(id => id && id !== 'map');
  MODULES.forEach(m => window.go(m));
  const blank = containers.filter(id => {
    if (charts[id]) return false;
    const el = doc.getElementById(id);
    // An explicit empty-state message counts as handled.
    return !el.textContent.trim();
  });
  assert(blank.length === 0, 'containers never filled: ' + blank.join(', '));
  return `${containers.length} containers, all filled or explained`;
});

check('every chart container received a trace', () => {
  const drawn = Object.keys(charts);
  assert(drawn.length >= 30, `only ${drawn.length} charts drawn`);
  const emptyCharts = drawn.filter(k => !charts[k].n);
  assert(emptyCharts.length === 0, 'charts with no traces: ' + emptyCharts.join(', '));
  return `${drawn.length} charts, all with data`;
});

check('no chart contains NaN or undefined values', () => {
  const bad = [];
  Object.entries(charts).forEach(([id, c]) => {
    c.traces.forEach(t => {
      ['x','y','z','values'].forEach(axis => {
        const v = t[axis];
        if (!Array.isArray(v)) return;
        const flat = Array.isArray(v[0]) ? v.flat() : v;
        if (flat.some(n => typeof n === 'number' && !isFinite(n))) bad.push(`${id}.${axis}`);
      });
    });
  });
  assert(bad.length === 0, 'non-finite values in: ' + bad.slice(0,5).join(', '));
  return 'all numeric series finite';
});

check('KPI cards populated in every module', () => {
  const ids = ['exKpis','opKpis','flKpis','whKpis','dlKpis','fcKpis','simKpis','recKpis','xaiKpis'];
  const emptyBlocks = ids.filter(id => {
    const el = doc.getElementById(id);
    return !el || el.children.length === 0;
  });
  assert(emptyBlocks.length === 0, 'empty KPI blocks: ' + emptyBlocks.join(', '));
  const dashes = ids.filter(id => {
    const txt = doc.getElementById(id).textContent;
    return (txt.match(/—/g) || []).length > 3;
  });
  assert(dashes.length === 0, 'KPI blocks mostly blank: ' + dashes.join(', '));
  return ids.length + ' KPI blocks filled';
});

check('tables rendered with rows', () => {
  const ids = ['exTable','flTable','flDrivers','whTable','recOptTable'];
  ids.forEach(id => {
    const rows = doc.querySelectorAll(`#${id} tbody tr`);
    assert(rows.length > 0, `${id} has no rows`);
  });
  return ids.map(id => `${id}:${doc.querySelectorAll(`#${id} tbody tr`).length}`).join(' ');
});

console.log('\n── data reconciliation ─────────────────────────────────────');

check('cube totals match the executive KPI block', () => {
  const t = DATA.cube.reduce((a, r) => {
    a.orders += r.orders; a.revenue += r.revenue; a.cost += r.cost; a.late += r.late;
    return a;
  }, { orders:0, revenue:0, cost:0, late:0 });
  const k = DATA.executive.kpis;
  assert(t.orders === k.total_orders, `orders ${t.orders} vs ${k.total_orders}`);
  near(t.revenue, k.revenue_usd, Math.abs(k.revenue_usd) * 0.001, 'revenue mismatch');
  near(t.cost, k.cost_usd, Math.abs(k.cost_usd) * 0.001, 'cost mismatch');
  const onTime = (1 - t.late / t.orders) * 100;
  near(onTime, k.on_time_rate, 0.05, 'on-time rate mismatch');
  return `${t.orders.toLocaleString()} orders reconcile`;
});

check('displayed executive KPIs equal the payload', () => {
  window.go('executive');
  const txt = doc.getElementById('exKpis').textContent;
  const k = DATA.executive.kpis;
  assert(txt.includes(k.total_orders.toLocaleString('en')),
    `order count not shown (looking for ${k.total_orders.toLocaleString('en')})`);
  assert(txt.includes(k.on_time_rate.toFixed(1)), 'on-time rate not shown');
  return 'orders and on-time rate on screen';
});

check('filters genuinely re-aggregate the numbers', () => {
  window.go('executive');
  const all = doc.getElementById('exKpis').textContent;

  doc.getElementById('fRegion').value = DATA.meta.regions[0];
  doc.getElementById('fRegion').onchange();
  const filtered = doc.getElementById('exKpis').textContent;
  assert(all !== filtered, 'KPI block unchanged after filtering — filter is decorative');

  // The filtered order count must equal the cube slice, not just differ.
  const expect = DATA.cube.filter(r => r.region === DATA.meta.regions[0])
                          .reduce((s, r) => s + r.orders, 0);
  assert(filtered.includes(expect.toLocaleString('en')),
    `expected ${expect.toLocaleString('en')} orders for ${DATA.meta.regions[0]}`);

  doc.getElementById('btnReset').onclick();
  const reset = doc.getElementById('exKpis').textContent;
  assert(reset === all, 'reset did not restore the unfiltered view');
  return `${DATA.meta.regions[0]} → ${expect.toLocaleString('en')} orders`;
});

check('every region and domain filter combination renders', () => {
  let n = 0;
  DATA.meta.regions.forEach(region => {
    DATA.meta.domains.slice(0, 3).forEach(domain => {
      doc.getElementById('fRegion').value = region;
      doc.getElementById('fDomain').value = domain;
      const before = consoleErrors.length;
      doc.getElementById('fRegion').onchange();
      assert(consoleErrors.length === before,
        `${region}/${domain}: ${consoleErrors.slice(before)[0]}`);
      n++;
    });
  });
  doc.getElementById('btnReset').onclick();
  return `${n} combinations clean`;
});

console.log('\n── map ─────────────────────────────────────────────────────');

check('map builds all seven layers', () => {
  window.go('delivery');
  assert(mapState.tiles > 0, 'no tile layer added');
  assert(mapState.layers.length === 7, `${mapState.layers.length} layer groups, expected 7`);
  assert(mapState.markers > 100, `only ${mapState.markers} markers created`);
  assert(mapState.polylines > 0, 'no route polylines drawn');
  return `${mapState.markers} markers, ${mapState.polylines} routes, ${mapState.layers.length} layers`;
});

check('map layer toggles are wired', () => {
  const btns = doc.querySelectorAll('.mapLayer');
  assert(btns.length === 7, `${btns.length} layer buttons, expected 7`);
  const first = btns[0];
  const wasOn = first.dataset.on;
  first.onclick();
  assert(first.dataset.on !== wasOn, 'toggle did not flip state');
  first.onclick();
  return '7 toggles responding';
});

check('map coordinates are plausible', () => {
  const bad = [];
  DATA.map.warehouses.forEach(w => {
    if (Math.abs(w.lat) > 90 || Math.abs(w.lon) > 180) bad.push(w.id);
  });
  DATA.map.vehicles.forEach(v => {
    if (Math.abs(v.lat) > 90 || Math.abs(v.lon) > 180) bad.push(v.vehicle_id);
  });
  assert(bad.length === 0, 'out-of-range coordinates: ' + bad.slice(0,3).join(', '));
  return `${DATA.map.warehouses.length} sites, ${DATA.map.vehicles.length} vehicles in range`;
});

console.log('\n── simulator ───────────────────────────────────────────────');

check('baseline scenario reproduces the actual network', () => {
  window.go('simulator');
  const base = window.simulate(window.baselineScenario());
  const B = DATA.simulator.baseline;
  near(base.orders, B.orders, 1, 'baseline order count drifted');
  near(base.on_time, B.on_time_rate, 0.01, 'baseline on-time drifted');
  // Cost is rebuilt from its components, so allow a small reconstruction gap.
  near(base.cost, B.total_cost, B.total_cost * 0.05, 'baseline cost drifted');
  return `on-time ${base.on_time.toFixed(2)}% vs actual ${B.on_time_rate}%`;
});

check('simulator responds in the right direction', () => {
  const base = window.simulate(window.baselineScenario());
  const v = window.baselineScenario;

  const moreTraffic = window.simulate({ ...v(), traffic: 180 });
  assert(moreTraffic.on_time < base.on_time, 'heavier traffic did not reduce on-time rate');
  assert(moreTraffic.cost > base.cost, 'heavier traffic did not raise cost');

  const pricierFuel = window.simulate({ ...v(), fuel: 200 });
  assert(pricierFuel.cost > base.cost, 'fuel price rise did not raise cost');

  // Vehicles and drivers are complementary: scaling both must clearly help.
  const resourced = window.simulate({ ...v(), demand: 150, fleet: 150, drivers: 150 });
  const strained  = window.simulate({ ...v(), demand: 150, fleet: 100, drivers: 100 });
  assert(resourced.on_time > strained.on_time,
    'adding fleet and drivers under the same demand did not improve service');

  // Adding only the abundant factor should help a little, but far less than
  // adding the scarce one - that asymmetry is the point of the capacity model.
  const fleetOnly  = window.simulate({ ...v(), demand: 150, fleet: 150, drivers: 100 });
  assert(fleetOnly.on_time > strained.on_time,
    'spare vehicles gave no benefit at all — capacity model is pure Leontief');
  assert(fleetOnly.on_time < resourced.on_time,
    'fleet alone matched fleet+drivers — the binding constraint is not binding');

  const optimised = window.simulate({ ...v(), route: 100 });
  assert(optimised.km < base.km, 'route optimisation did not reduce distance');
  assert(optimised.on_time > base.on_time, 'route optimisation did not improve service');

  const denser = window.simulate({ ...v(), wh: 150 });
  assert(denser.km_per_order < base.km_per_order,
    'more warehouses did not shorten the average leg');

  const badWeather = window.simulate({ ...v(), weather: 250 });
  assert(badWeather.on_time < base.on_time, 'adverse weather did not reduce on-time rate');

  return 'traffic, fuel, fleet, routing, warehouses and weather all directionally correct';
});

check('simulator output stays within physical bounds', () => {
  const v = window.baselineScenario;
  const extremes = [
    { ...v(), demand:250, fleet:50, drivers:50, traffic:200, weather:300 },
    { ...v(), demand:50, fleet:200, drivers:150, traffic:50, weather:0, route:100 },
    { ...v(), fuel:250, wh:200 }
  ];
  extremes.forEach((s, i) => {
    const r = window.simulate(s);
    assert(isFinite(r.cost) && r.cost > 0, `scenario ${i}: cost is ${r.cost}`);
    assert(r.on_time >= 0 && r.on_time <= 100, `scenario ${i}: on-time ${r.on_time}%`);
    assert(isFinite(r.margin), `scenario ${i}: margin is ${r.margin}`);
    assert(r.km > 0, `scenario ${i}: distance ${r.km}`);
  });
  return '3 extreme scenarios bounded';
});

check('scenarios can be saved and cleared', () => {
  window.go('simulator');
  doc.getElementById('simSave').onclick();
  doc.getElementById('simSave').onclick();
  let rows = doc.querySelectorAll('#simTable tbody tr');
  assert(rows.length === 2, `expected 2 saved scenarios, got ${rows.length}`);
  doc.getElementById('simClear').onclick();
  rows = doc.querySelectorAll('#simTable tbody tr');
  assert(rows.length <= 1, 'clear did not empty the scenario table');
  return 'save and clear working';
});

console.log('\n── ML / explainability ─────────────────────────────────────');

check('SHAP attribution is present and ordered', () => {
  const s = DATA.ml.shap_global;
  assert(s && s.length >= 5, 'fewer than 5 SHAP features');
  for (let i = 1; i < s.length; i++) {
    assert(s[i-1].mean_abs_shap >= s[i].mean_abs_shap, 'SHAP values not sorted descending');
  }
  return `${s.length} features, top = ${s[0].feature}`;
});

check('model leaderboard metrics are in range', () => {
  const lb = DATA.ml.leaderboard;
  assert(lb && lb.length >= 3, 'fewer than 3 models on the leaderboard');
  lb.forEach(m => {
    ['roc_auc','accuracy','precision','recall','f1'].forEach(k => {
      if (m[k] != null) assert(m[k] >= 0 && m[k] <= 1, `${m.model}.${k} = ${m[k]} out of range`);
    });
  });
  return `${lb.length} models, best ${lb[0].model}`;
});

check('per-instance explanation switches instance', () => {
  window.go('xai');
  const sel = doc.getElementById('xaiInstance');
  assert(sel.options.length > 1, 'only one instance available');
  const first = JSON.stringify(charts['xaiLocal'].traces[0].y);
  sel.value = sel.options[1].value;
  sel.onchange({ target: sel });
  const second = JSON.stringify(charts['xaiLocal'].traces[0].y);
  assert(first !== second, 'switching instance did not change the explanation');
  return `${sel.options.length} instances`;
});

console.log('\n── forecast ────────────────────────────────────────────────');

check('forecast interval always brackets the point estimate', () => {
  const f = DATA.forecast.forecast;
  assert(f.length > 0, 'no forecast points');
  f.forEach((p, i) => {
    assert(p.lower <= p.point && p.point <= p.upper,
      `point ${i}: ${p.lower} / ${p.point} / ${p.upper} out of order`);
  });
  const widthFirst = f[0].upper - f[0].lower;
  const widthLast  = f[f.length-1].upper - f[f.length-1].lower;
  assert(widthLast > widthFirst, 'uncertainty does not widen with horizon');
  return `${f.length} days, interval widens ${(widthLast/widthFirst).toFixed(2)}x`;
});

check('forecast accuracy is reported out of sample', () => {
  const a = DATA.forecast.accuracy;
  assert(a.basis && a.basis.includes('out-of-sample'), 'accuracy basis not stated as out-of-sample');
  assert(a.mape > 0 && a.mape < 60, `implausible MAPE ${a.mape}%`);
  assert(a.in_sample_mape < a.mape, 'in-sample error should be lower than out-of-sample');
  return `MAPE ${a.mape}% out-of-sample vs ${a.in_sample_mape}% in-sample`;
});

check('seasonal factors are normalised around 1', () => {
  const dow = DATA.forecast.seasonality.map(s => s.factor);
  const mean = dow.reduce((a,b)=>a+b,0) / dow.length;
  near(mean, 1, 0.02, 'weekday factors not centred on 1');
  const months = DATA.forecast.month_factors.map(m => m.factor);
  near(months.reduce((a,b)=>a+b,0)/months.length, 1, 0.02, 'month factors not centred on 1');
  return `weekday mean ${mean.toFixed(3)}`;
});

console.log('\n── recommendations & exports ───────────────────────────────');

check('recommendations are evidenced and ranked', () => {
  const R = DATA.recommendations;
  assert(R.length >= 4, `only ${R.length} recommendations`);
  R.forEach(r => {
    assert(r.title && r.evidence && r.action, `recommendation ${r.rank} is missing a field`);
    assert(r.evidence.length > 40, `recommendation ${r.rank} evidence is too thin`);
    assert(['high','medium','low'].includes(r.confidence), `bad confidence: ${r.confidence}`);
  });
  const q = R.filter(r => r.impact_usd);
  for (let i = 1; i < q.length; i++) {
    assert(q[i-1].impact_usd >= q[i].impact_usd, 'not sorted by impact');
  }
  return `${R.length} recommendations, ${q.length} quantified`;
});

check('export preview builds for all three formats', () => {
  ['pdf','ppt','xls'].forEach(t => {
    const before = consoleErrors.length;
    window.setActiveTab(t);
    assert(consoleErrors.length === before, `${t}: ${consoleErrors.slice(before)[0]}`);
    const body = doc.getElementById('exBody');
    assert(body.innerHTML.length > 200, `${t} preview is empty`);
  });
  return 'pdf, pptx and xlsx previews render';
});

check('Excel export assembles populated sheets', () => {
  const sheets = window.currentTables();
  const names = Object.keys(sheets);
  assert(names.length >= 8, `only ${names.length} sheets`);
  const populated = names.filter(n => (sheets[n] || []).length > 0);
  assert(populated.length >= 8, 'empty sheets: ' +
    names.filter(n => !(sheets[n]||[]).length).join(', '));
  return `${populated.length} sheets with data`;
});

check('PDF and Excel exports run without throwing', () => {
  window.exportExcel();
  window.exportPdf();
  return 'both export paths execute';
});

console.log('\n── theme & interaction ─────────────────────────────────────');

check('theme toggle re-renders cleanly', () => {
  const before = consoleErrors.length;
  doc.getElementById('btnTheme').onclick();
  assert(!doc.documentElement.classList.contains('dark'), 'theme did not switch to light');
  doc.getElementById('btnTheme').onclick();
  assert(doc.documentElement.classList.contains('dark'), 'theme did not switch back');
  assert(consoleErrors.length === before, consoleErrors.slice(before)[0]);
  return 'dark ⇄ light without errors';
});

check('table sorting works', () => {
  window.go('warehouse');
  const th = doc.querySelector('#whTable thead th:nth-child(6)');
  const firstBefore = doc.querySelector('#whTable tbody tr td').textContent;
  th.onclick();
  const firstAfter = doc.querySelector('#whTable tbody tr td').textContent;
  assert(firstBefore !== firstAfter || doc.querySelectorAll('#whTable tbody tr').length === 1,
    'sorting did not reorder rows');
  return 'sort reorders rows';
});

check('navigating every module twice stays clean', () => {
  const before = consoleErrors.length;
  MODULES.forEach(m => window.go(m));
  MODULES.forEach(m => window.go(m));
  assert(consoleErrors.length === before,
    'errors on re-navigation: ' + consoleErrors.slice(before).slice(0,2).join(' | '));
  return '18 navigations clean';
});

check('no errors accumulated across the whole run', () => {
  assert(consoleErrors.length === 0,
    `${consoleErrors.length} errors: ` + consoleErrors.slice(0,3).join(' | '));
  return 'zero console errors';
});

/* ----------------------------------------------------------------- report -- */
console.log('\n════════════════════════════════════════════════════════════');
console.log(`${pass}/${pass + fail} passed`);
if (fail) {
  console.log('failed: ' + failures.join(', '));
  process.exit(1);
}
process.exit(0);
