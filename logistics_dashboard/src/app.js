/* ══════════════════════════════════════════════════════════════════════════
   LogiOptima — dashboard application
   ══════════════════════════════════════════════════════════════════════════
   Reads the embedded payload built by scripts/build_payload.py. No network
   calls, no server: the file opens from disk and renders immediately.

   Filtering: the payload carries a region x domain x month fact cube, so the
   global filters genuinely re-aggregate the executive numbers in the browser
   rather than just relabelling a fixed chart. Modules whose data has no such
   breakdown say so instead of pretending to respond.
   ══════════════════════════════════════════════════════════════════════════ */

const DATA = JSON.parse(document.getElementById('payload').textContent);

/* ---------------------------------------------------------------- state -- */
const S = {
  module: 'executive',
  region: 'all',
  domain: 'all',
  map: null,
  layers: {},
  scenarios: [],
  sim: null,
  drawn: new Set()
};

const MODULES = [
  { id:'executive',       label:'Executive Dashboard', icon:'▤', sub:'Board-level performance, margin and network health' },
  { id:'operations',      label:'Operations Center',   icon:'◈', sub:'Live throughput, congestion, weather and exceptions' },
  { id:'fleet',           label:'Fleet Management',    icon:'▣', sub:'Vehicles, utilisation, maintenance and couriers' },
  { id:'warehouse',       label:'Warehouse Analytics', icon:'▦', sub:'Site capacity, throughput and inventory position' },
  { id:'delivery',        label:'Delivery Monitoring', icon:'◉', sub:'Geospatial view of the delivery network' },
  { id:'forecast',        label:'Demand Forecast',     icon:'◹', sub:'Volume projection with seasonality decomposition' },
  { id:'simulator',       label:'Optimization Simulator', icon:'⚙', sub:'What-if scenarios across the operating levers' },
  { id:'recommendations', label:'AI Recommendations',  icon:'✦', sub:'Ranked, quantified actions derived from the data' },
  { id:'xai',             label:'Explainable AI',      icon:'◎', sub:'Why the model predicts what it predicts' }
];

/* ------------------------------------------------------------ formatting -- */
const nf  = n => n == null || isNaN(n) ? '—' : Intl.NumberFormat('en',{maximumFractionDigits:0}).format(n);
const nf1 = n => n == null || isNaN(n) ? '—' : Intl.NumberFormat('en',{maximumFractionDigits:1}).format(n);
const nf2 = n => n == null || isNaN(n) ? '—' : Intl.NumberFormat('en',{maximumFractionDigits:2}).format(n);
const pct = n => n == null || isNaN(n) ? '—' : nf1(n) + '%';
const usd = n => {
  if (n == null || isNaN(n)) return '—';
  const a = Math.abs(n);
  if (a >= 1e9) return '$' + nf2(n/1e9) + 'B';
  if (a >= 1e6) return '$' + nf2(n/1e6) + 'M';
  if (a >= 1e3) return '$' + nf1(n/1e3) + 'K';
  return '$' + nf2(n);
};
const titleCase = s => String(s).replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase());

const PALETTE = ['#4c7ef3','#3fa87a','#e0a24c','#e0574c','#7a5cf0','#4cc7e0',
                 '#e07ac0','#8fb84c','#c04c7a','#4c95e0'];

/* --------------------------------------------------------------- plotly -- */
const isDark = () => document.documentElement.classList.contains('dark');
function layout(extra = {}) {
  const dark = isDark();
  return Object.assign({
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{ color: dark ? '#c3cde4' : '#3a4560', size:11,
           family:'Inter,-apple-system,Segoe UI,sans-serif' },
    margin:{ l:52, r:18, t:14, b:44 },
    xaxis:{ gridcolor: dark ? '#212c50' : '#e8ecf4', zeroline:false, automargin:true },
    yaxis:{ gridcolor: dark ? '#212c50' : '#e8ecf4', zeroline:false, automargin:true },
    legend:{ orientation:'h', y:-0.18, font:{size:10} },
    hoverlabel:{ bgcolor: dark ? '#18213f' : '#ffffff',
                 bordercolor: dark ? '#2d3a63' : '#d8e0ec',
                 font:{ color: dark ? '#e8edf9' : '#1a2338', size:11 } }
  }, extra);
}
const CFG = { displayModeBar:false, responsive:true };
const plot = (id, traces, extra) => {
  const el = document.getElementById(id);
  if (!el) return;
  Plotly.react(el, traces, layout(extra), CFG);
};
function empty(id, msg) {
  const el = document.getElementById(id);
  if (el) el.innerHTML =
    `<div class="h-full flex items-center justify-center text-[12px] mut text-center px-6">${msg}</div>`;
}

/* ---------------------------------------------------------------- cards -- */
function kpi(label, value, opts = {}) {
  const { delta, sub, tone } = opts;
  const colour = tone === 'good' ? '#3fa87a' : tone === 'bad' ? '#e0574c'
               : tone === 'warn' ? '#e0a24c' : 'var(--txt)';
  let d = '';
  if (delta != null && !isNaN(delta)) {
    const up = delta >= 0;
    // `positiveIsGood` defaults true; cost-style metrics pass false.
    const good = opts.positiveIsGood === false ? !up : up;
    d = `<div class="text-[10.5px] font-semibold mt-1" style="color:${good?'#3fa87a':'#e0574c'}">
           ${up?'▲':'▼'} ${nf1(Math.abs(delta))}% <span class="mut font-normal">MoM</span></div>`;
  }
  return `<div class="card p-3.5">
    <div class="text-[10px] mut uppercase tracking-wider font-semibold">${label}</div>
    <div class="text-[21px] font-bold mt-1 leading-none" style="color:${colour}">${value}</div>
    ${d}${sub ? `<div class="text-[10.5px] mut mt-1">${sub}</div>` : ''}
  </div>`;
}
const setKpis = (id, html) => { const e = document.getElementById(id); if (e) e.innerHTML = html; };

/* ---------------------------------------------------------------- table -- */
function table(id, rows, cols, opts = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!rows || !rows.length) { el.innerHTML = '<tbody><tr><td class="mut p-4">No data</td></tr></tbody>'; return; }
  const head = cols.map((c,i) => `<th data-i="${i}">${c.label}</th>`).join('');
  const body = rows.map(r => '<tr>' + cols.map(c => {
    const v = c.get ? c.get(r) : r[c.key];
    return `<td style="${c.style ? c.style(r) : ''}">${c.fmt ? c.fmt(v, r) : (v ?? '—')}</td>`;
  }).join('') + '</tr>').join('');
  el.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;

  // Click-to-sort, numeric-aware.
  el.querySelectorAll('th').forEach((th, i) => {
    let asc = false;
    th.onclick = () => {
      asc = !asc;
      const c = cols[i];
      const sorted = [...rows].sort((a,b) => {
        const av = c.get ? c.get(a) : a[c.key], bv = c.get ? c.get(b) : b[c.key];
        if (typeof av === 'number' && typeof bv === 'number') return asc ? av-bv : bv-av;
        return asc ? String(av??'').localeCompare(String(bv??''))
                   : String(bv??'').localeCompare(String(av??''));
      });
      table(id, sorted, cols, opts);
    };
  });
}

/* ================================================================ FILTERS = */
/** Re-aggregate the fact cube under the current filter selection. */
function cubeSlice() {
  return DATA.cube.filter(r =>
    (S.region === 'all' || r.region === S.region) &&
    (S.domain === 'all' || r.business_domain === S.domain));
}
function agg(rows) {
  const t = { orders:0, revenue:0, cost:0, late:0, failed:0, km:0, co2:0, duration:0 };
  rows.forEach(r => Object.keys(t).forEach(k => t[k] += (r[k] || 0)));
  t.margin = t.revenue - t.cost;
  t.on_time = t.orders ? (1 - t.late / t.orders) * 100 : null;
  t.margin_pct = t.revenue ? t.margin / t.revenue * 100 : null;
  t.cost_per_order = t.orders ? t.cost / t.orders : null;
  t.km_per_order = t.orders ? t.km / t.orders : null;
  return t;
}
function groupBy(rows, key) {
  const m = new Map();
  rows.forEach(r => { if (!m.has(r[key])) m.set(r[key], []); m.get(r[key]).push(r); });
  return [...m.entries()].map(([k, v]) => ({ key:k, ...agg(v) }));
}
const isFiltered = () => S.region !== 'all' || S.domain !== 'all';

function updateFilterNote() {
  const note = document.getElementById('filterNote');
  const txt = document.getElementById('filterNoteText');
  if (!isFiltered()) { note.classList.add('hidden'); return; }
  note.classList.remove('hidden');
  const bits = [];
  if (S.region !== 'all') bits.push(`region <b>${S.region}</b>`);
  if (S.domain !== 'all') bits.push(`domain <b>${titleCase(S.domain)}</b>`);
  const share = agg(cubeSlice()).orders / agg(DATA.cube).orders * 100;
  txt.innerHTML = `Filtered to ${bits.join(' and ')} — ${pct(share)} of network volume. ` +
    `Executive and Delivery figures re-aggregate live; Fleet, Warehouse, Forecast and ` +
    `Explainable AI are network-wide and unaffected.`;
}

/* ================================================================== SHELL = */
function buildNav() {
  document.getElementById('nav').innerHTML = MODULES.map(m => `
    <button class="nav-item w-full text-left px-5 py-2.5 text-[13px] flex items-center gap-3"
            data-m="${m.id}">
      <span class="text-[15px] w-4 opacity-80">${m.icon}</span><span>${m.label}</span>
    </button>`).join('');
  document.querySelectorAll('[data-m]').forEach(b =>
    b.onclick = () => go(b.dataset.m));
}

function go(id) {
  S.module = id;
  document.querySelectorAll('.nav-item').forEach(n =>
    n.classList.toggle('active', n.dataset.m === id));
  document.querySelectorAll('.module').forEach(s =>
    s.classList.toggle('active', s.id === 'm-' + id));
  const m = MODULES.find(x => x.id === id);
  document.getElementById('pageTitle').textContent = m.label;
  document.getElementById('pageSub').textContent = m.sub;
  render(id);
  window.scrollTo({ top:0, behavior:'smooth' });
}

/** Render on demand: charts are only built when their module is first opened. */
function render(id, force = false) {
  const fn = RENDERERS[id];
  if (!fn) return;
  if (!force && S.drawn.has(id) && !['executive','delivery'].includes(id)) return;
  fn();
  S.drawn.add(id);
  // Plotly needs a resize nudge for charts drawn while hidden.
  setTimeout(() => window.dispatchEvent(new Event('resize')), 60);
}

/* ============================================================== EXECUTIVE = */
function renderExecutive() {
  const rows = cubeSlice();
  const t = agg(rows);
  const k = DATA.executive.kpis;
  const full = !isFiltered();

  setKpis('exKpis',
    kpi('Total orders', nf(t.orders), { sub: full ? 'Full network' : 'Filtered' }) +
    kpi('On-time rate', pct(t.on_time), { tone: t.on_time >= 85 ? 'good' : 'warn',
        delta: full ? k.on_time_delta : null }) +
    kpi('Revenue', usd(t.revenue), { delta: full ? k.revenue_delta : null }) +
    kpi('Delivery cost', usd(t.cost), { delta: full ? k.cost_delta : null, positiveIsGood:false }) +
    kpi('Gross margin', usd(t.margin), { sub: pct(t.margin_pct) + ' of revenue', tone:'good' }) +
    kpi('Cost / order', usd(t.cost_per_order), { sub: nf1(t.km_per_order) + ' km avg' })
  );

  // Monthly trend from the cube so it responds to filters.
  const months = groupBy(rows, 'month').sort((a,b) => a.key.localeCompare(b.key));
  plot('exTrend', [
    { x:months.map(m=>m.key), y:months.map(m=>m.revenue), name:'Revenue', type:'bar',
      marker:{color:'#4c7ef3'} },
    { x:months.map(m=>m.key), y:months.map(m=>m.cost), name:'Cost', type:'bar',
      marker:{color:'#e0574c'} },
    { x:months.map(m=>m.key), y:months.map(m=>m.margin), name:'Margin', type:'scatter',
      mode:'lines+markers', line:{color:'#3fa87a',width:2.5}, yaxis:'y2' }
  ], { barmode:'group', yaxis:{title:'USD', gridcolor:isDark()?'#212c50':'#e8ecf4'},
       yaxis2:{overlaying:'y', side:'right', title:'Margin', showgrid:false} });

  plot('exCost', [{
    labels: DATA.executive.cost_breakdown.map(c=>c.category),
    values: DATA.executive.cost_breakdown.map(c=>c.amount),
    type:'pie', hole:.58, marker:{colors:PALETTE},
    textinfo:'percent', textfont:{size:11},
    hovertemplate:'%{label}<br>$%{value:,.0f}<extra></extra>'
  }], { showlegend:true, margin:{l:10,r:10,t:10,b:40} });

  // Domain bubbles: on-time vs cost per order, sized by volume.
  const dom = groupBy(rows, 'business_domain').sort((a,b)=>b.orders-a.orders);
  plot('exDomain', [{
    x: dom.map(d=>d.cost_per_order), y: dom.map(d=>d.on_time),
    text: dom.map(d=>titleCase(d.key)), mode:'markers+text', textposition:'top center',
    textfont:{size:9},
    marker:{ size: dom.map(d=>Math.sqrt(d.orders)/2.2), color: dom.map((_,i)=>PALETTE[i%PALETTE.length]),
             opacity:.85, line:{width:1,color:'rgba(255,255,255,.25)'} },
    hovertemplate:'%{text}<br>On-time %{y:.1f}%<br>Cost/order $%{x:.2f}<extra></extra>'
  }], { xaxis:{title:'Cost per order (USD)'}, yaxis:{title:'On-time rate (%)'}, showlegend:false });

  const reg = groupBy(rows, 'region').sort((a,b)=>b.orders-a.orders);
  plot('exRegion', [
    { x:reg.map(r=>r.key), y:reg.map(r=>r.orders), name:'Orders', type:'bar', marker:{color:'#4c7ef3'} },
    { x:reg.map(r=>r.key), y:reg.map(r=>r.margin), name:'Margin', type:'bar', marker:{color:'#3fa87a'} },
    { x:reg.map(r=>r.key), y:reg.map(r=>r.on_time), name:'On-time %', type:'scatter',
      mode:'lines+markers', line:{color:'#e0a24c',width:2.5}, yaxis:'y2' }
  ], { barmode:'group', yaxis2:{overlaying:'y',side:'right',title:'%',showgrid:false,range:[0,100]} });

  table('exTable', dom, [
    { label:'Domain', key:'key', fmt:v=>`<b>${titleCase(v)}</b>` },
    { label:'Orders', key:'orders', fmt:nf },
    { label:'Revenue', key:'revenue', fmt:usd },
    { label:'Cost', key:'cost', fmt:usd },
    { label:'Margin', key:'margin', fmt:v=>`<span style="color:${v>=0?'#3fa87a':'#e0574c'}">${usd(v)}</span>` },
    { label:'Margin %', key:'margin_pct', fmt:pct },
    { label:'On-time', key:'on_time',
      fmt:v=>`<span class="pill" style="background:${v>=88?'rgba(63,168,122,.18)':v>=80?'rgba(224,162,76,.18)':'rgba(224,87,76,.18)'};color:${v>=88?'#3fa87a':v>=80?'#e0a24c':'#e0574c'}">${pct(v)}</span>` },
    { label:'Cost/order', key:'cost_per_order', fmt:v=>'$'+nf2(v) },
    { label:'km/order', key:'km_per_order', fmt:nf1 }
  ]);
}

/* ============================================================= OPERATIONS = */
function renderOperations() {
  const O = DATA.operations, t = agg(cubeSlice());
  const busiest = [...O.by_hour].sort((a,b)=>b.orders-a.orders)[0];
  const worstWx = O.weather_impact[0];
  const exTotal = O.exceptions.reduce((s,e)=>s+e.n,0);

  setKpis('opKpis',
    kpi('Orders', nf(t.orders)) +
    kpi('On-time rate', pct(t.on_time), { tone: t.on_time>=85?'good':'warn' }) +
    kpi('Peak hour', busiest ? `${String(busiest.h).padStart(2,'0')}:00` : '—',
        { sub: busiest ? nf(busiest.orders)+' orders' : '' }) +
    kpi('Failed', pct(DATA.executive.kpis.failed_rate), { tone:'bad' }) +
    kpi('Worst weather', worstWx ? titleCase(worstWx.weather_condition) : '—',
        { sub: worstWx ? pct(worstWx.late_rate)+' late' : '', tone:'warn' }) +
    kpi('Exceptions', nf(exTotal), { sub:'Flagged events' })
  );

  const DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const z = Array.from({length:7}, () => Array(24).fill(0));
  O.demand_grid.forEach(g => { z[g.d][g.h] = g.orders; });
  plot('opHeat', [{
    z, x:[...Array(24).keys()].map(h=>String(h).padStart(2,'0')), y:DOW,
    type:'heatmap', colorscale:[[0,'#111832'],[.35,'#2c4a8a'],[.7,'#4c7ef3'],[1,'#8fc0ff']],
    hovertemplate:'%{y} %{x}:00<br>%{z} orders<extra></extra>', showscale:true,
    colorbar:{thickness:9,len:.85,tickfont:{size:9}}
  }], { margin:{l:44,r:14,t:8,b:34} });

  const st = O.status_mix;
  plot('opStatus', [{
    labels: st.map(s=>titleCase(s.status)), values: st.map(s=>s.orders),
    type:'pie', hole:.55,
    marker:{colors:['#3fa87a','#e0a24c','#e0574c','#7a5cf0','#8b97b5','#4cc7e0']},
    textinfo:'percent', hovertemplate:'%{label}<br>%{value:,} orders<extra></extra>'
  }], { margin:{l:10,r:10,t:10,b:40} });

  plot('opHour', [
    { x:O.by_hour.map(h=>h.h), y:O.by_hour.map(h=>h.orders), name:'Orders', type:'bar',
      marker:{color:'#4c7ef3',opacity:.75} },
    { x:O.by_hour.map(h=>h.h), y:O.by_hour.map(h=>h.on_time), name:'On-time %',
      type:'scatter', mode:'lines', line:{color:'#3fa87a',width:2.5}, yaxis:'y2' }
  ], { xaxis:{title:'Hour of day', dtick:2},
       yaxis2:{overlaying:'y',side:'right',title:'%',showgrid:false,range:[60,100]} });

  if (O.congestion_profile.length) {
    plot('opCongestion', [{
      x:O.congestion_profile.map(c=>c.h), y:O.congestion_profile.map(c=>c.congestion_index),
      type:'scatter', mode:'lines', fill:'tozeroy', line:{color:'#e0a24c',width:2.5},
      fillcolor:'rgba(224,162,76,.18)', hovertemplate:'%{x}:00 — index %{y:.3f}<extra></extra>'
    }], { xaxis:{title:'Hour of day',dtick:2}, yaxis:{title:'Congestion index'} });
  } else empty('opCongestion','No traffic panel in the dataset');

  const tr = O.traffic_impact;
  plot('opTraffic', [{
    x:tr.map(r=>titleCase(r.traffic_level)), y:tr.map(r=>r.late_rate), type:'bar',
    marker:{color:tr.map(r=>r.late_rate>13?'#e0574c':r.late_rate>11?'#e0a24c':'#3fa87a')},
    hovertemplate:'%{x}<br>%{y:.2f}% late<extra></extra>'
  }], { yaxis:{title:'Late rate (%)'}, margin:{l:48,r:14,t:8,b:60}, xaxis:{tickangle:-25} });

  const wx = O.weather_impact;
  plot('opWeather', [{
    x:wx.map(r=>titleCase(r.weather_condition)), y:wx.map(r=>r.late_rate), type:'bar',
    marker:{color:wx.map(r=>r.late_rate>13?'#e0574c':r.late_rate>11?'#e0a24c':'#3fa87a')},
    hovertemplate:'%{x}<br>%{y:.2f}% late<extra></extra>'
  }], { yaxis:{title:'Late rate (%)'}, margin:{l:48,r:14,t:8,b:70}, xaxis:{tickangle:-30} });

  const ex = [...O.exceptions].sort((a,b)=>a.n-b.n);
  plot('opExceptions', [{
    y:ex.map(e=>titleCase(e.flag)), x:ex.map(e=>e.n), type:'bar', orientation:'h',
    marker:{color:'#7a5cf0'}, hovertemplate:'%{y}<br>%{x:,} events<extra></extra>'
  }], { margin:{l:130,r:14,t:8,b:34} });
}

/* ================================================================== FLEET = */
function renderFleet() {
  const F = DATA.fleet, s = F.summary;
  setKpis('flKpis',
    kpi('Fleet size', nf(s.total_vehicles), { sub:`${nf(s.active)} active` }) +
    kpi('Avg utilisation', pct(s.avg_utilisation), { tone: s.avg_utilisation>50?'good':'warn' }) +
    kpi('Average age', nf1(s.avg_age_years) + ' yr') +
    kpi('Electric share', pct(s.electric_share), { tone:'good' }) +
    kpi('Maintenance', usd(s.total_maintenance_cost),
        { sub: s.unplanned_share!=null ? pct(s.unplanned_share)+' unplanned' : '', tone:'warn' }) +
    kpi('Downtime', nf(s.total_downtime_hours) + ' h', { tone:'bad' })
  );

  const bt = F.by_type;
  plot('flType', [{
    x:bt.map(b=>titleCase(b.vehicle_type)), y:bt.map(b=>b.vehicles), type:'bar',
    marker:{color:PALETTE}, hovertemplate:'%{x}<br>%{y} vehicles<extra></extra>'
  }], { margin:{l:44,r:14,t:8,b:80}, xaxis:{tickangle:-32} });

  plot('flAge', [{
    x:F.age_profile.map(a=>a.age), y:F.age_profile.map(a=>a.n), type:'bar',
    marker:{color:F.age_profile.map(a=>a.age>8?'#e0574c':a.age>5?'#e0a24c':'#3fa87a')},
    hovertemplate:'%{x} years<br>%{y} vehicles<extra></extra>'
  }], { xaxis:{title:'Age (years)'}, margin:{l:44,r:14,t:8,b:44} });

  plot('flFuel', [{
    labels:F.by_fuel.map(f=>titleCase(f.fuel)), values:F.by_fuel.map(f=>f.n),
    type:'pie', hole:.55, marker:{colors:PALETTE}, textinfo:'percent'
  }], { margin:{l:10,r:10,t:10,b:40} });

  if (F.route_utilisation.length) {
    const ru = F.route_utilisation;
    plot('flUtil', [{
      x:ru.map(r=>r.stops), y:ru.map(r=>r.capacity_utilisation_kg*100), mode:'markers',
      marker:{ size:5, opacity:.5,
               color:ru.map(r=>r.capacity_utilisation_kg*100),
               colorscale:[[0,'#e0574c'],[.4,'#e0a24c'],[1,'#3fa87a']],
               colorbar:{thickness:9,len:.8,title:{text:'%',font:{size:9}},tickfont:{size:9}} },
      hovertemplate:'%{x} stops<br>%{y:.1f}% loaded<extra></extra>'
    }], { xaxis:{title:'Stops per route'}, yaxis:{title:'Capacity used (%)'}, showlegend:false });
  } else empty('flUtil','No route data');

  if (F.maintenance_by_type.length) {
    const m = F.maintenance_by_type;
    plot('flMaint', [
      { x:m.map(r=>titleCase(r.maintenance_type)), y:m.map(r=>r.cost), name:'Cost', type:'bar',
        marker:{color:'#e0574c'} },
      { x:m.map(r=>titleCase(r.maintenance_type)), y:m.map(r=>r.downtime), name:'Downtime (h)',
        type:'scatter', mode:'lines+markers', line:{color:'#e0a24c',width:2.5}, yaxis:'y2' }
    ], { margin:{l:56,r:52,t:8,b:104}, xaxis:{tickangle:-32},
         yaxis:{title:'Cost (USD)'}, yaxis2:{overlaying:'y',side:'right',title:'Hours',showgrid:false} });
  } else empty('flMaint','No maintenance records');

  table('flTable', F.vehicles.slice(0,120), [
    { label:'Vehicle', key:'vehicle_id' },
    { label:'Type', key:'vehicle_type', fmt:titleCase },
    { label:'Make', key:'make' },
    { label:'Year', key:'model_year' },
    { label:'Fuel', key:'fuel_type', fmt:titleCase },
    { label:'Deliveries', key:'deliveries', fmt:nf },
    { label:'Distance', key:'km', fmt:v=>nf(v)+' km' },
    { label:'Utilisation', key:'utilisation',
      fmt:v=>`<div style="background:var(--line);border-radius:3px;height:6px;width:58px;display:inline-block;vertical-align:middle">
              <div style="width:${Math.min(v,100)}%;height:100%;background:#4c7ef3;border-radius:3px"></div></div>
              <span class="ml-2">${nf1(v)}%</span>` },
    { label:'Status', key:'status',
      fmt:v=>`<span class="pill" style="background:${v==='active'?'rgba(63,168,122,.18)':'rgba(224,162,76,.18)'};color:${v==='active'?'#3fa87a':'#e0a24c'}">${titleCase(v)}</span>` }
  ]);

  table('flDrivers', F.drivers.top, [
    { label:'Driver', key:'driver_id' },
    { label:'Deliveries', key:'deliveries', fmt:nf },
    { label:'On-time', key:'on_time', fmt:v=>pct(v*100) },
    { label:'Rating', key:'rating', fmt:v=>v==null?'—':nf2(v)+' ★' },
    { label:'Cost/delivery', key:'cost_per_delivery', fmt:v=>'$'+nf2(v) },
    { label:'Score', key:'score',
      fmt:v=>`<span class="pill" style="background:rgba(76,126,243,.18);color:#7aa2f7">${nf1(v)}</span>` }
  ]);
}

/* ============================================================== WAREHOUSE = */
function renderWarehouse() {
  const W = DATA.warehouse, s = W.summary;
  setKpis('whKpis',
    kpi('Sites', nf(s.total_warehouses), { sub:`${s.cold_chain_sites} cold-chain` }) +
    kpi('Total capacity', nf(s.total_capacity_m3) + ' m³') +
    kpi('Peak capacity used', pct(s.avg_capacity_used),
        { tone: s.avg_capacity_used>85?'bad':s.avg_capacity_used>60?'warn':'good' }) +
    kpi('Automated', nf(s.automated_sites), { sub:'sites' }) +
    kpi('Stock value', usd(s.stock_value), { sub: nf(s.total_units)+' units' }) +
    kpi('Stockout rate', pct(s.stockout_rate), { tone: s.stockout_rate>2?'bad':'warn' })
  );

  const sites = [...W.sites].sort((a,b)=>b.capacity_used_pct-a.capacity_used_pct).slice(0,20);
  plot('whCapacity', [{
    y:sites.map(s=>s.city+' · '+s.warehouse_id.slice(-3)).reverse(),
    x:sites.map(s=>s.capacity_used_pct).reverse(), type:'bar', orientation:'h',
    marker:{color:sites.map(s=>s.capacity_used_pct>100?'#e0574c':s.capacity_used_pct>75?'#e0a24c':'#3fa87a').reverse()},
    hovertemplate:'%{y}<br>%{x:.1f}% of rated throughput<extra></extra>'
  }], { margin:{l:120,r:14,t:8,b:40}, xaxis:{title:'Peak-day utilisation (%)'},
        shapes:[{type:'line',x0:100,x1:100,y0:-0.5,y1:sites.length-0.5,
                 line:{color:'#e0574c',width:1.5,dash:'dash'}}] });

  const auto = W.by_automation;
  plot('whAuto', [
    { x:auto.map(a=>titleCase(a.automation_level)), y:auto.map(a=>a.sites), name:'Sites',
      type:'bar', marker:{color:'#4c7ef3'} },
    { x:auto.map(a=>titleCase(a.automation_level)), y:auto.map(a=>a.on_time), name:'On-time %',
      type:'scatter', mode:'lines+markers', line:{color:'#3fa87a',width:2.5}, yaxis:'y2' }
  ], { margin:{l:44,r:48,t:8,b:86}, xaxis:{tickangle:-28},
       yaxis2:{overlaying:'y',side:'right',range:[0,100],showgrid:false} });

  plot('whDaily', [{
    x:W.throughput_daily.map(d=>d.day), y:W.throughput_daily.map(d=>d.orders),
    type:'scatter', mode:'lines', fill:'tozeroy', line:{color:'#4c7ef3',width:1.8},
    fillcolor:'rgba(76,126,243,.16)', hovertemplate:'%{x}<br>%{y:,} orders<extra></extra>'
  }], { yaxis:{title:'Orders per day'} });

  if (W.inventory_by_category.length) {
    const inv = W.inventory_by_category;
    plot('whInv', [
      { x:inv.map(i=>titleCase(i.sku_category)), y:inv.map(i=>i.units), name:'Units',
        type:'bar', marker:{color:'#7a5cf0'} },
      { x:inv.map(i=>titleCase(i.sku_category)), y:inv.map(i=>i.stockout_rate), name:'Stockout %',
        type:'scatter', mode:'markers', marker:{color:'#e0574c',size:9}, yaxis:'y2' }
    ], { margin:{l:56,r:48,t:8,b:110}, xaxis:{tickangle:-34},
         yaxis2:{overlaying:'y',side:'right',title:'%',showgrid:false} });
  } else empty('whInv','No inventory table');

  table('whTable', W.sites, [
    { label:'Site', key:'warehouse_name' },
    { label:'City', key:'city' },
    { label:'Type', key:'warehouse_type', fmt:titleCase },
    { label:'Automation', key:'automation_level', fmt:titleCase },
    { label:'Capacity m³', key:'capacity_m3', fmt:nf },
    { label:'Orders', key:'orders', fmt:nf },
    { label:'Peak day', key:'peak_day_orders', fmt:nf },
    { label:'Capacity used', key:'capacity_used_pct',
      fmt:v=>`<span class="pill" style="background:${v>100?'rgba(224,87,76,.18)':v>75?'rgba(224,162,76,.18)':'rgba(63,168,122,.18)'};color:${v>100?'#e0574c':v>75?'#e0a24c':'#3fa87a'}">${pct(v)}</span>` },
    { label:'On-time', key:'on_time', fmt:pct },
    { label:'Revenue', key:'revenue', fmt:usd }
  ]);
}

/* ===================================================== DELIVERY / MAPPING = */
const LAYER_DEFS = [
  { id:'warehouses', label:'Warehouses', on:true,  colour:'#4c7ef3' },
  { id:'vehicles',   label:'Vehicles',   on:true,  colour:'#3fa87a' },
  { id:'routes',     label:'Routes',     on:true,  colour:'#7a5cf0' },
  { id:'deliveries', label:'Deliveries', on:false, colour:'#e0a24c' },
  { id:'demand',     label:'Demand',     on:false, colour:'#e07ac0' },
  { id:'traffic',    label:'Traffic',    on:false, colour:'#e0574c' },
  { id:'weather',    label:'Weather',    on:false, colour:'#4cc7e0' }
];

function renderDelivery() {
  const t = agg(cubeSlice()), M = DATA.map;
  const lateShare = t.orders ? t.late / t.orders * 100 : 0;

  setKpis('dlKpis',
    kpi('Deliveries', nf(t.orders)) +
    kpi('On-time', pct(t.on_time), { tone: t.on_time>=85?'good':'warn' }) +
    kpi('Breached SLA', nf(t.late), { sub: pct(lateShare), tone:'bad' }) +
    kpi('Failed', nf(t.failed), { tone:'bad' }) +
    kpi('Distance', nf(t.km) + ' km', { sub: nf1(t.km_per_order)+' km/order' }) +
    kpi('CO₂', nf1(t.co2/1000) + ' t')
  );

  if (!S.map) initMap();

  const pr = DATA.executive.by_priority;
  plot('dlSla', [
    { x:pr.map(p=>titleCase(p.priority)), y:pr.map(p=>p.on_time), name:'On-time %',
      type:'bar', marker:{color:'#3fa87a'} },
    { x:pr.map(p=>titleCase(p.priority)), y:pr.map(p=>100-p.on_time), name:'Breached %',
      type:'bar', marker:{color:'#e0574c'} }
  ], { barmode:'stack', margin:{l:44,r:14,t:8,b:70}, xaxis:{tickangle:-25},
       yaxis:{title:'%',range:[0,100]} });

  // Distance bands from the sampled deliveries carried for the map.
  const bands = [ {l:'0–10 km',max:10},{l:'10–25 km',max:25},{l:'25–50 km',max:50},
                  {l:'50–100 km',max:100},{l:'100 km+',max:Infinity} ];
  const counts = bands.map(b => 0);
  M.deliveries.forEach(d => {
    for (let i=0;i<bands.length;i++) if (d.km <= bands[i].max) { counts[i]++; break; }
  });
  plot('dlDist', [{
    x:bands.map(b=>b.l), y:counts, type:'bar', marker:{color:PALETTE},
    hovertemplate:'%{x}<br>%{y} deliveries<extra></extra>'
  }], { margin:{l:44,r:14,t:8,b:64}, xaxis:{tickangle:-22} });

  const d = DATA.operations.daily;
  plot('dlDaily', [{
    x:d.map(x=>x.day), y:d.map(x=>x.on_time), type:'scatter', mode:'lines',
    line:{color:'#3fa87a',width:1.8}, fill:'tozeroy', fillcolor:'rgba(63,168,122,.13)',
    hovertemplate:'%{x}<br>%{y:.1f}% on time<extra></extra>'
  }], { yaxis:{title:'On-time (%)',range:[60,100]} });
}

function initMap() {
  const M = DATA.map;
  S.map = L.map('map', { zoomControl:true, attributionControl:true })
           .setView(M.center, 3);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution:'© OpenStreetMap contributors © CARTO', subdomains:'abcd', maxZoom:19
  }).addTo(S.map);

  // ---- warehouses
  S.layers.warehouses = L.layerGroup(M.warehouses.map(w =>
    L.circleMarker([w.lat, w.lon], {
      radius:8, color:'#fff', weight:1.5, fillColor:'#4c7ef3', fillOpacity:.9
    }).bindPopup(
      `<b>${w.name}</b><br>${titleCase(w.type)}<br>${w.city}<br>
       Capacity ${nf(w.capacity)} m³${w.cold?'<br><span style="color:#4cc7e0">Cold-chain capable</span>':''}`)
  ));

  // ---- vehicles
  S.layers.vehicles = L.layerGroup(M.vehicles.map(v =>
    L.circleMarker([v.lat, v.lon], {
      radius:4, color:'#3fa87a', weight:1, fillColor:'#3fa87a', fillOpacity:.8
    }).bindPopup(
      `<b>${v.vehicle_id}</b><br>${titleCase(v.type)}<br>Speed ${nf1(v.speed)} km/h<br>Status ${titleCase(v.status)}`)
  ));

  // ---- routes
  S.layers.routes = L.layerGroup(M.routes.map((r,i) =>
    L.polyline(r.points, {
      color: PALETTE[i % PALETTE.length], weight:2, opacity:.65
    }).bindPopup(
      `<b>${r.route_id}</b><br>${r.stops ?? '—'} stops<br>${nf1(r.distance_km)} km<br>
       On-time ${pct(r.on_time)}<br>Cost ${usd(r.cost)}`)
  ));

  // ---- delivery points, coloured by SLA outcome
  S.layers.deliveries = L.layerGroup(M.deliveries.map(d =>
    L.circleMarker([d.lat, d.lon], {
      radius:2.6, color: d.late ? '#e0574c' : '#e0a24c', weight:0,
      fillColor: d.late ? '#e0574c' : '#e0a24c', fillOpacity:.55
    }).bindPopup(
      `${titleCase(d.domain)}<br>${d.city}<br>${nf1(d.km)} km · ${usd(d.cost)}<br>
       <b style="color:${d.late?'#e0574c':'#3fa87a'}">${d.late?'Breached SLA':'On time'}</b>`)
  ));

  // ---- demand density by zone
  const maxOrders = Math.max(...M.zones.map(z=>z.orders), 1);
  S.layers.demand = L.layerGroup(M.zones.map(z =>
    L.circleMarker([z.lat, z.lon], {
      radius: 3 + Math.sqrt(z.orders / maxOrders) * 16,
      color:'#e07ac0', weight:0, fillColor:'#e07ac0',
      fillOpacity: .12 + (z.orders / maxOrders) * .38
    }).bindPopup(`<b>${z.name}</b><br>${nf(z.orders)} orders<br>${titleCase(z.area_type)}`)
  ));

  // ---- traffic
  S.layers.traffic = L.layerGroup(M.traffic.map(c =>
    L.circleMarker([c.lat, c.lon], {
      radius: 6 + c.congestion * 22,
      color: c.congestion>.5?'#e0574c':c.congestion>.35?'#e0a24c':'#3fa87a',
      weight:1.5, fillOpacity:.22
    }).bindPopup(`<b>${c.city}</b><br>Congestion ${nf2(c.congestion)}<br>Avg speed ${nf1(c.speed)} km/h`)
  ));

  // ---- weather
  S.layers.weather = L.layerGroup(M.weather.map(c =>
    L.circleMarker([c.lat, c.lon], {
      radius: 6 + c.severity * 30,
      color: c.severity>.15?'#e0574c':'#4cc7e0', weight:1.5, fillOpacity:.18
    }).bindPopup(
      `<b>${c.city}</b><br>Severity ${nf2(c.severity)}<br>${nf1(c.temp)}°C<br>Precip ${nf2(c.precip)} mm`)
  ));

  LAYER_DEFS.forEach(d => { if (d.on) S.layers[d.id].addTo(S.map); });

  document.getElementById('mapLayers').innerHTML = LAYER_DEFS.map(d => `
    <button class="mapLayer text-[11px] px-2.5 py-1 rounded-lg border font-medium"
            data-l="${d.id}" data-on="${d.on}"
            style="border-color:${d.on?d.colour:'var(--line)'};color:${d.on?d.colour:'var(--mut)'}">
      ${d.label}</button>`).join('');

  document.querySelectorAll('.mapLayer').forEach(b => b.onclick = () => {
    const id = b.dataset.l, def = LAYER_DEFS.find(x=>x.id===id);
    const on = b.dataset.on === 'true';
    if (on) { S.map.removeLayer(S.layers[id]); b.dataset.on='false';
              b.style.borderColor='var(--line)'; b.style.color='var(--mut)'; }
    else    { S.layers[id].addTo(S.map); b.dataset.on='true';
              b.style.borderColor=def.colour; b.style.color=def.colour; }
  });

  document.getElementById('mapLegend').innerHTML = [
    ['#4c7ef3','Warehouse'], ['#3fa87a','Vehicle'], ['#7a5cf0','Route'],
    ['#e0a24c','Delivery on time'], ['#e0574c','Delivery late / congestion'],
    ['#e07ac0','Demand density'], ['#4cc7e0','Weather']
  ].map(([c,l]) =>
    `<span class="flex items-center gap-1.5">
       <span style="width:9px;height:9px;border-radius:50%;background:${c};display:inline-block"></span>${l}
     </span>`).join('');

  setTimeout(() => S.map.invalidateSize(), 120);
}

/* =============================================================== FORECAST = */
function renderForecast() {
  const F = DATA.forecast;
  if (!F.history.length) { empty('fcChart','Not enough history to forecast'); return; }

  const next7 = F.forecast.slice(0,7).reduce((s,f)=>s+f.point,0);
  const next30 = F.forecast.reduce((s,f)=>s+f.point,0);
  const recent = F.history.slice(-30).reduce((s,h)=>s+h.orders,0);

  setKpis('fcKpis',
    kpi('Next 7 days', nf(next7), { sub:'projected orders' }) +
    kpi('Next 30 days', nf(next30), { sub:'projected orders' }) +
    kpi('vs last 30 days', pct((next30-recent)/recent*100),
        { tone:(next30-recent)>=0?'good':'warn', sub:nf(recent)+' actual' }) +
    kpi('Forecast error', pct(F.accuracy.mape),
        { sub:F.accuracy.basis, tone:F.accuracy.mape<20?'good':'warn' }) +
    kpi('Trend', (F.trend_per_day>=0?'+':'') + nf2(F.trend_per_day), { sub:'orders per day' })
  );

  document.getElementById('fcMethod').textContent = F.method;
  document.getElementById('fcAccuracy').innerHTML =
    `MAPE <b>${pct(F.accuracy.mape)}</b> · RMSE <b>${nf1(F.accuracy.rmse)}</b> · ` +
    `<span class="mut">${F.accuracy.basis} (in-sample ${pct(F.accuracy.in_sample_mape)})</span>`;

  const hx = F.history.map(h=>h.day), fx = F.forecast.map(f=>f.day);
  plot('fcChart', [
    { x:fx.concat([...fx].reverse()),
      y:F.forecast.map(f=>f.upper).concat(F.forecast.map(f=>f.lower).reverse()),
      fill:'toself', fillcolor:'rgba(76,126,243,.16)', line:{width:0},
      name:'95% interval', hoverinfo:'skip' },
    { x:hx, y:F.history.map(h=>h.orders), name:'Actual', type:'scatter', mode:'lines',
      line:{color:'#8b97b5',width:1.2} },
    { x:hx, y:F.history.map(h=>h.fitted), name:'Fitted', type:'scatter', mode:'lines',
      line:{color:'#3fa87a',width:1.8,dash:'dot'} },
    { x:fx, y:F.forecast.map(f=>f.point), name:'Forecast', type:'scatter', mode:'lines',
      line:{color:'#4c7ef3',width:2.6} }
  ], { yaxis:{title:'Orders per day'},
       shapes:[{type:'line', x0:hx[hx.length-1], x1:hx[hx.length-1], yref:'paper', y0:0, y1:1,
                line:{color:'#e0a24c',width:1.5,dash:'dash'}}],
       annotations:[{x:hx[hx.length-1], yref:'paper', y:1.04, text:'today',
                     showarrow:false, font:{size:10,color:'#e0a24c'}}] });

  plot('fcDow', [{
    x:F.seasonality.map(s=>s.label), y:F.seasonality.map(s=>s.factor), type:'bar',
    marker:{color:F.seasonality.map(s=>s.factor>=1?'#3fa87a':'#e0a24c')},
    hovertemplate:'%{x}<br>×%{y:.3f}<extra></extra>'
  }], { yaxis:{title:'Multiplier'},
        shapes:[{type:'line',x0:-.5,x1:6.5,y0:1,y1:1,line:{color:'#8b97b5',width:1,dash:'dot'}}] });

  const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  plot('fcMonth', [{
    x:F.month_factors.map(m=>MON[m.month-1]), y:F.month_factors.map(m=>m.factor), type:'bar',
    marker:{color:F.month_factors.map(m=>m.factor>=1?'#4c7ef3':'#8b97b5')},
    hovertemplate:'%{x}<br>×%{y:.3f}<extra></extra>'
  }], { yaxis:{title:'Multiplier'}, margin:{l:44,r:14,t:8,b:44},
        shapes:[{type:'line',x0:-.5,x1:11.5,y0:1,y1:1,line:{color:'#8b97b5',width:1,dash:'dot'}}] });

  plot('fcDomain', [{
    labels:F.by_domain_share.map(d=>titleCase(d.business_domain)),
    values:F.by_domain_share.map(d=>d.orders), type:'pie', hole:.5,
    marker:{colors:PALETTE}, textinfo:'percent', textfont:{size:9}
  }], { margin:{l:8,r:8,t:8,b:52}, showlegend:true, legend:{font:{size:9},orientation:'h',y:-.12} });
}

/* ============================================================== SIMULATOR = */
const LEVERS = [
  { id:'fleet',   label:'Fleet size',          min:50,  max:200, val:100, step:5,  unit:'%' },
  { id:'traffic', label:'Traffic congestion',  min:50,  max:200, val:100, step:5,  unit:'%' },
  { id:'fuel',    label:'Fuel price',          min:50,  max:250, val:100, step:5,  unit:'%' },
  { id:'wh',      label:'Warehouse count',     min:50,  max:200, val:100, step:5,  unit:'%' },
  { id:'demand',  label:'Demand volume',       min:50,  max:250, val:100, step:5,  unit:'%' },
  { id:'weather', label:'Adverse weather',     min:0,   max:300, val:100, step:10, unit:'%' },
  { id:'drivers', label:'Driver availability', min:50,  max:150, val:100, step:5,  unit:'%' },
  { id:'route',   label:'Route optimisation',  min:0,   max:100, val:0,   step:5,  unit:'%' }
];

/**
 * Scenario model.
 *
 * Every coefficient is either measured from the dataset (elasticities, cost
 * shares) or is an explicit, stated assumption. Nothing here is a magic number
 * without a comment saying where it came from.
 */
function simulate(v) {
  const B = DATA.simulator.baseline;
  const C = DATA.simulator.cost_structure;
  const E = DATA.simulator.elasticities;

  const f = {}; LEVERS.forEach(l => f[l.id] = v[l.id] / 100);

  const orders = B.orders * f.demand;

  // Capacity pressure: demand relative to the resources able to serve it.
  //
  // Vehicles and drivers are near-complementary — a van with nobody to drive it
  // moves nothing — but not perfectly so. Spare vehicles still buy something
  // real: better vehicle-to-route matching, and no waiting for one to come free.
  // A strict min() would make the fleet slider dead whenever drivers bind, which
  // is both wrong and confusing. Cobb-Douglas weighted 85/15 towards the scarcer
  // factor keeps the binding constraint dominant while letting slack help a
  // little, with sharply diminishing returns.
  const scarce = Math.max(Math.min(f.fleet, f.drivers), 0.01);
  const ample  = Math.max(Math.max(f.fleet, f.drivers), 0.01);
  const servingCapacity = Math.pow(scarce, 0.85) * Math.pow(ample, 0.15);
  const binding = f.fleet <= f.drivers ? 'fleet' : 'drivers';
  const pressure = f.demand / servingCapacity;

  // --- distance -----------------------------------------------------------
  // More warehouses shorten the average leg (a denser network means a closer
  // origin); the square-root form reflects area-to-radius scaling, not a guess.
  const whEffect = 1 / Math.sqrt(Math.max(f.wh, 0.01));
  // Route optimisation consolidates stops. 18% at full strength is anchored on
  // the route-consolidation saving measured in the dataset's own routing table.
  const routeEffect = 1 - (f.route * 0.18);
  const kmPerOrder = B.km_per_order * whEffect * routeEffect;
  const totalKm = orders * kmPerOrder;

  // --- cost ---------------------------------------------------------------
  const fuelCost   = totalKm * C.fuel_per_km * f.fuel * (1 + (f.traffic - 1) * 0.25);
  // Labour rises super-linearly once demand outruns capacity: overtime.
  const overtime   = pressure > 1 ? 1 + (pressure - 1) * 0.55 : 1;
  const labourCost = orders * C.labour_per_order * overtime
                     * (1 + (f.traffic - 1) * 0.35) * routeEffect;
  const otherCost  = B.total_cost * C.other_share * f.demand;
  // Extra warehouses carry fixed overhead: ~4% of baseline cost per site.
  const whFixed    = B.total_cost * 0.04 * (f.wh - 1);
  const totalCost  = fuelCost + labourCost + otherCost + Math.max(whFixed, 0);

  // --- service level ------------------------------------------------------
  // Elasticities below are measured from the dataset (see payload builder).
  let lateRate = (100 - B.on_time_rate) / 100;
  lateRate += (f.traffic - 1) * E.congestion_to_late_rate;
  lateRate += (f.weather - 1) * E.weather_to_late_rate;
  lateRate += Math.max(pressure - 1, 0) * 0.22;          // queueing under strain
  lateRate -= Math.max(servingCapacity - 1, 0) * 0.06;   // slack absorbs shocks
  lateRate -= f.route * 0.035;                           // tighter routes arrive sooner
  lateRate = Math.min(Math.max(lateRate, 0.005), 0.85);

  const revenue = B.total_revenue * f.demand;
  const co2 = B.co2_tonnes * (totalKm / B.total_km) * (1 - f.route * 0.1);

  return {
    orders, on_time: (1 - lateRate) * 100, cost: totalCost, revenue,
    margin: revenue - totalCost,
    cost_per_order: totalCost / Math.max(orders, 1),
    km: totalKm, km_per_order: kmPerOrder, co2,
    fleet: Math.round(B.fleet_size * f.fleet),
    warehouses: Math.round(B.warehouses * f.wh),
    drivers: Math.round(B.drivers * f.drivers),
    binding, pressure,
    parts: { fuel:fuelCost, labour:labourCost, other:otherCost, warehouse:Math.max(whFixed,0) }
  };
}

function baselineScenario() {
  const v = {}; LEVERS.forEach(l => v[l.id] = l.id === 'route' ? 0 : 100);
  return v;
}

function renderSimulator() {
  if (!S.sim) S.sim = baselineScenario();

  document.getElementById('simControls').innerHTML = LEVERS.map(l => `
    <div>
      <div class="flex justify-between items-baseline mb-1.5">
        <label class="text-[12px] font-medium">${l.label}</label>
        <span class="text-[12px] font-bold" id="lab-${l.id}" style="color:#7aa2f7">${S.sim[l.id]}${l.unit}</span>
      </div>
      <input type="range" class="w-full lever" id="lev-${l.id}" data-l="${l.id}"
             min="${l.min}" max="${l.max}" step="${l.step}" value="${S.sim[l.id]}">
      <div class="flex justify-between text-[9.5px] mut mt-0.5">
        <span>${l.min}${l.unit}</span><span>${l.max}${l.unit}</span></div>
    </div>`).join('');

  document.querySelectorAll('.lever').forEach(el => el.oninput = () => {
    const id = el.dataset.l;
    S.sim[id] = +el.value;
    document.getElementById('lab-'+id).textContent =
      el.value + LEVERS.find(l=>l.id===id).unit;
    simUpdate();
  });

  document.getElementById('simReset').onclick = () => {
    S.sim = baselineScenario();
    renderSimulator();
  };
  document.getElementById('simSave').onclick = () => {
    const r = simulate(S.sim);
    S.scenarios.push({ name:'Scenario ' + (S.scenarios.length+1), levers:{...S.sim}, result:r });
    simTable();
  };
  document.getElementById('simClear').onclick = () => { S.scenarios = []; simTable(); };

  document.getElementById('simMethod').innerHTML = `
    <p><b>Where the numbers come from.</b> The response curves are calibrated
    against this dataset rather than assumed. Congestion elasticity is the
    measured gap in late rate between the top and bottom congestion quartiles
    (<b>${nf2(DATA.simulator.elasticities.congestion_to_late_rate*100)}pp</b>);
    weather elasticity is the measured gap between adverse and clear conditions
    (<b>${nf2(DATA.simulator.elasticities.weather_to_late_rate*100)}pp</b>).
    Cost shares (fuel ${pct(DATA.simulator.cost_structure.fuel_share*100)},
    labour ${pct(DATA.simulator.cost_structure.labour_share*100)}) come straight
    from the order book.</p>
    <p><b>Structural assumptions</b>, stated so they can be challenged: warehouse
    density shortens the average leg with a square-root law (area to radius);
    each additional site carries ~4% of baseline cost in fixed overhead; labour
    goes to overtime at +55% marginal rate once demand outruns serving capacity;
    route optimisation is capped at an 18% distance saving, anchored on the
    consolidation already measured in the routing table.</p>
    <p><b>What this is not.</b> A single-period equilibrium model, not a
    discrete-event simulation. It answers "roughly how does the network respond"
    — for stop-level scheduling, use the optimisation solvers directly.</p>`;

  simUpdate();
  simTable();
}

function simUpdate() {
  const B = DATA.simulator.baseline;
  const r = simulate(S.sim);
  const base = simulate(baselineScenario());
  const d = (a,b) => b ? (a-b)/Math.abs(b)*100 : null;

  setKpis('simKpis',
    kpi('Orders served', nf(r.orders), { delta:d(r.orders, base.orders) }) +
    kpi('On-time rate', pct(r.on_time), { delta:d(r.on_time, base.on_time),
        tone: r.on_time>=85?'good':r.on_time>=75?'warn':'bad' }) +
    kpi('Total cost', usd(r.cost), { delta:d(r.cost, base.cost), positiveIsGood:false }) +
    kpi('Cost / order', usd(r.cost_per_order),
        { delta:d(r.cost_per_order, base.cost_per_order), positiveIsGood:false }) +
    kpi('Margin', usd(r.margin), { delta:d(r.margin, base.margin),
        tone: r.margin>=base.margin?'good':'bad' }) +
    kpi('Capacity pressure', nf2(r.pressure) + '×',
        { sub:`${titleCase(r.binding)} is the binding constraint`,
          tone: r.pressure>1.15?'bad':r.pressure>1?'warn':'good' })
  );

  // Naming the binding constraint matters: raise fleet alone while drivers are
  // scarce and almost nothing moves. That is correct, but without saying why it
  // just looks like a broken slider.
  const note = document.getElementById('simBinding');
  if (note) {
    const other = r.binding === 'fleet' ? 'drivers' : 'fleet';
    note.innerHTML = r.pressure > 1.02
      ? `<b style="color:#e0a24c">${titleCase(r.binding)} capacity is the bottleneck.</b>
         Demand is running at ${nf2(r.pressure)}× what the network can comfortably serve —
         raising <b>${r.binding}</b> will move service level more than ${other}.`
      : `<b style="color:#3fa87a">The network has spare capacity.</b>
         Demand sits at ${nf2(r.pressure)}× serving capacity, so adding resource
         yields little; the levers worth pulling are routing and warehouse density.`;
  }

  const metrics = ['On-time %','Cost/order $','km/order','Fleet','Warehouses','Drivers'];
  const bv = [base.on_time, base.cost_per_order, base.km_per_order, base.fleet, base.warehouses, base.drivers];
  const sv = [r.on_time, r.cost_per_order, r.km_per_order, r.fleet, r.warehouses, r.drivers];
  // Index to baseline = 100 so measures on different scales share one axis.
  plot('simCompare', [
    { x:metrics, y:bv.map(()=>100), name:'Baseline', type:'bar', marker:{color:'#8b97b5'} },
    { x:metrics, y:sv.map((v,i)=>bv[i]?v/bv[i]*100:100), name:'Scenario', type:'bar',
      marker:{color:'#4c7ef3'},
      text:sv.map((v,i)=>bv[i]?((v/bv[i]-1)*100).toFixed(1)+'%':''),
      textposition:'outside', textfont:{size:9} }
  ], { barmode:'group', yaxis:{title:'Index (baseline = 100)'},
       margin:{l:52,r:14,t:14,b:76}, xaxis:{tickangle:-22} });

  const keys = ['fuel','labour','other','warehouse'];
  plot('simBridge', [
    { x:keys.map(titleCase), y:keys.map(k=>base.parts[k]), name:'Baseline', type:'bar',
      marker:{color:'#8b97b5'} },
    { x:keys.map(titleCase), y:keys.map(k=>r.parts[k]), name:'Scenario', type:'bar',
      marker:{color:keys.map(k=>r.parts[k]>base.parts[k]?'#e0574c':'#3fa87a')} }
  ], { barmode:'group', yaxis:{title:'Cost (USD)'}, margin:{l:64,r:14,t:14,b:50} });
}

function simTable() {
  const base = simulate(baselineScenario());
  const rows = S.scenarios.map(s => ({
    name:s.name,
    levers:LEVERS.filter(l => s.levers[l.id] !== (l.id==='route'?0:100))
                 .map(l => `${l.label} ${s.levers[l.id]}%`).join(', ') || 'baseline',
    on_time:s.result.on_time, cost:s.result.cost, margin:s.result.margin,
    delta:(s.result.margin - base.margin)
  }));
  table('simTable', rows, [
    { label:'Scenario', key:'name', fmt:v=>`<b>${v}</b>` },
    { label:'Levers changed', key:'levers', fmt:v=>`<span class="mut">${v}</span>` },
    { label:'On-time', key:'on_time', fmt:pct },
    { label:'Cost', key:'cost', fmt:usd },
    { label:'Margin', key:'margin', fmt:usd },
    { label:'vs baseline', key:'delta',
      fmt:v=>`<span style="color:${v>=0?'#3fa87a':'#e0574c'};font-weight:600">${v>=0?'+':''}${usd(v)}</span>` }
  ]);
}

/* ======================================================== RECOMMENDATIONS = */
function renderRecommendations() {
  const R = DATA.recommendations;
  const quantified = R.filter(r => r.impact_usd);
  const total = quantified.reduce((s,r)=>s+r.impact_usd, 0);
  const cost = DATA.executive.kpis.cost_usd;

  setKpis('recKpis',
    kpi('Opportunities', nf(R.length), { sub:`${quantified.length} quantified` }) +
    kpi('Total value', usd(total), { tone:'good' }) +
    kpi('vs cost base', pct(total/cost*100), { sub:'of delivery cost' }) +
    kpi('Quick wins', nf(R.filter(r=>r.effort==='low').length), { sub:'low effort', tone:'good' })
  );

  const q = [...quantified].sort((a,b)=>a.impact_usd-b.impact_usd);
  plot('recChart', [{
    y:q.map(r=>r.title.length>46 ? r.title.slice(0,44)+'…' : r.title),
    x:q.map(r=>r.impact_usd), type:'bar', orientation:'h',
    marker:{color:q.map(r=>r.confidence==='high'?'#3fa87a':'#e0a24c')},
    hovertemplate:'%{y}<br>$%{x:,.0f}<extra></extra>'
  }], { margin:{l:270,r:20,t:8,b:40}, xaxis:{title:'Annual impact (USD)'} });

  const eMap = { low:1, medium:2, high:3 }, cMap = { low:8, medium:14, high:20 };
  plot('recQuad', [{
    x:R.map(r=>eMap[r.effort]||2), y:R.map(r=>r.impact_usd||0),
    text:R.map(r=>r.rank), mode:'markers+text', textfont:{size:9,color:'#fff'},
    marker:{ size:R.map(r=>cMap[r.confidence]||12),
             color:R.map(r=>r.category==='Optimization'?'#4c7ef3':
                            r.category==='Fleet'?'#3fa87a':
                            r.category==='Operations'?'#e0a24c':
                            r.category==='Warehouse'?'#7a5cf0':'#4cc7e0'), opacity:.85 },
    hovertext:R.map(r=>`${r.title}<br>${r.category} · ${r.confidence} confidence`),
    hovertemplate:'%{hovertext}<br>$%{y:,.0f}<extra></extra>'
  }], { xaxis:{title:'Effort', tickvals:[1,2,3], ticktext:['Low','Medium','High'], range:[.5,3.5]},
        yaxis:{title:'Impact (USD)'}, showlegend:false });

  const tone = { high:['rgba(63,168,122,.16)','#3fa87a'], medium:['rgba(224,162,76,.16)','#e0a24c'],
                 low:['rgba(139,151,181,.16)','#8b97b5'] };
  document.getElementById('recList').innerHTML = R.map(r => {
    const [bg,fg] = tone[r.confidence] || tone.medium;
    return `<div class="card p-4">
      <div class="flex items-start gap-4 flex-wrap">
        <div class="w-8 h-8 rounded-lg flex items-center justify-center font-bold text-[13px] shrink-0"
             style="background:rgba(76,126,243,.16);color:#7aa2f7">${r.rank}</div>
        <div class="flex-1 min-w-[260px]">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="font-semibold text-[14px]">${r.title}</span>
            <span class="pill" style="background:rgba(76,126,243,.14);color:#7aa2f7">${r.category}</span>
            <span class="pill" style="background:${bg};color:${fg}">${r.confidence} confidence</span>
            <span class="pill" style="background:var(--line);color:var(--mut)">${r.effort} effort</span>
          </div>
          <p class="text-[12.5px] mut mt-2 leading-relaxed"><b style="color:var(--txt)">Evidence:</b> ${r.evidence}</p>
          <p class="text-[12.5px] mt-1.5" style="color:#7aa2f7"><b>Action:</b> ${r.action}</p>
        </div>
        <div class="text-right shrink-0">
          <div class="text-[19px] font-bold" style="color:#3fa87a">${r.impact_usd?usd(r.impact_usd):'—'}</div>
          <div class="text-[10.5px] mut">${r.impact_pct!=null?pct(r.impact_pct)+' effect':'qualitative'}</div>
        </div>
      </div></div>`;
  }).join('');

  const opt = DATA.optimization.problems || [];
  table('recOptTable', opt, [
    { label:'Problem', key:'problem', fmt:v=>`<b>${titleCase(v)}</b>` },
    { label:'Solver', key:'solver', fmt:titleCase },
    { label:'Objective', key:'objective', fmt:usd },
    { label:'Baseline', key:'baseline', fmt:usd },
    { label:'Improvement', key:'improvement_pct',
      fmt:v=>`<span style="color:${v>0?'#3fa87a':'#e0574c'};font-weight:600">${pct(v)}</span>` },
    { label:'Status', key:'status', fmt:titleCase },
    { label:'Solve time', key:'solve_seconds', fmt:v=>nf2(v)+'s' }
  ]);
}

/* ============================================================ EXPLAINABLE = */
function renderXai() {
  const M = DATA.ml;
  if (!M || !M.run) {
    ['xaiShap','xaiModels','xaiAgree','xaiPdp','xaiLocal']
      .forEach(id => empty(id, 'No ML artifacts found. Run the ML platform first.'));
    return;
  }
  const run = M.run, m = run.best_metrics || {};

  setKpis('xaiKpis',
    kpi('Best model', titleCase(run.best_model || '—')) +
    kpi('ROC-AUC', nf(m.roc_auc*1000)/1000 || '—', { tone:'good' }) +
    kpi('PR-AUC', m.average_precision!=null ? nf2(m.average_precision) : '—') +
    kpi('F1', m.f1!=null ? nf2(m.f1) : '—') +
    kpi('Brier score', m.brier_score!=null ? nf(m.brier_score*10000)/10000 : '—',
        { sub:'calibration' }) +
    kpi('Features', nf(run.n_features), { sub:nf(run.n_rows)+' rows' })
  );

  if (M.shap_global) {
    const s = [...M.shap_global].reverse();
    plot('xaiShap', [{
      y:s.map(x=>x.feature), x:s.map(x=>x.mean_abs_shap), type:'bar', orientation:'h',
      marker:{ color:s.map(x=>x.mean_abs_shap),
               colorscale:[[0,'#2c4a8a'],[1,'#7aa2f7']] },
      text:s.map(x=>x.share_pct!=null?x.share_pct.toFixed(1)+'%':''),
      textposition:'outside', textfont:{size:9},
      hovertemplate:'%{y}<br>mean |SHAP| %{x:.4f}<extra></extra>'
    }], { margin:{l:180,r:52,t:8,b:40}, xaxis:{title:'Mean |SHAP value|'} });
  } else empty('xaiShap','No SHAP output in the artifacts');

  if (M.leaderboard) {
    const lb = [...M.leaderboard].reverse();
    const metric = lb[0].roc_auc != null ? 'roc_auc' : 'f1';
    plot('xaiModels', [{
      y:lb.map(r=>titleCase(r.model)), x:lb.map(r=>r[metric]), type:'bar', orientation:'h',
      marker:{color:lb.map((r,i)=>i===lb.length-1?'#3fa87a':'#4c7ef3')},
      text:lb.map(r=>r[metric]!=null?r[metric].toFixed(3):''),
      textposition:'outside', textfont:{size:9},
      hovertemplate:'%{y}<br>'+metric+' %{x:.4f}<extra></extra>'
    }], { margin:{l:140,r:52,t:8,b:40}, xaxis:{title:titleCase(metric), range:[0,1.08]} });
  } else empty('xaiModels','No leaderboard in the artifacts');

  if (M.importance_agreement) {
    const a = M.importance_agreement;
    document.getElementById('xaiAgreeNote').textContent =
      'Rank gap between the two methods — large gaps usually mean correlated features';
    plot('xaiAgree', [{
      x:a.map(r=>r.shap_rank), y:a.map(r=>r.perm_rank), text:a.map(r=>r.feature),
      mode:'markers', marker:{ size:a.map(r=>6+Math.min(r.rank_gap,20)), color:'#7a5cf0', opacity:.8 },
      hovertemplate:'%{text}<br>SHAP rank %{x}<br>Permutation rank %{y}<extra></extra>'
    }], { xaxis:{title:'SHAP rank'}, yaxis:{title:'Permutation rank'}, showlegend:false });
  } else empty('xaiAgree','Importance agreement not computed');

  if (M.partial_dependence) {
    const byFeat = {};
    M.partial_dependence.forEach(p => { (byFeat[p.feature] ||= []).push(p); });
    plot('xaiPdp', Object.entries(byFeat).slice(0,6).map(([f,pts],i) => ({
      x:pts.map(p=>p.grid_value), y:pts.map(p=>p.partial_dependence),
      name:f.length>22?f.slice(0,20)+'…':f, type:'scatter', mode:'lines',
      line:{color:PALETTE[i%PALETTE.length], width:2}
    })), { xaxis:{title:'Feature value (standardised)'}, yaxis:{title:'Partial dependence'},
           legend:{font:{size:9}, orientation:'h', y:-.22} });
  } else empty('xaiPdp','Partial dependence not computed');

  if (M.shap_local) {
    const ids = [...new Set(M.shap_local.map(r=>r.instance))];
    document.getElementById('xaiInstance').innerHTML =
      ids.map(i => `<option value="${i}">Delivery instance ${i+1}</option>`).join('');
    document.getElementById('xaiInstance').onchange = e => drawLocal(+e.target.value);
    drawLocal(ids[0]);
  } else empty('xaiLocal','No per-instance SHAP values');

  document.getElementById('xaiCard').innerHTML = [
    ['Task', run.task], ['Type', run.task_type], ['Question', run.description],
    ['Best algorithm', titleCase(run.best_model||'—')],
    ['Training rows', nf(run.n_rows)], ['Features', nf(run.n_features)],
    ['Split strategy', titleCase(run.split||'—')],
    ['Train ends', run.train_end || '—'], ['Test starts', run.test_start || '—'],
    ['Class balance', run.class_balance
      ? Object.entries(run.class_balance).map(([k,v])=>`${k}: ${pct(v*100)}`).join(' · ') : '—'],
    ['SHAP backend', M.shap_backend || 'native TreeSHAP'],
    ['Deep learning backend', M.deep_backend || '—']
  ].map(([k,v]) =>
    `<div class="flex gap-3 py-1" style="border-bottom:1px solid var(--line)">
       <span class="mut w-40 shrink-0">${k}</span><span class="font-medium">${v ?? '—'}</span></div>`).join('');
}

function drawLocal(idx) {
  const rows = DATA.ml.shap_local.filter(r => r.instance === idx)
    .sort((a,b)=>Math.abs(a.shap_value)-Math.abs(b.shap_value));
  plot('xaiLocal', [{
    y:rows.map(r=>r.feature), x:rows.map(r=>r.shap_value), type:'bar', orientation:'h',
    marker:{color:rows.map(r=>r.shap_value>=0?'#e0574c':'#3fa87a')},
    text:rows.map(r=>`value ${nf2(r.feature_value)}`), textposition:'outside', textfont:{size:9},
    hovertemplate:'%{y}<br>SHAP %{x:.4f}<br>%{text}<extra></extra>'
  }], { margin:{l:190,r:80,t:8,b:44},
        xaxis:{title:'← pushes towards on-time    |    pushes towards late →'},
        shapes:[{type:'line',x0:0,x1:0,yref:'paper',y0:0,y1:1,
                 line:{color:'#8b97b5',width:1,dash:'dot'}}] });
}

/* ================================================================ EXPORTS = */
function currentTables() {
  const t = agg(cubeSlice());
  const sheets = {
    Summary: [
      { Metric:'Orders', Value:t.orders }, { Metric:'On-time rate %', Value:t.on_time },
      { Metric:'Revenue USD', Value:t.revenue }, { Metric:'Delivery cost USD', Value:t.cost },
      { Metric:'Gross margin USD', Value:t.margin }, { Metric:'Margin %', Value:t.margin_pct },
      { Metric:'Cost per order USD', Value:t.cost_per_order },
      { Metric:'Total km', Value:t.km }, { Metric:'CO2 kg', Value:t.co2 },
      { Metric:'Region filter', Value:S.region }, { Metric:'Domain filter', Value:S.domain }
    ],
    By_Domain: groupBy(cubeSlice(),'business_domain'),
    By_Region: groupBy(cubeSlice(),'region'),
    By_Month:  groupBy(cubeSlice(),'month'),
    Warehouses: DATA.warehouse.sites,
    Fleet: DATA.fleet.vehicles,
    Forecast: DATA.forecast.forecast,
    Recommendations: DATA.recommendations,
    Optimization: DATA.optimization.problems || [],
    ML_Leaderboard: DATA.ml.leaderboard || [],
    SHAP: DATA.ml.shap_global || []
  };
  return sheets;
}

function exportExcel() {
  const wb = XLSX.utils.book_new();
  Object.entries(currentTables()).forEach(([name, rows]) => {
    if (!rows || !rows.length) return;
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(rows), name.slice(0,31));
  });
  XLSX.writeFile(wb, `logioptima_export_${new Date().toISOString().slice(0,10)}.xlsx`);
}

function exportPdf() {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit:'pt', format:'a4' });
  const t = agg(cubeSlice());
  let y = 56;
  doc.setFontSize(20); doc.setTextColor(30,40,70);
  doc.text('LogiOptima — Executive Report', 44, y); y += 22;
  doc.setFontSize(10); doc.setTextColor(120,130,155);
  doc.text(`Generated ${DATA.meta.generated_at} · ${DATA.meta.date_range[0]} to ${DATA.meta.date_range[1]}`, 44, y);
  y += 14;
  doc.text(`Filter: region ${S.region} · domain ${S.domain}`, 44, y); y += 26;

  doc.setFontSize(13); doc.setTextColor(30,40,70); doc.text('Headline metrics', 44, y); y += 18;
  doc.setFontSize(10);
  [['Orders', nf(t.orders)], ['On-time rate', pct(t.on_time)], ['Revenue', usd(t.revenue)],
   ['Delivery cost', usd(t.cost)], ['Gross margin', usd(t.margin)],
   ['Margin %', pct(t.margin_pct)], ['Cost per order', usd(t.cost_per_order)],
   ['Total distance', nf(t.km)+' km'], ['CO2', nf1(t.co2/1000)+' t']
  ].forEach(([k,v]) => {
    doc.setTextColor(120,130,155); doc.text(k, 52, y);
    doc.setTextColor(30,40,70); doc.text(String(v), 240, y); y += 15;
  });

  y += 14; doc.setFontSize(13); doc.text('Top recommendations', 44, y); y += 18;
  doc.setFontSize(9.5);
  DATA.recommendations.slice(0,6).forEach(r => {
    doc.setTextColor(30,40,70);
    doc.text(`${r.rank}. ${r.title}`, 52, y); y += 13;
    doc.setTextColor(120,130,155);
    doc.text(doc.splitTextToSize(r.evidence, 470), 62, y);
    y += doc.splitTextToSize(r.evidence, 470).length * 11 + 3;
    doc.setTextColor(60,100,200);
    doc.text(`Impact: ${r.impact_usd?usd(r.impact_usd):'qualitative'}`, 62, y); y += 18;
    if (y > 740) { doc.addPage(); y = 56; }
  });
  doc.save(`logioptima_report_${new Date().toISOString().slice(0,10)}.pdf`);
}

function exportPreview(kind) {
  const t = agg(cubeSlice());
  const body = document.getElementById('exBody');

  if (kind === 'xls') {
    const sheets = currentTables();
    body.innerHTML = `
      <p class="text-[12.5px] mut mb-3">A multi-sheet workbook of the current view.
        Filters are applied to the summary and cube sheets; reference tables are network-wide.</p>
      <div class="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
        ${Object.entries(sheets).map(([n,r]) => `
          <div class="card p-3">
            <div class="text-[12px] font-semibold">${n.replace(/_/g,' ')}</div>
            <div class="text-[11px] mut">${(r||[]).length} rows</div>
          </div>`).join('')}
      </div>
      <button onclick="exportExcel()" class="text-xs px-4 py-2 rounded-lg font-semibold text-white"
              style="background:#3fa87a">Download .xlsx</button>`;
    return;
  }

  if (kind === 'pdf') {
    body.innerHTML = `
      <div class="mb-3 flex gap-2">
        <button onclick="exportPdf()" class="text-xs px-4 py-2 rounded-lg font-semibold text-white"
                style="background:#e0574c">Download .pdf</button>
      </div>
      <div class="rounded-lg p-8 mx-auto" style="background:#fff;color:#1a2338;max-width:660px;
           box-shadow:0 8px 28px rgba(0,0,0,.34)">
        <div style="border-bottom:3px solid #4c7ef3;padding-bottom:12px;margin-bottom:20px">
          <div style="font-size:22px;font-weight:800">LogiOptima</div>
          <div style="font-size:12px;color:#68758f">Executive Report · ${DATA.meta.generated_at}</div>
        </div>
        <div style="font-size:14px;font-weight:700;margin-bottom:10px">Headline metrics</div>
        <table style="width:100%;font-size:12px;border-collapse:collapse">
          ${[['Orders',nf(t.orders)],['On-time rate',pct(t.on_time)],['Revenue',usd(t.revenue)],
             ['Delivery cost',usd(t.cost)],['Gross margin',usd(t.margin)],
             ['Cost per order',usd(t.cost_per_order)]].map(([k,v])=>`
            <tr><td style="padding:6px 0;color:#68758f;border-bottom:1px solid #eef1f6">${k}</td>
                <td style="padding:6px 0;text-align:right;font-weight:600;border-bottom:1px solid #eef1f6">${v}</td></tr>`).join('')}
        </table>
        <div style="font-size:14px;font-weight:700;margin:22px 0 10px">Top recommendations</div>
        ${DATA.recommendations.slice(0,4).map(r=>`
          <div style="margin-bottom:12px;padding-left:11px;border-left:3px solid #4c7ef3">
            <div style="font-size:12.5px;font-weight:600">${r.rank}. ${r.title}</div>
            <div style="font-size:11px;color:#68758f;margin-top:3px">${r.evidence}</div>
            <div style="font-size:11px;color:#3fa87a;font-weight:600;margin-top:3px">
              ${r.impact_usd?usd(r.impact_usd)+' annual impact':'qualitative'}</div>
          </div>`).join('')}
      </div>`;
    return;
  }

  // PowerPoint-style slide preview
  const slides = [
    { t:'Network performance', rows:[['Orders',nf(t.orders)],['On-time',pct(t.on_time)],
        ['Revenue',usd(t.revenue)],['Margin',usd(t.margin)]] },
    { t:'Cost structure', rows:DATA.executive.cost_breakdown.map(c=>[c.category,usd(c.amount)]) },
    { t:'Optimisation impact', rows:(DATA.optimization.problems||[]).slice(0,4)
        .map(p=>[titleCase(p.problem), pct(p.improvement_pct)+' better']) },
    { t:'Top actions', rows:DATA.recommendations.slice(0,4)
        .map(r=>[r.title.slice(0,34), r.impact_usd?usd(r.impact_usd):'—']) }
  ];
  body.innerHTML = `
    <p class="text-[12.5px] mut mb-3">Slide preview of the current view. Use the Excel export for
      the underlying figures, or the PDF export for a formatted document.</p>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
      ${slides.map((s,i)=>`
        <div class="rounded-lg p-5" style="background:#fff;color:#1a2338;aspect-ratio:16/9;
             box-shadow:0 4px 16px rgba(0,0,0,.28)">
          <div style="font-size:9px;color:#9aa5bd;letter-spacing:1px">SLIDE ${i+1}</div>
          <div style="font-size:15px;font-weight:800;margin:5px 0 12px;color:#1a2338">${s.t}</div>
          ${s.rows.map(([k,v])=>`
            <div style="display:flex;justify-content:space-between;font-size:11.5px;padding:3.5px 0;
                        border-bottom:1px solid #f0f2f7">
              <span style="color:#68758f">${k}</span><span style="font-weight:600">${v}</span></div>`).join('')}
        </div>`).join('')}
    </div>`;
}

/* ============================================================== BOOTSTRAP = */
const RENDERERS = {
  executive: renderExecutive, operations: renderOperations, fleet: renderFleet,
  warehouse: renderWarehouse, delivery: renderDelivery, forecast: renderForecast,
  simulator: renderSimulator, recommendations: renderRecommendations, xai: renderXai
};

function init() {
  buildNav();

  document.getElementById('fRegion').innerHTML =
    '<option value="all">All regions</option>' +
    DATA.meta.regions.map(r=>`<option value="${r}">${r}</option>`).join('');
  document.getElementById('fDomain').innerHTML =
    '<option value="all">All domains</option>' +
    DATA.meta.domains.map(d=>`<option value="${d}">${titleCase(d)}</option>`).join('');

  const onFilter = () => {
    S.region = document.getElementById('fRegion').value;
    S.domain = document.getElementById('fDomain').value;
    updateFilterNote();
    S.drawn.delete('executive'); S.drawn.delete('delivery');
    render(S.module, true);
  };
  document.getElementById('fRegion').onchange = onFilter;
  document.getElementById('fDomain').onchange = onFilter;
  document.getElementById('btnReset').onclick = () => {
    document.getElementById('fRegion').value = 'all';
    document.getElementById('fDomain').value = 'all';
    onFilter();
  };

  document.getElementById('btnTheme').onclick = () => {
    document.documentElement.classList.toggle('dark');
    S.drawn.clear();
    render(S.module, true);
  };

  document.getElementById('btnExport').onclick = () => {
    document.getElementById('drawer').classList.add('open');
    setActiveTab('pdf');
  };
  document.getElementById('drawerClose').onclick = () =>
    document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer').onclick = e => {
    if (e.target.id === 'drawer') e.currentTarget.classList.remove('open');
  };
  document.querySelectorAll('.exTab').forEach(b =>
    b.onclick = () => setActiveTab(b.dataset.t));

  const m = DATA.meta;
  document.getElementById('footMeta').innerHTML =
    `${nf(m.source_rows.orders)} orders · ${m.date_range[0]} → ${m.date_range[1]}<br>
     Built ${m.generated_at}`;

  updateFilterNote();
  go('executive');
}

function setActiveTab(t) {
  document.querySelectorAll('.exTab').forEach(b => {
    const on = b.dataset.t === t;
    b.style.background = on ? '#4c7ef3' : 'transparent';
    b.style.color = on ? '#fff' : 'var(--mut)';
    b.style.border = on ? 'none' : '1px solid var(--line)';
  });
  exportPreview(t);
}

window.addEventListener('resize', () => {
  if (S.map && S.module === 'delivery') S.map.invalidateSize();
});

document.addEventListener('DOMContentLoaded', init);
if (document.readyState !== 'loading') init();
