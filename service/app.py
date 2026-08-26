"""
Lightweight RUL inference service — simulates edge deployment.

A single-file Flask app that loads the ~200KB XGBoost model into memory once and
serves predictions over HTTP. Small enough to run on a gateway/edge box next to
the machinery instead of round-tripping raw sensor streams to the cloud.

Endpoints:
  GET  /              industrial command-center dashboard (live RUL monitor)
  GET  /health        liveness + model metadata
  POST /predict       score one engine's recent cycle window -> RUL + alert

POST body:
{
  "unit_id": 7,
  "cycles": [                     # most recent cycles, oldest -> newest
    {"op_setting_1":..., ..., "sensor_1":..., ..., "sensor_21":...},
    ...
  ]
}
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, render_template_string

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data_prep as dp

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
WARN_RUL, CRITICAL_RUL = 50, 20  # maintenance thresholds (cycles)
RUL_CLIP_DEFAULT = 125

app = Flask(__name__)

with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    META = json.load(f)
MODEL = xgb.XGBRegressor()
MODEL.load_model(os.path.join(MODEL_DIR, "rul_xgb.json"))


def _alert(rul):
    if rul <= CRITICAL_RUL:
        return "CRITICAL", "Schedule maintenance immediately — imminent failure risk."
    if rul <= WARN_RUL:
        return "WARNING", "Plan maintenance in upcoming window to avoid unplanned downtime."
    return "OK", "Equipment healthy. Continue normal operation."


@app.get("/")
def dashboard():
    return render_template_string(DASHBOARD, rul_clip=META.get("rul_clip", RUL_CLIP_DEFAULT))


@app.get("/health")
def health():
    return jsonify(status="up", model="rul_xgb",
                   val_rmse=META.get("val_rmse"),
                   n_features=len(META["feature_cols"]))


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    cycles = payload.get("cycles", [])
    if not cycles:
        return jsonify(error="no cycles provided"), 400

    df = pd.DataFrame(cycles)
    df.insert(0, "cycle", np.arange(1, len(df) + 1))
    df.insert(0, "unit", payload.get("unit_id", 1))
    for c in dp.SENSOR_COLS + dp.OP_COLS:
        if c not in df.columns:
            df[c] = 0.0

    feats = dp.engineer(df, META["sensors"], tuple(META["windows"]))
    x_last = feats.iloc[[-1]][META["feature_cols"]]
    rul = float(np.clip(MODEL.predict(x_last)[0], 0, META["rul_clip"]))
    level, msg = _alert(rul)
    return jsonify(unit_id=payload.get("unit_id", 1),
                   predicted_rul_cycles=round(rul, 1),
                   health_index=round(100 * rul / META["rul_clip"], 1),
                   alert=level, recommendation=msg)


DASHBOARD = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>C-MAPSS RUL Monitor — Edge Inference</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  :root{
    --bg:#0a0d10; --panel:#0f1318; --line:#1e252d; --text:#E4E7EB; --muted:#7d8794;
    --teal:#c8860a; --teal-bright:#ffb300; --amber:#ff8c1a; --red:#E53935;
    --glass:rgba(18,23,28,.62);
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif;
    /* micro-dot diagnostic grid + soft teal glow */
    background-image:
      radial-gradient(circle at 20% 0%, rgba(200,134,10,.10), transparent 40%),
      radial-gradient(circle at 90% 100%, rgba(200,134,10,.06), transparent 45%),
      radial-gradient(rgba(255,255,255,.028) 1px, transparent 1px);
    background-size:auto, auto, 22px 22px;
  }
  /* film-grain noise overlay */
  body::after{
    content:""; position:fixed; inset:0; pointer-events:none; opacity:.035; z-index:9;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }
  .wrap{max-width:1080px; margin:0 auto; padding:34px 22px 60px; position:relative; z-index:1}
  .mono{font-family:'Share Tech Mono',monospace}
  .disp{font-family:Rajdhani,sans-serif; letter-spacing:.5px}

  /* header */
  header{display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; flex-wrap:wrap; gap:12px}
  .brand{display:flex; align-items:center; gap:12px}
  .logo{width:34px;height:34px;border:1.5px solid var(--teal);border-radius:7px;position:relative;box-shadow:0 0 14px rgba(255,179,0,.4) inset}
  .logo::before{content:"";position:absolute;inset:8px;border:1.5px solid var(--teal-bright);border-radius:3px}
  h1{font-family:Rajdhani; font-weight:700; font-size:22px; margin:0; letter-spacing:1px; text-transform:uppercase}
  .status-pill{font-family:'Share Tech Mono',monospace; font-size:12px; color:var(--teal-bright);
    border:1px solid rgba(255,179,0,.35); border-radius:999px; padding:5px 12px; display:flex; align-items:center; gap:8px}
  .dot-live{width:7px;height:7px;border-radius:50%;background:var(--teal-bright);box-shadow:0 0 8px var(--teal-bright);animation:blink 1.4s infinite}
  @keyframes blink{50%{opacity:.25}}
  .tagline{color:var(--muted); font-size:14px; margin:0 0 26px}

  .grid{display:grid; grid-template-columns:230px 1fr; gap:18px}
  @media(max-width:820px){.grid{grid-template-columns:1fr}}

  .glass{
    background:var(--glass); border:1px solid var(--line); border-radius:14px;
    backdrop-filter:blur(9px) saturate(120%); -webkit-backdrop-filter:blur(9px) saturate(120%);
    position:relative; overflow:hidden;
  }
  .glass::before{ /* top hairline sheen */
    content:"";position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,179,0,.5),transparent)}
  .pad{padding:18px}
  .label{font-family:'Share Tech Mono',monospace; font-size:11px; letter-spacing:1.5px;
    text-transform:uppercase; color:var(--muted); margin:0 0 12px}

  /* asset rail */
  .asset{display:flex;align-items:center;gap:10px;padding:11px 12px;border-radius:9px;cursor:pointer;
    border:1px solid transparent;transition:.15s;font-family:'Share Tech Mono',monospace;font-size:13px}
  .asset:hover{background:rgba(255,255,255,.03)}
  .asset.sel{border-color:rgba(255,179,0,.4);background:rgba(200,134,10,.10)}
  .asset .id{color:var(--text)}
  .asset .mini{width:8px;height:8px;border-radius:50%;margin-left:auto;background:var(--teal-bright);box-shadow:0 0 7px var(--teal-bright)}

  /* center console */
  .console{min-height:340px;display:flex;flex-direction:column}
  .console.OK{--accent:var(--teal-bright)}
  .console.WARNING{--accent:#ff7a00}
  .console.CRITICAL{--accent:var(--red)}
  .console-top{display:flex;gap:22px;padding:22px;align-items:center;flex-wrap:wrap}

  /* battery health cell */
  .battery-wrap{display:flex;flex-direction:column;align-items:center;gap:8px;min-width:150px}
  .battery{width:120px;height:210px;border:3px solid var(--line);border-radius:14px;position:relative;
    padding:7px;background:rgba(0,0,0,.25)}
  .battery::before{content:"";position:absolute;top:-13px;left:50%;transform:translateX(-50%);
    width:42px;height:11px;background:var(--line);border-radius:4px 4px 0 0}
  .cells{position:absolute;inset:7px;display:flex;flex-direction:column-reverse;gap:5px}
  .fill{width:100%;border-radius:4px;background:var(--accent,#00d4d4);
    box-shadow:0 0 12px var(--accent,#00d4d4);transition:height .9s cubic-bezier(.2,.8,.2,1),background .4s;position:relative}
  .fill::after{content:"";position:absolute;inset:0;border-radius:4px;
    background:repeating-linear-gradient(0deg,transparent,transparent 9px,rgba(0,0,0,.22) 9px,rgba(0,0,0,.22) 12px)}
  .hpct{font-family:Rajdhani;font-weight:700;font-size:26px;color:var(--accent,#00d4d4)}
  .hlab{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase}

  /* RUL readout */
  .readout{flex:1;min-width:200px}
  .rul-num{font-family:Rajdhani;font-weight:700;font-size:82px;line-height:.9;color:var(--accent,#00d4d4);
    text-shadow:0 0 22px color-mix(in srgb,var(--accent,#00d4d4) 55%,transparent)}
  .rul-unit{font-family:'Share Tech Mono',monospace;font-size:12px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-top:2px}
  .badge{display:inline-flex;align-items:center;gap:8px;font-family:Rajdhani;font-weight:700;font-size:15px;
    letter-spacing:1.5px;padding:7px 16px;border-radius:8px;margin-top:16px;text-transform:uppercase;
    border:1px solid var(--accent,#00d4d4);color:var(--accent,#00d4d4);background:color-mix(in srgb,var(--accent,#00d4d4) 12%,transparent)}
  .badge .bd{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent)}
  .glitch{animation:glitch .45s steps(2) infinite}
  @keyframes glitch{
    0%{transform:translate(0,0);text-shadow:0 0 22px var(--red)}
    25%{transform:translate(-1.5px,1px);text-shadow:2px 0 var(--teal-bright),-2px 0 var(--red)}
    50%{transform:translate(1.5px,-1px)}
    75%{transform:translate(-1px,0);text-shadow:-2px 0 var(--teal-bright),2px 0 var(--red)}
  }
  .rec{color:var(--muted);font-size:13.5px;padding:0 22px 8px;line-height:1.55;max-width:560px}

  /* sensor sparkline matrix */
  .matrix{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:0 18px 18px}
  @media(max-width:640px){.matrix{grid-template-columns:repeat(2,1fr)}}
  .spark{background:rgba(0,0,0,.22);border:1px solid var(--line);border-radius:10px;padding:10px}
  .spark .sn{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:1px;text-transform:uppercase}
  .spark .sv{font-family:Rajdhani;font-weight:600;font-size:17px;color:var(--text)}
  .spark svg{width:100%;height:34px;display:block;margin-top:3px}

  /* controls */
  .ctl{padding:18px}
  .ctl label{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:1px;color:var(--muted);
    text-transform:uppercase;display:flex;justify-content:space-between;margin-bottom:7px}
  .ctl label b{color:var(--teal-bright)}
  input[type=range]{-webkit-appearance:none;width:100%;height:5px;border-radius:5px;
    background:linear-gradient(90deg,var(--teal),var(--line));outline:none;margin-bottom:18px}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:17px;height:17px;border-radius:50%;
    background:var(--teal-bright);box-shadow:0 0 10px var(--teal-bright);cursor:pointer;border:2px solid #0b1417}
  button{font-family:Rajdhani;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;font-size:15px;
    background:linear-gradient(180deg,#ffb300,#c8860a);color:#1a1206;border:0;border-radius:9px;
    padding:13px;width:100%;cursor:pointer;color:#1a1206;box-shadow:0 0 0 1px rgba(255,179,0,.35),0 6px 20px rgba(200,134,10,.25)}
  button:hover{filter:brightness(1.12)}
  button:active{transform:translateY(1px)}

  /* scanline over the console */
  .scan{position:absolute;left:0;right:0;height:64px;pointer-events:none;z-index:2;
    background:linear-gradient(180deg,transparent,rgba(255,179,0,.06) 45%,rgba(255,179,0,.10) 50%,transparent);
    animation:scandown 5.5s linear infinite}
  @keyframes scandown{0%{top:-64px}100%{top:100%}}
  @media(prefers-reduced-motion:reduce){.scan,.glitch,.dot-live{animation:none}}
  footer{margin-top:26px;color:var(--muted);font-size:11.5px;font-family:'Share Tech Mono',monospace;text-align:center;letter-spacing:.5px}
  footer a{color:var(--teal-bright);text-decoration:none}

  /* 3D digital-twin battery core, fixed behind everything */
  #core{position:fixed;inset:0;width:100vw;height:100vh;z-index:0;pointer-events:none;opacity:.9}
  /* ambient fallback glow (shows if WebGL/Three.js is unavailable) */
  #coreFallback{position:fixed;inset:0;z-index:0;pointer-events:none;
    background:radial-gradient(ellipse 44% 58% at 50% 48%, rgba(255,179,0,.12), transparent 70%);
    animation:breathe 7s ease-in-out infinite}
  @keyframes breathe{50%{opacity:.5;transform:scale(1.05)}}

  /* second-half pipeline section */
  .pipeline{margin-top:40px}
  .sec-label{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:2px;color:var(--muted);
    text-transform:uppercase;text-align:center;margin-bottom:6px}
  .sec-title{font-family:Rajdhani;font-weight:700;font-size:26px;text-transform:uppercase;letter-spacing:1px;
    text-align:center;margin:0 0 26px}
  .flow{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  @media(max-width:820px){.flow{grid-template-columns:1fr}}
  .stage{padding:20px}
  .stage .n{font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--teal-bright);letter-spacing:1px}
  .stage h3{font-family:Rajdhani;font-weight:600;font-size:18px;margin:8px 0 8px;letter-spacing:.5px}
  .stage p{color:var(--muted);font-size:13px;line-height:1.6;margin:0}
  .metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px}
  @media(max-width:820px){.metrics{grid-template-columns:1fr}}
  .metric{padding:18px;text-align:center}
  .metric .big{font-family:Rajdhani;font-weight:700;font-size:34px;color:var(--teal-bright)}
  .metric .cap{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:1px;color:var(--muted);text-transform:uppercase;margin-top:4px}
</style>
</head>
<body>
<div id="coreFallback"></div>
<canvas id="core"></canvas>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo"></div>
      <div>
        <h1>C-MAPSS RUL Monitor</h1>
        <div class="mono" style="font-size:11px;color:var(--muted);letter-spacing:1px">EDGE INFERENCE // TURBOFAN PROGNOSTICS</div>
      </div>
    </div>
    <div class="status-pill"><span class="dot-live"></span>SERVICE ONLINE · :8000</div>
  </header>
  <p class="tagline">Predicting asset degradation before unplanned downtime occurs — remaining useful life estimated live from sensor telemetry.</p>

  <div class="grid">
    <!-- asset rail -->
    <div class="glass pad">
      <p class="label">Asset Nodes</p>
      <div id="rail"></div>
      <div class="ctl" style="padding:14px 0 0">
        <label>Wear level <b id="wearv">60%</b></label>
        <input type="range" id="wear" min="0" max="100" value="60">
        <button onclick="run()">Run Inference</button>
      </div>
    </div>

    <!-- center console -->
    <div class="glass console OK" id="console">
      <div class="scan"></div>
      <div class="console-top">
        <div class="battery-wrap">
          <div class="battery"><div class="cells"><div class="fill" id="fill" style="height:60%"></div></div></div>
          <div class="hpct" id="hpct">—</div>
          <div class="hlab">Asset Health Index</div>
        </div>
        <div class="readout">
          <p class="label" style="margin-bottom:4px">Remaining Useful Life</p>
          <div class="rul-num" id="rul">—</div>
          <div class="rul-unit">operating cycles remaining</div>
          <div class="badge" id="badge"><span class="bd"></span><span id="badget">STANDBY</span></div>
        </div>
      </div>
      <div class="rec" id="rec">Select an asset node and run inference to estimate remaining useful life from its recent sensor window.</div>
      <div class="matrix" id="matrix"></div>
    </div>
  </div>

  <section class="pipeline">
    <div class="sec-label">// System Architecture</div>
    <h2 class="sec-title">The Edge Inference Pipeline</h2>
    <div class="flow">
      <div class="glass stage">
        <div class="n">01 — INGEST</div>
        <h3>Sensor Telemetry</h3>
        <p>Twenty-one raw sensor channels plus operating settings stream in per operating cycle. Rolling windows expose degradation trend rather than instantaneous noise.</p>
      </div>
      <div class="glass stage">
        <div class="n">02 — MODEL</div>
        <h3>Degradation Estimator</h3>
        <p>A gradient-boosted regressor maps the engineered window to remaining useful life, using a piecewise-linear target clipped where wear begins.</p>
      </div>
      <div class="glass stage">
        <div class="n">03 — SERVE</div>
        <h3>Dockerized Edge Node</h3>
        <p>The ~200&nbsp;KB model runs inference locally on a gateway box beside the asset, returning RUL and a maintenance alert without a cloud round-trip.</p>
      </div>
    </div>

    <div class="sec-label" style="margin-top:36px">// Why Asymmetric Scoring</div>
    <h2 class="sec-title">Cost of Being Wrong</h2>
    <div class="metrics">
      <div class="glass metric"><div class="big">13×</div><div class="cap">Late-prediction penalty vs early</div></div>
      <div class="glass metric"><div class="big" id="m-rmse">—</div><div class="cap">Validation RMSE (cycles)</div></div>
      <div class="glass metric"><div class="big">0</div><div class="cap">Unplanned outages when flagged early</div></div>
    </div>
  </section>

  <footer>MODEL rul_xgb · asymmetric NASA scoring · <a href="/health">/health</a> · <a href="/predict">/predict</a> — synthetic telemetry demo; swap in real C-MAPSS FD001 for production readings</footer>
</div>

<script>
const RUL_CLIP = {{ rul_clip }};
const SENSORS = [["Core Temp","K"],["Fan Speed","rpm"],["Pressure","psi"],["Fuel Flow","pps"]];
let selected = 1;
const $ = id => document.getElementById(id);

// build asset rail
const rail = $('rail');
for(let i=1;i<=6;i++){
  const el = document.createElement('div');
  el.className = 'asset' + (i===1?' sel':'');
  el.innerHTML = `<span class="id">TURBOFAN_${String(i).padStart(3,'0')}</span><span class="mini"></span>`;
  el.onclick = ()=>{ selected=i; document.querySelectorAll('.asset').forEach(a=>a.classList.remove('sel')); el.classList.add('sel'); run(); };
  rail.appendChild(el);
}
$('wear').oninput = e => $('wearv').textContent = e.target.value + '%';

function sparkline(seed, drift){
  let pts=[], v=50;
  for(let i=0;i<24;i++){ v += (Math.sin(i*0.7+seed)*4) + (Math.random()-.5)*5 + drift; pts.push(v); }
  const min=Math.min(...pts), max=Math.max(...pts), rng=(max-min)||1;
  const d = pts.map((p,i)=>`${(i/23*100).toFixed(1)},${(30-(p-min)/rng*26-2).toFixed(1)}`).join(' ');
  return `<svg viewBox="0 0 100 34" preserveAspectRatio="none"><polyline points="${d}" fill="none" stroke="var(--accent,#00d4d4)" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>`;
}
function renderMatrix(wear){
  $('matrix').innerHTML = SENSORS.map((s,i)=>{
    const val = (400 + i*180 + wear*2 + Math.random()*8).toFixed(1);
    return `<div class="spark"><div class="sn">${s[0]}</div><div class="sv">${val} <span style="font-size:11px;color:var(--muted)">${s[1]}</span></div>${sparkline(i*2+selected, wear/40)}</div>`;
  }).join('');
}
renderMatrix(60);

function buildCycles(wear){
  const w = wear/100, cy=[];
  for(let i=0;i<20;i++){
    const c={};
    for(let s=1;s<=21;s++){ c["sensor_"+s] = 480 + s*6 + w*40*(s%3) + i*w*1.5 + (Math.random()-.5); }
    c.op_setting_1=0; c.op_setting_2=0; c.op_setting_3=100; cy.push(c);
  }
  return cy;
}

async function run(){
  const wear = +$('wear').value;
  renderMatrix(wear);
  const body = { unit_id: selected, cycles: buildCycles(wear) };
  try{
    const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await r.json();
    const con = $('console'); con.className = 'glass console ' + d.alert;
    const hi = d.health_index != null ? d.health_index : Math.round(100*d.predicted_rul_cycles/RUL_CLIP);
    $('rul').textContent = d.predicted_rul_cycles;
    $('hpct').textContent = hi + '%';
    $('fill').style.height = Math.max(3,Math.min(100,hi)) + '%';
    $('badget').textContent = d.alert;
    $('rec').textContent = d.recommendation;
    const badge = $('badge');
    badge.classList.toggle('glitch', d.alert === 'CRITICAL');
    if (window.setCoreState) window.setCoreState(d.alert, hi);
  }catch(e){ $('rec').textContent = 'Cannot reach inference service — is the server running on :8000? ('+e+')'; }
}

// fill the validation-RMSE metric from /health
fetch('/health').then(r=>r.json()).then(d=>{
  if(d.val_rmse!=null) $('m-rmse').textContent = d.val_rmse.toFixed(1);
}).catch(()=>{});

run();

/* ---------- 3D cylindrical battery cells (digital twin) ---------- */
(function(){
  if(!window.THREE){ return; }   // CDN blocked/offline -> CSS fallback glow stays
  const fb = document.getElementById('coreFallback'); if(fb) fb.style.display='none';
  const canvas = document.getElementById('core');
  const renderer = new THREE.WebGLRenderer({canvas, alpha:true, antialias:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(38, innerWidth/innerHeight, 0.1, 100);
  cam.position.set(4.3, 2.4, 7.4);
  cam.lookAt(0,0,0);

  // lighting so the cylinders read as solid, glossy cells
  scene.add(new THREE.AmbientLight(0x404a55, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.15); key.position.set(6,9,7); scene.add(key);
  const rim = new THREE.DirectionalLight(0xffb300, 0.45); rim.position.set(-7,2,-4); scene.add(rim);

  const group = new THREE.Group();
  group.rotation.z = 0.5;   // lay the cells on an industrial tilt
  scene.add(group);

  const COL = { OK:0xffb300, WARNING:0xff7a00, CRITICAL:0xe53935 };
  const DARK = 0x14181d, METAL = 0x9aa4ad;
  let accent = COL.OK, targetFill = 0.9, curFill = 0.9, spin = 0.0028, jitter = 0;

  const bodyMats = [], glowMats = [], capMats = [];

  // Build one cylindrical cell: dark body + colored band + metal cap + raised terminal.
  function makeCell(){
    const cell = new THREE.Group();
    const R = 1.15, H = 4.6;

    // main dark body (matte)
    const bodyMat = new THREE.MeshStandardMaterial({color:DARK, metalness:.35, roughness:.55});
    const body = new THREE.Mesh(new THREE.CylinderGeometry(R,R,H,48), bodyMat);
    cell.add(body); bodyMats.push(bodyMat);

    // colored accent band (the state-reactive "charge" zone) around lower body
    const glowMat = new THREE.MeshStandardMaterial({color:accent, emissive:accent,
      emissiveIntensity:.55, metalness:.3, roughness:.4});
    const band = new THREE.Mesh(new THREE.CylinderGeometry(R*1.005,R*1.005,H*0.42,48), glowMat);
    band.position.y = -H*0.24; cell.add(band); glowMats.push(glowMat);

    // positive terminal cap (metal) at top
    const capMat = new THREE.MeshStandardMaterial({color:METAL, metalness:.9, roughness:.25});
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(R*0.99,R*0.99,0.12,48), capMat);
    cap.position.y = H/2+0.02; cell.add(cap); capMats.push(capMat);
    // raised nub
    const nub = new THREE.Mesh(new THREE.CylinderGeometry(R*0.34,R*0.34,0.22,32), capMat);
    nub.position.y = H/2+0.16; cell.add(nub);
    // "+" bars on the cap
    const plusMat = new THREE.MeshStandardMaterial({color:0xe4e7eb, metalness:.4, roughness:.4});
    const b1 = new THREE.Mesh(new THREE.BoxGeometry(0.5,0.03,0.12), plusMat);
    const b2 = new THREE.Mesh(new THREE.BoxGeometry(0.12,0.03,0.5), plusMat);
    b1.position.y = b2.position.y = H/2+0.28; cell.add(b1); cell.add(b2);

    // subtle wireframe overlay -> "digital twin" read
    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.CylinderGeometry(R*1.01,R*1.01,H,24)),
      new THREE.LineBasicMaterial({color:accent, transparent:true, opacity:0.18}));
    cell.add(wire);

    cell._fillTarget = band;   // reference for fill animation
    cell._H = H;
    return cell;
  }

  const c1 = makeCell(); c1.position.set(-1.55,0,0.4); c1.rotation.z = 0.05;
  const c2 = makeCell(); c2.position.set(1.75,-0.25,-0.7); c2.rotation.z = -0.08; c2.scale.setScalar(0.96);
  group.add(c1); group.add(c2);

  // floating telemetry particles drifting up around the cells
  const pGeo = new THREE.BufferGeometry();
  const N = 130, pos = new Float32Array(N*3), spd = new Float32Array(N);
  for(let i=0;i<N;i++){
    pos[i*3]=(Math.random()-0.5)*11; pos[i*3+1]=(Math.random()-0.5)*11; pos[i*3+2]=(Math.random()-0.5)*11;
    spd[i]=0.004+Math.random()*0.012;
  }
  pGeo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
    color:accent, size:0.055, transparent:true, opacity:0.55}));
  scene.add(particles);

  function applyColor(hex){
    glowMats.forEach(m=>{ m.color.setHex(hex); m.emissive.setHex(hex); });
    particles.material.color.setHex(hex);
  }

  window.setCoreState = function(alert, health){
    accent = COL[alert] || COL.OK;
    applyColor(accent);
    targetFill = Math.max(0.05, Math.min(1, health/100));
    spin = alert==='WARNING' ? 0.0018 : alert==='CRITICAL' ? 0.0010 : 0.0028;
    jitter = alert==='CRITICAL' ? 1 : 0;
  };

  function animate(){
    requestAnimationFrame(animate);
    group.rotation.y += spin;
    // ease charge band height/position to represent health (fills bottom-up)
    curFill += (targetFill - curFill)*0.06;
    [c1,c2].forEach(cell=>{
      const H = cell._H, band = cell._fillTarget;
      const h = Math.max(0.04, curFill) * H;
      band.scale.y = h / (H*0.42);
      band.position.y = -H/2 + h/2;    // grow up from the base
    });
    glowMats.forEach(m=> m.emissiveIntensity = 0.4 + 0.25*curFill + (jitter? Math.random()*0.4:0));
    // critical micro-glitch
    if(jitter){ group.position.x=(Math.random()-0.5)*0.09; group.position.y=(Math.random()-0.5)*0.09; }
    else { group.position.x*=0.8; group.position.y*=0.8; }
    const p = pGeo.attributes.position.array;
    for(let i=0;i<N;i++){ p[i*3+1]+=spd[i]; if(p[i*3+1]>5.5) p[i*3+1]=-5.5; }
    pGeo.attributes.position.needsUpdate = true;
    renderer.render(scene, cam);
  }
  function resize(){ renderer.setSize(innerWidth, innerHeight); cam.aspect=innerWidth/innerHeight; cam.updateProjectionMatrix(); }
  addEventListener('resize', resize); resize(); animate();
})();
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)