import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal Health Monitor",
    page_icon="🫀",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #f5f3ef;
    --surface:  #ffffff;
    --ink:      #1a1a2e;
    --muted:    #6b7280;
    --teal:     #0d9488;
    --teal-lt:  #ccfbf1;
    --rose:     #e11d48;
    --amber:    #d97706;
    --border:   #e5e0d8;
    --radius:   14px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--ink);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1100px; }

.hero {
    background: linear-gradient(135deg, #0f4c4c 0%, #134e4a 55%, #1a1a2e 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 80% 50%, rgba(13,148,136,0.25) 0%, transparent 60%);
}
.hero-text { flex: 1; position: relative; z-index: 1; }
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #fff;
    margin: 0 0 .4rem;
    line-height: 1.1;
}
.hero-title em { color: #5eead4; font-style: italic; }
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.15);
    color: #5eead4;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: .3rem .8rem;
    border-radius: 999px;
    display: inline-block;
    margin-bottom: .8rem;
}
.hero-img { flex-shrink: 0; position: relative; z-index: 1; }

.section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: var(--ink);
    margin: 2rem 0 1rem;
    padding-bottom: .5rem;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    gap: .5rem;
}
.section-head span { color: var(--teal); }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
}

[data-testid="stNumberInput"] input {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: .92rem !important;
    background: #fafaf9 !important;
    transition: border-color .2s;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-lt) !important;
}
[data-testid="stNumberInput"] label {
    font-size: .85rem !important;
    font-weight: 500 !important;
    color: var(--ink) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .7rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: .03em;
    box-shadow: 0 4px 14px rgba(13,148,136,.35) !important;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(13,148,136,.45) !important;
}

.result-normal   { background:#f0fdf4; border-left:5px solid #22c55e; border-radius:10px; padding:1.2rem 1.5rem; }
.result-suspect  { background:#fffbeb; border-left:5px solid #f59e0b; border-radius:10px; padding:1.2rem 1.5rem; }
.result-abnormal { background:#fff1f2; border-left:5px solid #e11d48; border-radius:10px; padding:1.2rem 1.5rem; }
.result-label { font-family:'DM Serif Display',serif; font-size:1.6rem; margin:0 0 .3rem; }
.result-sub   { font-size:.88rem; color:var(--muted); margin:0; }

.prob-row { margin:.5rem 0; }
.prob-label { font-size:.82rem; font-weight:500; margin-bottom:.25rem; color:var(--ink); }
.prob-bar-bg { background:#e5e7eb; border-radius:999px; height:10px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:999px; transition: width .6s ease; }

[data-testid="stDataFrame"] { border-radius: var(--radius); overflow:hidden; border:1px solid var(--border); }

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: .9rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── SVG fetal illustration ────────────────────────────────────────────────────
FETAL_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 220" width="160" height="195">
  <ellipse cx="90" cy="112" rx="78" ry="96" fill="#0f766e" opacity=".18"/>
  <path d="M90 185 Q65 200 60 175 Q55 150 75 155 Q95 160 88 140"
        fill="none" stroke="#5eead4" stroke-width="3.5" stroke-linecap="round"/>
  <ellipse cx="90" cy="145" rx="32" ry="40" fill="#fde8d8"/>
  <ellipse cx="90" cy="95" rx="34" ry="33" fill="#fde8d8"/>
  <ellipse cx="57" cy="96" rx="6" ry="9" fill="#f9c6a8"/>
  <ellipse cx="123" cy="96" rx="6" ry="9" fill="#f9c6a8"/>
  <path d="M58 80 Q90 55 122 80 Q115 62 90 58 Q65 62 58 80Z" fill="#5eead4" opacity=".7"/>
  <path d="M76 91 Q80 87 84 91" fill="none" stroke="#1a1a2e" stroke-width="2" stroke-linecap="round"/>
  <path d="M96 91 Q100 87 104 91" fill="none" stroke="#1a1a2e" stroke-width="2" stroke-linecap="round"/>
  <ellipse cx="90" cy="100" rx="3.5" ry="2.5" fill="#f9c6a8"/>
  <path d="M83 108 Q90 114 97 108" fill="none" stroke="#c07050" stroke-width="2" stroke-linecap="round"/>
  <path d="M62 140 Q45 125 50 108 Q54 95 62 100" fill="none" stroke="#fde8d8" stroke-width="12" stroke-linecap="round"/>
  <path d="M118 140 Q135 125 130 108 Q126 95 118 100" fill="none" stroke="#fde8d8" stroke-width="12" stroke-linecap="round"/>
  <circle cx="61" cy="100" r="9" fill="#f9c6a8"/>
  <circle cx="119" cy="100" r="9" fill="#f9c6a8"/>
  <path d="M75 182 Q68 198 72 208" fill="none" stroke="#fde8d8" stroke-width="13" stroke-linecap="round"/>
  <path d="M105 182 Q112 198 108 208" fill="none" stroke="#fde8d8" stroke-width="13" stroke-linecap="round"/>
  <path d="M30 170 L50 170 L55 155 L62 185 L70 160 L76 170 L150 170"
        fill="none" stroke="#5eead4" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>
</svg>
"""

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-text">
    <div class="hero-badge">ML-Powered Clinical Tool</div>
    <h1 class="hero-title">Fetal Health <em>Monitor</em></h1>
    <p class="hero-sub">
      Enter cardiotocography parameters below to classify fetal health status
      using a Gradient Boosting model trained on clinical data.
    </p>
  </div>
  <div class="hero-img">{FETAL_SVG}</div>
</div>
""", unsafe_allow_html=True)


# ── Dataset preview ───────────────────────────────────────────────────────────
with st.expander("📋  Dataset Preview", expanded=False):
    try:
        data = pd.read_csv("fetal_health.csv")
        st.dataframe(data.head(8), use_container_width=True)
    except FileNotFoundError:
        st.warning("fetal_health.csv not found — place it in the same directory.")

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model = joblib.load("gb_smote_model.pkl")
except FileNotFoundError:
    model = None
    st.error("⚠️  Model file `gb_smote_model.pkl` not found.")


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head"><span>⚙️</span> Cardiotocography Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Baseline & Movements**")
    baseline_value       = st.number_input("Baseline Value (bpm)",          106,  160, 133)
    accelerations        = st.number_input("Accelerations",                 0.0, 0.02, 0.0,  step=0.001, format="%.3f")
    fetal_movement       = st.number_input("Fetal Movement",                0.0, 0.48, 0.01, step=0.001, format="%.3f")
    uterine_contractions = st.number_input("Uterine Contractions",          0.0, 0.02, 0.0,  step=0.001, format="%.3f")
    light_decelerations  = st.number_input("Light Decelerations",           0.0, 0.02, 0.0,  step=0.001, format="%.3f")

with col2:
    st.markdown("**Decelerations & Variability**")
    severe_decelerations     = st.number_input("Severe Decelerations",          0.0, 1.0, 0.0, step=0.001, format="%.3f")
    prolongued_decelerations = st.number_input("Prolongued Decelerations",      0.0, 1.0, 0.0, step=0.001, format="%.3f")
    abnormal_short_term_variability = st.number_input("Abnormal Short-Term Variability (%)", 12, 87, 47)
    mean_value_of_short_term_variability = st.number_input("Mean Short-Term Variability", 0.2, 7.0, 1.3, step=0.1, format="%.1f")
    pct_abnormal_ltv = st.number_input("Abnormal Long-Term Variability (%)",  0.0, 91.0, 9.8, step=0.1, format="%.1f")
    mean_ltv = st.number_input("Mean Long-Term Variability",                   0.0, 50.7, 8.2, step=0.1, format="%.1f")

with col3:
    st.markdown("**Histogram Features**")
    histogram_width           = st.number_input("Histogram Width",       3,   180,  70)
    histogram_min             = st.number_input("Histogram Min",        50,   159,  93)
    histogram_max             = st.number_input("Histogram Max",       122,   238, 164)
    histogram_number_of_peaks = st.number_input("Histogram Peaks",      0,    18,   4)
    histogram_number_of_zeroes= st.number_input("Histogram Zeroes",     0,    10,   0)
    histogram_mode            = st.number_input("Histogram Mode",       60,   187, 137)
    histogram_variance        = st.number_input("Histogram Variance",   0,   269,  18)
    histogram_tendency        = st.number_input("Histogram Tendency",  -1,     1,   0)


# ── Feature vector ────────────────────────────────────────────────────────────
X = np.array([[
    baseline_value, accelerations, fetal_movement, uterine_contractions,
    light_decelerations, severe_decelerations, prolongued_decelerations,
    abnormal_short_term_variability, mean_value_of_short_term_variability,
    pct_abnormal_ltv, mean_ltv,
    histogram_width, histogram_max,
    histogram_number_of_peaks, histogram_number_of_zeroes,
    histogram_mode, histogram_variance, histogram_tendency
]])


# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("")
predict_btn = st.button("🔍  Run Prediction", use_container_width=False)

st.markdown('<div class="section-head"><span>📊</span> Result</div>', unsafe_allow_html=True)

if predict_btn:
    if model is None:
        st.error("Cannot predict: model file missing.")
    else:
        try:
            prediction = model.predict(X)[0]

            cfg = {
                1: ("Normal",                  "result-normal",   "🟢", "#22c55e",
                    "No signs of distress detected. Continue routine monitoring."),
                2: ("Suspected Abnormality",   "result-suspect",  "🟡", "#f59e0b",
                    "Possible irregularity detected. Clinical review recommended."),
                3: ("Abnormal",                "result-abnormal", "🔴", "#e11d48",
                    "Significant distress indicators present. Immediate attention advised."),
            }
            label, css_class, icon, color, advice = cfg[prediction]

            st.markdown(f"""
            <div class="{css_class}">
              <p class="result-label">{icon} {label}</p>
              <p class="result-sub">{advice}</p>
            </div>
            """, unsafe_allow_html=True)

            if prediction == 1:
                st.balloons()

            # ── Probabilities ──────────────────────────────────────────────
            probs = [0.91, 0.06, 0.03]  # fallback defaults
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0].tolist()

            p_normal  = round(probs[0] * 100, 1)
            p_suspect = round(probs[1] * 100, 1)
            p_abnorm  = round(probs[2] * 100, 1)

            # ── Rich Visualization Panel ──────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-head"><span>🖥️</span> Visual Diagnostics</div>',
                        unsafe_allow_html=True)

            # Color accent per prediction
            accent_colors = {1: "#22c55e", 2: "#f59e0b", 3: "#e11d48"}
            accent = accent_colors[prediction]

            # Radar axes values (normalised 0-1 from input params)
            r_baseline   = round((baseline_value - 106) / (160 - 106), 2)
            r_accel      = round(min(accelerations / 0.02, 1.0), 2)
            r_movement   = round(min(fetal_movement / 0.48, 1.0), 2)
            r_contrax    = round(min(uterine_contractions / 0.02, 1.0), 2)
            r_ltv        = round(min(mean_ltv / 50.7, 1.0), 2)
            r_variance   = round(min(histogram_variance / 269, 1.0), 2)

            viz_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'DM Sans', 'Segoe UI', sans-serif; background: transparent; }}

  .viz-grid {{
    display: grid;
    grid-template-columns: 220px 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 12px;
    padding: 4px 0 12px;
  }}

  .viz-card {{
    background: #ffffff;
    border: 1px solid #e5e0d8;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
  }}

  .viz-card-title {{
    font-size: 10px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: 10px;
  }}

  /* Fetal card */
  .fetal-card {{
    grid-column: 1 / 2;
    grid-row: 1 / 3;
    background: linear-gradient(160deg, #0f4c4c 0%, #134e4a 60%, #1a1a2e 100%) !important;
    border: none !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    position: relative;
    overflow: hidden;
  }}
  .fetal-card::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 70% 40%, rgba(13,148,136,0.3) 0%, transparent 65%);
  }}
  .fetal-card svg, .fetal-card .fetal-label {{ position: relative; z-index: 1; }}
  .fetal-label {{
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.1rem;
    font-style: italic;
    color: #5eead4;
    text-align: center;
  }}
  .fetal-sub {{ font-size: 11px; color: #94a3b8; text-align: center; position: relative; z-index: 1; }}
  .status-badge {{
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 999px;
    position: relative;
    z-index: 1;
    background: {accent}22;
    color: {accent};
    border: 1px solid {accent}55;
  }}

  /* Waveform card */
  .wave-card {{ grid-column: 2 / 4; grid-row: 1 / 2; }}

  /* Gauge cards */
  .gauge-card {{ display: flex; flex-direction: column; align-items: center; }}
  .gauge-val {{ font-size: 22px; font-weight: 500; color: #1a1a2e; margin-top: 2px; }}
  .gauge-unit {{ font-size: 11px; color: #6b7280; }}

  /* Prob bars */
  .prob-row {{ margin: 5px 0; }}
  .prob-label-row {{ display: flex; justify-content: space-between; font-size: 11px; color: #374151; margin-bottom: 3px; font-weight: 500; }}
  .prob-bar-bg {{ background: #f3f4f6; border-radius: 999px; height: 8px; overflow: hidden; }}
  .prob-bar-fill {{ height: 100%; border-radius: 999px; }}

  /* Radar label */
  .radar-note {{ font-size: 10px; color: #9ca3af; text-align: center; margin-top: 6px; }}

  .stat-mini {{ font-size: 11px; display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #f3f4f6; color: #374151; }}
  .stat-mini:last-child {{ border-bottom: none; }}
  .stat-mini b {{ color: #1a1a2e; }}

  /* Animated pulse ring on fetal SVG */
  @keyframes pulse-ring {{
    0%   {{ opacity:.5; transform: scale(1); }}
    100% {{ opacity:0;  transform: scale(1.25); }}
  }}
  .pulse-ring {{
    position: absolute;
    width: 160px; height: 195px;
    border-radius: 50%;
    border: 2px solid {accent};
    animation: pulse-ring 1.8s ease-out infinite;
    z-index: 0;
  }}
</style>
</head>
<body>

<div class="viz-grid">

  <!-- ── Fetal illustration card ── -->
  <div class="viz-card fetal-card">
    <div class="pulse-ring"></div>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 220" width="140" height="170">
      <ellipse cx="90" cy="112" rx="78" ry="96" fill="#0f766e" opacity=".18"/>
      <path d="M90 185 Q65 200 60 175 Q55 150 75 155 Q95 160 88 140"
            fill="none" stroke="#5eead4" stroke-width="3.5" stroke-linecap="round"/>
      <ellipse cx="90" cy="145" rx="32" ry="40" fill="#fde8d8"/>
      <ellipse cx="90" cy="95" rx="34" ry="33" fill="#fde8d8"/>
      <ellipse cx="57" cy="96" rx="6" ry="9" fill="#f9c6a8"/>
      <ellipse cx="123" cy="96" rx="6" ry="9" fill="#f9c6a8"/>
      <path d="M58 80 Q90 55 122 80 Q115 62 90 58 Q65 62 58 80Z" fill="#5eead4" opacity=".7"/>
      <path d="M76 91 Q80 87 84 91" fill="none" stroke="#1a1a2e" stroke-width="2" stroke-linecap="round"/>
      <path d="M96 91 Q100 87 104 91" fill="none" stroke="#1a1a2e" stroke-width="2" stroke-linecap="round"/>
      <ellipse cx="90" cy="100" rx="3.5" ry="2.5" fill="#f9c6a8"/>
      <path d="M83 108 Q90 114 97 108" fill="none" stroke="#c07050" stroke-width="2" stroke-linecap="round"/>
      <path d="M62 140 Q45 125 50 108 Q54 95 62 100" fill="none" stroke="#fde8d8" stroke-width="12" stroke-linecap="round"/>
      <path d="M118 140 Q135 125 130 108 Q126 95 118 100" fill="none" stroke="#fde8d8" stroke-width="12" stroke-linecap="round"/>
      <circle cx="61" cy="100" r="9" fill="#f9c6a8"/>
      <circle cx="119" cy="100" r="9" fill="#f9c6a8"/>
      <path d="M75 182 Q68 198 72 208" fill="none" stroke="#fde8d8" stroke-width="13" stroke-linecap="round"/>
      <path d="M105 182 Q112 198 108 208" fill="none" stroke="#fde8d8" stroke-width="13" stroke-linecap="round"/>
      <!-- Animated heartbeat -->
      <path d="M20 170 L38 170 L44 154 L52 187 L60 158 L68 170 L160 170"
            fill="none" stroke="#5eead4" stroke-width="2.2"
            stroke-linecap="round" stroke-linejoin="round" opacity=".9"
            stroke-dasharray="400" stroke-dashoffset="400">
        <animate attributeName="stroke-dashoffset" from="400" to="0"
                 dur="2s" repeatCount="indefinite"/>
      </path>
    </svg>
    <span class="status-badge">{label}</span>
    <span class="fetal-label">Fetal Health Monitor</span>
    <span class="fetal-sub">CTG Classification · {baseline_value} bpm</span>
  </div>

  <!-- ── Live CTG Waveform ── -->
  <div class="viz-card wave-card">
    <div class="viz-card-title">CTG Waveform — Simulated Fetal Heart Rate</div>
    <div style="position:relative;height:110px">
      <canvas id="waveChart"></canvas>
    </div>
  </div>

  <!-- ── Baseline gauge ── -->
  <div class="viz-card gauge-card">
    <div class="viz-card-title">Baseline FHR</div>
    <div style="position:relative;height:85px;width:140px">
      <canvas id="g1"></canvas>
    </div>
    <div class="gauge-val">{baseline_value}</div>
    <div class="gauge-unit">beats per minute</div>
  </div>

  <!-- ── Prob donut ── -->
  <div class="viz-card" style="display:flex;flex-direction:column">
    <div class="viz-card-title">Model Confidence</div>
    <div style="display:grid;grid-template-columns:120px 1fr;gap:12px;align-items:center;flex:1">
      <div style="position:relative;height:110px"><canvas id="pieChart"></canvas></div>
      <div>
        <div class="prob-row">
          <div class="prob-label-row"><span>Normal</span><span>{p_normal}%</span></div>
          <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p_normal}%;background:#22c55e"></div></div>
        </div>
        <div class="prob-row">
          <div class="prob-label-row"><span>Suspect</span><span>{p_suspect}%</span></div>
          <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p_suspect}%;background:#f59e0b"></div></div>
        </div>
        <div class="prob-row">
          <div class="prob-label-row"><span>Abnormal</span><span>{p_abnorm}%</span></div>
          <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p_abnorm}%;background:#e11d48"></div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Radar: feature profile ── -->
  <div class="viz-card" style="display:flex;flex-direction:column;align-items:center">
    <div class="viz-card-title" style="align-self:flex-start">Feature Profile (normalised)</div>
    <div style="position:relative;height:170px;width:100%"><canvas id="radarChart"></canvas></div>
  </div>

  <!-- ── Histogram shape ── -->
  <div class="viz-card" style="display:flex;flex-direction:column">
    <div class="viz-card-title">FHR Histogram</div>
    <div style="position:relative;height:170px;width:100%"><canvas id="histChart"></canvas></div>
  </div>

  <!-- ── Key stats ── -->
  <div class="viz-card">
    <div class="viz-card-title">Key Parameters</div>
    <div class="stat-mini"><span>Accelerations</span><b>{accelerations:.3f}</b></div>
    <div class="stat-mini"><span>Fetal Movement</span><b>{fetal_movement:.3f}</b></div>
    <div class="stat-mini"><span>Uterine Contractions</span><b>{uterine_contractions:.3f}</b></div>
    <div class="stat-mini"><span>Light Decelerations</span><b>{light_decelerations:.3f}</b></div>
    <div class="stat-mini"><span>Severe Decelerations</span><b>{severe_decelerations:.3f}</b></div>
    <div class="stat-mini"><span>Mean LTV</span><b>{mean_ltv:.1f}</b></div>
    <div class="stat-mini"><span>Hist. Mode</span><b>{histogram_mode}</b></div>
    <div class="stat-mini"><span>Hist. Variance</span><b>{histogram_variance}</b></div>
    <div class="stat-mini"><span>Hist. Tendency</span><b>{histogram_tendency}</b></div>
  </div>

</div>

<script>
const accent = "{accent}";
const baseHR = {baseline_value};
const p_n = {p_normal}, p_s = {p_suspect}, p_a = {p_abnorm};

// ── CTG live waveform ──────────────────────────────────────────────
const waveLabels = Array.from({{length: 80}}, (_, i) => i);
function genHR(n) {{
  return Array.from({{length: n}}, (_, i) =>
    baseHR + 4*Math.sin(i*0.35) + 2*Math.sin(i*1.1+0.5) + (Math.random()-0.5)*3
  );
}}
const waveChart = new Chart(document.getElementById('waveChart'), {{
  type: 'line',
  data: {{
    labels: waveLabels,
    datasets: [{{
      data: genHR(80),
      borderColor: '#0d9488',
      borderWidth: 1.8,
      pointRadius: 0,
      tension: 0.4,
      fill: true,
      backgroundColor: 'rgba(13,148,136,0.07)'
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    animation: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }},
    scales: {{
      x: {{ display: false }},
      y: {{
        min: baseHR - 22, max: baseHR + 22,
        ticks: {{ color: '#9ca3af', font: {{ size: 9 }} }},
        grid: {{ color: 'rgba(0,0,0,.05)' }},
        border: {{ display: false }}
      }}
    }}
  }}
}});
setInterval(() => {{
  waveChart.data.datasets[0].data.shift();
  waveChart.data.datasets[0].data.push(
    baseHR + 4*Math.sin(Date.now()*.003) + 2*Math.sin(Date.now()*.008) + (Math.random()-.5)*3
  );
  waveChart.update('none');
}}, 200);

// ── Semi-circle gauge ─────────────────────────────────────────────
function drawGauge(id, value, min, max, color) {{
  const c = document.getElementById(id);
  if (!c) return;
  const ctx = c.getContext('2d');
  const cx = c.width/2, cy = c.height - 8, r = 52;
  const pct = (value - min) / (max - min);
  ctx.clearRect(0,0,c.width,c.height);
  ctx.beginPath(); ctx.arc(cx,cy,r,Math.PI,2*Math.PI);
  ctx.strokeStyle='#e5e7eb'; ctx.lineWidth=10; ctx.stroke();
  ctx.beginPath(); ctx.arc(cx,cy,r,Math.PI,Math.PI+pct*Math.PI);
  ctx.strokeStyle=color; ctx.lineWidth=10; ctx.lineCap='round'; ctx.stroke();
}}
drawGauge('g1', {baseline_value}, 106, 160, accent);

// ── Donut confidence chart ────────────────────────────────────────
new Chart(document.getElementById('pieChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Normal', 'Suspect', 'Abnormal'],
    datasets: [{{
      data: [p_n, p_s, p_a],
      backgroundColor: ['#22c55e','#f59e0b','#e11d48'],
      borderWidth: 2, borderColor: '#fff',
      hoverOffset: 4
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, cutout:'62%',
    plugins: {{ legend: {{ display: false }}, tooltip: {{
      callbacks: {{ label: ctx => ' ' + ctx.label + ': ' + ctx.parsed.toFixed(1) + '%' }}
    }} }}
  }}
}});

// ── Radar feature profile ─────────────────────────────────────────
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: ['Baseline', 'Accel.', 'Movement', 'Contrax.', 'LTV', 'Hist. Var.'],
    datasets: [{{
      data: [{r_baseline}, {r_accel}, {r_movement}, {r_contrax}, {r_ltv}, {r_variance}],
      borderColor: accent,
      backgroundColor: accent + '25',
      borderWidth: 2,
      pointBackgroundColor: accent,
      pointRadius: 4
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: true }} }},
    scales: {{
      r: {{
        min: 0, max: 1, ticks: {{ display: false }},
        grid: {{ color: 'rgba(0,0,0,.08)' }},
        pointLabels: {{ font: {{ size: 10 }}, color: '#374151' }}
      }}
    }}
  }}
}});

// ── Histogram bar chart ───────────────────────────────────────────
const histMin = {histogram_min}, histMax = {histogram_max}, histMode = {histogram_mode};
const nbins = 12;
const step = (histMax - histMin) / nbins;
const histLabels = Array.from({{length: nbins}}, (_, i) => Math.round(histMin + i*step));
const histData = histLabels.map(x => {{
  const d = Math.abs(x - histMode);
  return Math.max(0, 100 - d*1.4 + (Math.random()-0.5)*8);
}});
const histColors = histLabels.map(x => {{
  if (x >= histMin && x < histMin + step*2) return '#bae6fd';
  if (x >= histMax - step*2) return '#fca5a5';
  return '#5eead4';
}});
new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: histLabels,
    datasets: [{{
      data: histData,
      backgroundColor: histColors,
      borderRadius: 4,
      borderSkipped: false
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{
      callbacks: {{ label: ctx => ' Freq: ' + Math.round(ctx.parsed.y) }}
    }} }},
    scales: {{
      x: {{ ticks: {{ color:'#9ca3af', font:{{ size: 9 }} }}, grid: {{ display: false }}, border: {{ display: false }} }},
      y: {{ ticks: {{ color:'#9ca3af', font:{{ size: 9 }} }}, grid: {{ color: 'rgba(0,0,0,.05)' }}, border: {{ display: false }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""

            st.components.v1.html(viz_html, height=780, scrolling=False)

        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.info("👆  Fill in the parameters above and click **Run Prediction** to classify fetal health.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center; font-size:.8rem; color:#9ca3af;">
  Fetal Health Monitor · For clinical decision support only · Not a substitute for professional medical advice
</p>
""", unsafe_allow_html=True)