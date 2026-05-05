"""
app.py — Canary: Real-Time Voice Forensics
Deepfake audio detection system with explainability.

UI aesthetic: CRT phosphor terminal / forensics lab.
Dark background, green glow, monospace, scan-line feel.
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import plotly.graph_objects as go
import time
import io
import tempfile
import os

from src.detector import DeepfakeDetector
from src.explainer import plot_detection_dashboard, plot_streaming_timeline
from src.fingerprint import fingerprint_audio
from src.robustness import apply_degradation, DegradationConfig, benchmark_degradation_configs
from src.evaluator import (
    generate_synthetic_benchmark, evaluate, roc_curve_data,
    benchmark_under_degradation
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Canary — Voice Forensics",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — CRT phosphor terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* ── Root ── */
:root {
  --bg:       #020805;
  --bg2:      #050e05;
  --bg3:      #091409;
  --green:    #00ff41;
  --green-dim:#1a5c1a;
  --green-mid:#00a020;
  --cyan:     #00e5ff;
  --amber:    #ffb700;
  --red:      #ff3131;
  --text:     #8cc88c;
  --text-dim: #3a5c3a;
  --border:   #1a3d1a;
}

/* ── Body ── */
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

.main { background-color: var(--bg) !important; }
.block-container { padding: 1.5rem 2rem !important; }

/* ── CRT scanline overlay ── */
.main::before {
  content: "";
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.15) 2px,
    rgba(0,0,0,0.15) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headers ── */
h1, h2, h3 {
  font-family: 'Orbitron', monospace !important;
  letter-spacing: 0.1em !important;
  color: var(--green) !important;
  text-shadow: 0 0 12px rgba(0,255,65,0.4) !important;
}
h1 { font-size: 1.6rem !important; }
h2 { font-size: 1.1rem !important; color: var(--green-mid) !important; }
h3 { font-size: 0.9rem !important; color: var(--text) !important; }

/* ── Metric boxes ── */
[data-testid="metric-container"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  padding: 0.5rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.7rem !important; }
[data-testid="stMetricValue"] { color: var(--green) !important; font-size: 1.3rem !important; }

/* ── Buttons ── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--green) !important;
  color: var(--green) !important;
  font-family: 'Share Tech Mono', monospace !important;
  letter-spacing: 0.08em !important;
  transition: all 0.15s !important;
  text-transform: uppercase !important;
  font-size: 0.8rem !important;
  padding: 0.4rem 1rem !important;
}
.stButton > button:hover {
  background: var(--green) !important;
  color: var(--bg) !important;
  box-shadow: 0 0 15px rgba(0,255,65,0.4) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  border: 1px dashed var(--green-dim) !important;
  background: var(--bg2) !important;
  border-radius: 2px !important;
  padding: 1rem !important;
}

/* ── Selectbox / slider ── */
.stSelectbox > div, .stSlider > div { color: var(--text) !important; }
[data-baseweb="select"] { background: var(--bg3) !important; border-color: var(--border) !important; }
[data-baseweb="select"] * { color: var(--text) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Share Tech Mono', monospace !important;
  color: var(--text-dim) !important;
  background: transparent !important;
  border: none !important;
  border-right: 1px solid var(--border) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.08em !important;
  padding: 0.5rem 1.2rem !important;
  text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
  color: var(--green) !important;
  background: var(--bg3) !important;
  border-bottom: 2px solid var(--green) !important;
  text-shadow: 0 0 8px rgba(0,255,65,0.5) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  background: var(--bg2) !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; }

/* ── Alert panel ── */
.alert-synthetic {
  border: 1px solid var(--red);
  background: rgba(255,49,49,0.08);
  padding: 1rem 1.2rem;
  margin: 0.5rem 0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.85rem;
  line-height: 1.6;
  color: #ff8080;
  position: relative;
}
.alert-human {
  border: 1px solid var(--green-dim);
  background: rgba(0,255,65,0.04);
  padding: 1rem 1.2rem;
  margin: 0.5rem 0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.85rem;
  color: var(--text);
}
.alert-uncertain {
  border: 1px solid var(--amber);
  background: rgba(255,183,0,0.06);
  padding: 1rem 1.2rem;
  margin: 0.5rem 0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.85rem;
  color: #ffd080;
}

/* ── Probability gauge ── */
.gauge-container {
  position: relative;
  height: 8px;
  background: var(--bg3);
  border: 1px solid var(--border);
  margin: 0.3rem 0;
  overflow: hidden;
}
.gauge-fill {
  height: 100%;
  transition: width 0.4s ease;
}

/* ── Terminal text ── */
.terminal {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.78rem;
  color: var(--text);
  background: var(--bg2);
  border: 1px solid var(--border);
  padding: 0.8rem 1rem;
  line-height: 1.7;
  white-space: pre-wrap;
  max-height: 300px;
  overflow-y: auto;
}

/* ── Blinking cursor ── */
.blink {
  animation: blink-anim 1s step-end infinite;
  color: var(--green);
}
@keyframes blink-anim { 50% { opacity: 0; } }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--green-dim); }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Info / warning / error ── */
[data-testid="stInfo"] { background: rgba(0,229,255,0.06) !important; border-color: var(--cyan) !important; color: var(--cyan) !important; }
[data-testid="stWarning"] { background: rgba(255,183,0,0.06) !important; border-color: var(--amber) !important; }
[data-testid="stError"] { background: rgba(255,49,49,0.08) !important; border-color: var(--red) !important; }

/* ── Radio ── */
.stRadio label { color: var(--text) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.8rem !important; }

/* ── Checkbox ── */
.stCheckbox label { color: var(--text) !important; font-size: 0.82rem !important; }

/* ── Number input ── */
.stNumberInput input { background: var(--bg3) !important; color: var(--text) !important; border-color: var(--border) !important; }

/* ── Caption ── */
.stCaption { color: var(--text-dim) !important; font-size: 0.72rem !important; }

/* ── Data frame ── */
[data-testid="stDataFrame"] { background: var(--bg2) !important; }

/* ── Waveform badge ── */
.badge-synth { color: var(--red); font-weight: bold; }
.badge-human { color: var(--green); font-weight: bold; }
.badge-unsure { color: var(--amber); font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_detector(use_model: bool):
    return DeepfakeDetector(use_model=use_model)


def load_audio(uploaded_file) -> tuple[np.ndarray, int]:
    """Load uploaded audio, return (mono float32, sr)."""
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_file.name)[1], delete=False) as f:
        f.write(uploaded_file.read())
        tmp_path = f.name
    try:
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
    finally:
        os.unlink(tmp_path)
    return audio.astype(np.float32), sr


def prob_to_html_bar(prob: float, height: int = 8) -> str:
    pct = int(prob * 100)
    if prob > 0.65:
        color = "#ff3131"
    elif prob > 0.40:
        color = "#ffb700"
    else:
        color = "#00ff41"
    return f"""
    <div class="gauge-container" style="height:{height}px;">
      <div class="gauge-fill" style="width:{pct}%; background:{color};
        box-shadow: 0 0 6px {color}88;"></div>
    </div>
    """


def verdict_html(prob: float, confidence: float) -> str:
    if prob > 0.65:
        cls = "alert-synthetic"
        verdict = "⚠ SYNTHETIC AUDIO DETECTED"
        badge = f'<span class="badge-synth">FAKE [{prob:.1%}]</span>'
    elif prob > 0.40:
        cls = "alert-uncertain"
        verdict = "◈ INCONCLUSIVE — FURTHER ANALYSIS REQUIRED"
        badge = f'<span class="badge-unsure">UNCERTAIN [{prob:.1%}]</span>'
    else:
        cls = "alert-human"
        verdict = "✓ NATURAL SPEECH — NO ANOMALIES DETECTED"
        badge = f'<span class="badge-human">HUMAN [{1-prob:.1%}]</span>'
    return f"""
    <div class="{cls}">
      <div style="font-family:Orbitron,monospace; font-size:0.9rem; margin-bottom:0.5rem;">
        {verdict}
      </div>
      {badge} &nbsp; confidence: {confidence:.1%}
    </div>
    """


def mini_waveform(audio: np.ndarray, sr: int, height: int = 80) -> go.Figure:
    """Small waveform plot."""
    t = np.linspace(0, len(audio) / sr, len(audio))
    # Downsample for speed
    step = max(1, len(audio) // 2000)
    fig = go.Figure(go.Scatter(
        x=t[::step], y=audio[::step],
        mode="lines",
        line=dict(color="#00ff41", width=0.8),
        fill="tozeroy",
        fillcolor="rgba(0,255,65,0.06)",
    ))
    fig.update_layout(
        paper_bgcolor="#050e05", plot_bgcolor="#050e05",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=height, showlegend=False,
    )
    return fig


def roc_figure(fpr, tpr, auc_val: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             line=dict(color="#00ff41", width=2), name=f"AUC={auc_val:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="#3a5c3a", width=1, dash="dash"), name="Chance"))
    fig.update_layout(
        paper_bgcolor="#020805", plot_bgcolor="#020805",
        font=dict(family="Share Tech Mono", color="#8cc88c"),
        xaxis=dict(title="FPR", gridcolor="#1a3d1a", color="#8cc88c"),
        yaxis=dict(title="TPR", gridcolor="#1a3d1a", color="#8cc88c"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=280, legend=dict(font=dict(color="#00ff41")),
        title=dict(text="ROC CURVE", font=dict(color="#00ff41", size=11, family="Orbitron")),
    )
    return fig


def degradation_bar_chart(results: dict) -> go.Figure:
    names = list(results.keys())
    scores = [results[n]["score"] for n in names]
    colors = ["#ff3131" if s > 0.65 else ("#ffb700" if s > 0.40 else "#00ff41") for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker_color=colors,
        marker_line=dict(color="#050e05", width=1),
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
        textfont=dict(color="#8cc88c", size=9, family="Share Tech Mono"),
    ))
    fig.add_shape(type="line", x0=0.65, x1=0.65, y0=-0.5, y1=len(names) - 0.5,
                  line=dict(color="#ff3131", width=1, dash="dash"))
    fig.update_layout(
        paper_bgcolor="#020805", plot_bgcolor="#020805",
        font=dict(family="Share Tech Mono", color="#8cc88c"),
        xaxis=dict(range=[0, 1.1], gridcolor="#1a3d1a", title="P(synthetic)"),
        yaxis=dict(gridcolor="#1a3d1a"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
        title=dict(text="DETECTION UNDER DEGRADATION", font=dict(color="#00ff41", size=11, family="Orbitron")),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1rem;">
      <div style="font-family:Orbitron,monospace; font-size:1.4rem; color:#00ff41;
                  text-shadow:0 0 15px rgba(0,255,65,0.6); letter-spacing:0.15em;">
        🐦 CANARY
      </div>
      <div style="font-size:0.65rem; color:#3a5c3a; letter-spacing:0.2em; margin-top:0.2rem;">
        VOICE FORENSICS v1.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem; color:#3a5c3a; letter-spacing:0.1em;'>SYSTEM CONFIG</div>", unsafe_allow_html=True)

    use_model = st.checkbox("Use Whisper Encoder", value=True,
                            help="Load openai/whisper-small for deep embeddings. ~500MB download on first use.")
    window_sec = st.slider("Analysis Window (s)", 2.0, 10.0, 4.0, 0.5)
    hop_sec = st.slider("Hop Size (s)", 0.5, 3.0, 2.0, 0.5)

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem; color:#3a5c3a; letter-spacing:0.1em;'>DETECTION THRESHOLD</div>", unsafe_allow_html=True)
    threshold = st.slider("Alert Threshold", 0.40, 0.90, 0.65, 0.05)

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem; color:#3a5c3a; letter-spacing:0.1em;'>DEGRADATION SIM</div>", unsafe_allow_html=True)
    codec = st.selectbox("Codec", ["None", "μ-law (Telephony)", "Opus (VoIP)", "AAC"])
    noise = st.selectbox("Noise", ["None", "White", "Pink (1/f)", "Babble"])
    snr = st.slider("SNR (dB)", 5, 40, 20)
    reverb = st.checkbox("Reverb (Room)")
    packet_loss = st.slider("Packet Loss %", 0.0, 20.0, 0.0, 0.5)

    def build_degrad_config():
        codec_map = {
            "None": None,
            "μ-law (Telephony)": "ulaw",
            "Opus (VoIP)": "opus_sim",
            "AAC": "aac_sim",
        }
        noise_map = {"None": None, "White": "white", "Pink (1/f)": "pink", "Babble": "babble"}
        return DegradationConfig(
            codec=codec_map[codec],
            target_sr=8000 if codec == "μ-law (Telephony)" else None,
            bandwidth_limit_hz=3400 if codec == "μ-law (Telephony)" else None,
            noise_type=noise_map[noise],
            noise_snr_db=snr,
            reverb_rt60=0.4 if reverb else None,
            packet_loss_pct=packet_loss,
        )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.65rem; color:#1a3d1a; line-height:1.8;">
    MODEL: openai/whisper-small<br>
    HEURISTICS: spectral, pitch, phase<br>
    FINGERPRINT: 5 TTS profiles<br>
    LATENCY TARGET: &lt;500ms<br>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
  <div>
    <div style="font-family:Orbitron,monospace; font-size:1.8rem; color:#00ff41;
                text-shadow:0 0 20px rgba(0,255,65,0.5); letter-spacing:0.12em;">
      🐦 CANARY
    </div>
    <div style="font-size:0.72rem; color:#3a5c3a; letter-spacing:0.25em; margin-top:0.1rem;">
      REAL-TIME DEEPFAKE AUDIO FORENSICS SYSTEM
    </div>
  </div>
  <div style="flex:1; height:1px; background: linear-gradient(to right, #00ff41, transparent);
              margin-top:0.8rem;"></div>
  <div style="font-size:0.65rem; color:#3a5c3a; font-family:Share Tech Mono,monospace;">
    STATUS: <span class="blink" style="color:#00ff41;">ONLINE</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_analyze, tab_stream, tab_fingerprint, tab_robustness, tab_metrics = st.tabs([
    "[ ANALYZE ]",
    "[ STREAM SIM ]",
    "[ FINGERPRINT ]",
    "[ ROBUSTNESS ]",
    "[ METRICS ]",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        st.markdown("#### AUDIO INPUT")
        uploaded = st.file_uploader(
            "DROP AUDIO FILE",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            label_visibility="collapsed",
        )
        st.caption("Supports WAV · MP3 · FLAC · OGG · M4A  |  8kHz–48kHz  |  Mono or stereo")

    with col_info:
        st.markdown("#### PIPELINE")
        st.markdown("""
        <div class="terminal">
> STAGE 1: Spectral Analysis
  · Flatness · Pitch · Phase · HNR
> STAGE 2: Whisper Encoder
  · Embedding anomaly score
> STAGE 3: Fusion
  · 40% heuristic + 60% embedding
> OUTPUT: P(synthetic) + explanation
        </div>
        """, unsafe_allow_html=True)

    if uploaded is not None:
        with st.spinner("Loading audio..."):
            audio, sr = load_audio(uploaded)

        # Apply degradation if configured
        degrad_cfg = build_degrad_config()
        any_degrad = any([
            degrad_cfg.codec, degrad_cfg.noise_type,
            degrad_cfg.reverb_rt60, degrad_cfg.packet_loss_pct > 0
        ])
        if any_degrad:
            with st.spinner("Applying degradation..."):
                audio_proc = apply_degradation(audio, sr, degrad_cfg)
            st.info(f"Degradation applied: codec={degrad_cfg.codec or 'none'} | "
                    f"noise={degrad_cfg.noise_type or 'none'} | reverb={degrad_cfg.reverb_rt60 or 'off'}")
        else:
            audio_proc = audio

        # Waveform preview
        st.markdown("#### WAVEFORM")
        wf_fig = mini_waveform(audio_proc, sr, height=80)
        st.plotly_chart(wf_fig, use_container_width=True, config={"displayModeBar": False})

        dur = len(audio_proc) / sr
        st.caption(f"Duration: {dur:.2f}s | SR: {sr}Hz | Samples: {len(audio_proc):,}")

        # ── RUN DETECTION ──────────────────────────────────────────────────
        if st.button("▶  RUN FORENSIC ANALYSIS", use_container_width=True):
            detector = get_detector(use_model)

            with st.spinner("Running detection pipeline..."):
                t0 = time.time()
                result = detector.detect(audio_proc, sr)
                elapsed = (time.time() - t0) * 1000

            # ── VERDICT ───────────────────────────────────────────────────
            st.markdown("#### VERDICT")
            st.markdown(verdict_html(result.synthetic_probability, result.confidence),
                        unsafe_allow_html=True)
            st.markdown(prob_to_html_bar(result.synthetic_probability, 12), unsafe_allow_html=True)

            # ── METRICS ROW ───────────────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("P(Synthetic)", f"{result.synthetic_probability:.3f}")
            m2.metric("Confidence", f"{result.confidence:.1%}")
            m3.metric("Heuristic", f"{result.heuristic_score:.3f}")
            m4.metric("Embedding", f"{result.embedding_score:.3f}")
            m5.metric("Latency", f"{elapsed:.0f} ms")

            # ── EXPLANATION ───────────────────────────────────────────────
            st.markdown("#### FORENSIC EXPLANATION")
            st.markdown(f"""
            <div class="terminal">
{result.explanation}

TRIGGERED RULES ({len(result.triggered_rules)}):
{"".join(f"  [{name}]  value={val:.4f}  threshold={thr:.4f}  dir={dire}  score={sc:.2f}"
         for name, val, thr, dire, sc in result.triggered_rules) or "  (none)"}
            </div>
            """, unsafe_allow_html=True)

            # ── DASHBOARD ─────────────────────────────────────────────────
            if result.features is not None:
                st.markdown("#### SPECTRAL FORENSICS DASHBOARD")
                dash = plot_detection_dashboard(result, sr)
                st.plotly_chart(dash, use_container_width=True,
                                config={"displayModeBar": False})

    else:
        st.markdown("""
        <div style="height:200px; display:flex; align-items:center; justify-content:center;
                    border: 1px dashed #1a3d1a; color:#1a5c1a; font-size:0.8rem; letter-spacing:0.2em;">
          AWAITING AUDIO INPUT <span class="blink">_</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: STREAM SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_stream:
    st.markdown("#### STREAMING ANALYSIS — SLIDING WINDOW SIMULATION")
    st.caption(f"Window: {window_sec}s | Hop: {hop_sec}s | Threshold: {threshold}")

    stream_upload = st.file_uploader("UPLOAD AUDIO FOR STREAM SIMULATION",
                                     type=["wav", "mp3", "flac", "ogg"],
                                     key="stream_upload")

    if stream_upload is not None:
        with st.spinner("Loading..."):
            audio_s, sr_s = load_audio(stream_upload)

        degrad_cfg_s = build_degrad_config()
        if any([degrad_cfg_s.codec, degrad_cfg_s.noise_type, degrad_cfg_s.reverb_rt60,
                degrad_cfg_s.packet_loss_pct > 0]):
            audio_s = apply_degradation(audio_s, sr_s, degrad_cfg_s)

        dur_s = len(audio_s) / sr_s
        n_chunks = max(1, int((dur_s - window_sec) / hop_sec) + 1)
        st.markdown(f"<div class='terminal'>Audio: {dur_s:.1f}s → {n_chunks} chunks</div>",
                    unsafe_allow_html=True)

        if st.button("▶  START STREAM ANALYSIS", use_container_width=True, key="stream_btn"):
            detector_s = get_detector(use_model)

            # Progress
            prog = st.progress(0)
            status = st.empty()
            chart_placeholder = st.empty()
            log_placeholder = st.empty()

            results_stream = []
            alert_count = 0
            log_lines = []

            for i, chunk_result in enumerate(
                detector_s.detect_stream(audio_s, sr_s, window_sec, hop_sec)
            ):
                results_stream.append(chunk_result)
                pct = (i + 1) / n_chunks

                prob = chunk_result.synthetic_probability
                flag = "⚠ ALERT" if prob > threshold else "✓ CLEAR"
                if prob > threshold:
                    alert_count += 1

                t_start = i * hop_sec
                log_lines.append(
                    f"[{t_start:6.1f}s] chunk {i:03d} | P(synth)={prob:.3f} | conf={chunk_result.confidence:.2f} | {flag}"
                )

                prog.progress(min(pct, 1.0))
                alert_color = "#ff3131" if alert_count else "#00ff41"
                status.markdown(
                    f"<span style='color:#3a5c3a; font-size:0.75rem;'>"
                    f"Chunk {i+1}/{n_chunks} | Alerts: "
                    f"<span style='color:{alert_color};'>{alert_count}</span>"
                    f"</span>",
                    unsafe_allow_html=True
                )

                # Update chart
                timeline_fig = plot_streaming_timeline(results_stream)
                chart_placeholder.plotly_chart(timeline_fig, use_container_width=True,
                                               config={"displayModeBar": False})

                # Update log
                log_text = '<br>'.join(log_lines[-15:])
                log_placeholder.markdown(
                    f"<div class='terminal'>{log_text}</div>",
                    unsafe_allow_html=True
                )

                time.sleep(0.05)  # visual pacing

            prog.progress(1.0)

            # Summary
            probs_all = [r.synthetic_probability for r in results_stream]
            st.markdown(f"""
            <div class="{'alert-synthetic' if np.mean(probs_all) > threshold else 'alert-human'}">
            ▸ STREAM ANALYSIS COMPLETE<br>
            ▸ Chunks: {len(results_stream)} | Alerts: {alert_count} ({alert_count/len(results_stream):.0%})<br>
            ▸ Mean P(synthetic): {np.mean(probs_all):.3f} | Max: {np.max(probs_all):.3f}<br>
            ▸ Avg latency: {np.mean([r.processing_time_ms for r in results_stream]):.1f} ms/chunk
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: FINGERPRINT
# ══════════════════════════════════════════════════════════════════════════════
with tab_fingerprint:
    st.markdown("#### TTS SYSTEM FINGERPRINTING")
    st.caption("Identify which AI/TTS system likely generated the audio")

    fp_upload = st.file_uploader("UPLOAD AUDIO FOR FINGERPRINTING",
                                  type=["wav", "mp3", "flac", "ogg"], key="fp_upload")

    # Known TTS profiles info
    with st.expander("TTS FINGERPRINT DATABASE"):
        st.markdown("""
        <div class="terminal">
MODEL          CHARACTERISTIC ARTIFACTS
──────────────────────────────────────────────────────────────
SpeechT5       Spectral peaks 3-5kHz | High flatness | Ultra-regular pitch
XTTS-v2        HF artifacts >6kHz | Low phase randomness | Vocoder patterns
OpenVoice      Mid-band smoothing 1-3kHz | Formant over-smoothing
WhisperTTS     Low-freq emphasis | Compressed dynamic range | Silent pauses
AudioLDM2      Structured phase | Very low spectral flux | Clean harmonics
──────────────────────────────────────────────────────────────
(Rule-based classifier. Production: train on labeled TTS outputs)
        </div>
        """, unsafe_allow_html=True)

    if fp_upload is not None:
        with st.spinner("Loading..."):
            audio_fp, sr_fp = load_audio(fp_upload)

        degrad_cfg_fp = build_degrad_config()
        if any([degrad_cfg_fp.codec, degrad_cfg_fp.noise_type]):
            audio_fp = apply_degradation(audio_fp, sr_fp, degrad_cfg_fp)

        if st.button("▶  FINGERPRINT AUDIO", use_container_width=True, key="fp_btn"):
            with st.spinner("Extracting spectral fingerprint..."):
                fp_result = fingerprint_audio(audio_fp, sr_fp)

            # Result
            color = "#ff3131" if fp_result.is_synthetic else "#00ff41"
            model_label = fp_result.likely_source

            st.markdown(f"""
            <div class="{'alert-synthetic' if fp_result.is_synthetic else 'alert-human'}">
              <div style="font-family:Orbitron,monospace; font-size:1rem; margin-bottom:0.5rem; color:{color};">
                {'⚠ SYNTHETIC ORIGIN IDENTIFIED' if fp_result.is_synthetic else '✓ HUMAN SPEECH DETECTED'}
              </div>
              <strong>Likely Source:</strong> {model_label}<br>
              <strong>Confidence:</strong> {fp_result.confidence:.1%}
            </div>
            """, unsafe_allow_html=True)

            # Score breakdown
            st.markdown("#### MODEL SIMILARITY SCORES")
            if fp_result.scores:
                models = list(fp_result.scores.keys())
                scores = list(fp_result.scores.values())
                colors = ["#ff3131" if m == model_label else "#00a020" for m in models]

                fig_fp = go.Figure(go.Bar(
                    x=models, y=scores,
                    marker_color=colors,
                    marker_line=dict(color="#050e05", width=1),
                    text=[f"{s:.2f}" for s in scores],
                    textposition="outside",
                    textfont=dict(color="#8cc88c", size=9, family="Share Tech Mono"),
                ))
                fig_fp.update_layout(
                    paper_bgcolor="#020805", plot_bgcolor="#020805",
                    font=dict(family="Share Tech Mono", color="#8cc88c"),
                    yaxis=dict(range=[0, 1], gridcolor="#1a3d1a", title="Similarity Score"),
                    xaxis=dict(gridcolor="#1a3d1a"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=220,
                )
                st.plotly_chart(fig_fp, use_container_width=True, config={"displayModeBar": False})

            # Artifacts
            if fp_result.artifacts:
                st.markdown("#### DETECTED ARTIFACTS")
                artifact_txt = "\n".join(f"  • {a}" for a in fp_result.artifacts)
                st.markdown(f"<div class='terminal'>{artifact_txt}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
with tab_robustness:
    st.markdown("#### ADVERSARIAL ROBUSTNESS TESTING")
    st.caption("Test detection performance under real-world audio degradation")

    rob_upload = st.file_uploader("UPLOAD AUDIO FOR ROBUSTNESS TEST",
                                   type=["wav", "mp3", "flac", "ogg"], key="rob_upload")
    rob_label = st.radio("Audio is:", ["Synthetic (TTS)", "Human (real)"],
                          horizontal=True)

    if rob_upload is not None:
        with st.spinner("Loading..."):
            audio_r, sr_r = load_audio(rob_upload)
        label_r = 1 if "Synthetic" in rob_label else 0

        if st.button("▶  RUN ROBUSTNESS BENCHMARK", use_container_width=True, key="rob_btn"):
            detector_r = get_detector(use_model)

            with st.spinner("Running 7 degradation conditions..."):
                degrad_results = benchmark_under_degradation(detector_r, audio_r, sr_r, label_r)

            # Chart
            fig_rob = degradation_bar_chart(degrad_results)
            st.plotly_chart(fig_rob, use_container_width=True, config={"displayModeBar": False})

            # Table
            st.markdown("#### CONDITION BREAKDOWN")
            rows = []
            for name, data in degrad_results.items():
                score = data["score"]
                correct = data.get("correct", False)
                latency = data.get("latency_ms", 0)
                status_icon = "✓" if correct else "✗"
                rows.append(f"  {status_icon}  {name:<20} score={score:.3f}  lat={latency:.0f}ms  {'CORRECT' if correct else 'MISS'}")
            rows_html = '<br>'.join(rows)
            st.markdown(f"<div class='terminal'>{rows_html}</div>", unsafe_allow_html=True)

            # Summary
            correct_count = sum(1 for d in degrad_results.values() if d.get("correct", False))
            total = len(degrad_results)
            st.markdown(f"""
            <div class="{'alert-human' if correct_count/total > 0.7 else 'alert-uncertain'}">
            ▸ Robustness Score: {correct_count}/{total} correct ({correct_count/total:.0%})<br>
            ▸ Avg latency: {np.mean([d['latency_ms'] for d in degrad_results.values()]):.1f} ms
            </div>
            """, unsafe_allow_html=True)

            # Degraded audio download
            st.markdown("#### PREVIEW DEGRADED AUDIO")
            degrad_config_demo = DegradationConfig(codec="ulaw", target_sr=8000,
                                                    noise_type="white", noise_snr_db=15)
            demo_degraded = apply_degradation(audio_r, sr_r, degrad_config_demo)
            buf = io.BytesIO()
            sf.write(buf, demo_degraded, sr_r, format="WAV")
            st.download_button("⬇ Download μ-law Degraded Audio",
                               buf.getvalue(), "degraded_ulaw.wav", "audio/wav")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("#### EVALUATION METRICS DASHBOARD")
    st.caption("EER · t-DCF · AUC · Accuracy under degradation")

    met_col1, met_col2 = st.columns([1, 1])

    with met_col1:
        st.markdown("#### BENCHMARK (SYNTHETIC DATA)")
        st.caption("Simulated score distributions: Beta(2,5) for human, Beta(5,2) for synthetic")

        n_human = st.slider("# Human samples", 20, 200, 100)
        n_synth = st.slider("# Synthetic samples", 20, 200, 100)

        if st.button("▶  RUN BENCHMARK EVAL", use_container_width=True):
            scores_b, labels_b = generate_synthetic_benchmark(n_human, n_synth)
            eval_result = evaluate(scores_b, labels_b)

            # Metrics
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("EER", f"{eval_result.eer:.3f}",
                      help="Equal Error Rate: lower is better (0=perfect)")
            e2.metric("t-DCF", f"{eval_result.tdcf:.3f}",
                      help="ASVspoof tandem DCF: lower is better")
            e3.metric("AUC", f"{eval_result.auc:.3f}",
                      help="Area under ROC: 1.0=perfect, 0.5=random")
            e4.metric("Accuracy", f"{eval_result.accuracy:.1%}")

            f1, f2 = st.columns(2)
            f1.metric("FAR", f"{eval_result.far_at_threshold:.3f}",
                      help="False Accept Rate at EER threshold")
            f2.metric("FRR", f"{eval_result.frr_at_threshold:.3f}",
                      help="False Reject Rate at EER threshold")

            # ROC
            fpr_r, tpr_r = roc_curve_data(scores_b, labels_b)
            roc_fig = roc_figure(fpr_r, tpr_r, eval_result.auc)
            st.plotly_chart(roc_fig, use_container_width=True, config={"displayModeBar": False})

    with met_col2:
        st.markdown("#### SCORE DISTRIBUTION")

        if st.button("▶  SHOW DISTRIBUTIONS", use_container_width=True):
            rng = np.random.RandomState(42)
            human_s = rng.beta(2, 5, 200)
            synth_s = rng.beta(5, 2, 200)

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=human_s, nbinsx=30, name="Human",
                marker_color="rgba(0,255,65,0.6)",
                marker_line=dict(color="#020805", width=0.5),
            ))
            fig_dist.add_trace(go.Histogram(
                x=synth_s, nbinsx=30, name="Synthetic",
                marker_color="rgba(255,49,49,0.6)",
                marker_line=dict(color="#020805", width=0.5),
            ))
            fig_dist.update_layout(
                barmode="overlay",
                paper_bgcolor="#020805", plot_bgcolor="#020805",
                font=dict(family="Share Tech Mono", color="#8cc88c"),
                xaxis=dict(title="P(synthetic)", gridcolor="#1a3d1a"),
                yaxis=dict(title="Count", gridcolor="#1a3d1a"),
                legend=dict(font=dict(color="#8cc88c")),
                margin=dict(l=10, r=10, t=10, b=10),
                height=280,
                title=dict(text="SCORE DISTRIBUTIONS", font=dict(color="#00ff41", size=11, family="Orbitron")),
            )
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        st.markdown("#### METRIC DEFINITIONS")
        st.markdown("""
        <div class="terminal">
EER (Equal Error Rate)
  Threshold where FAR = FRR.
  Lower = better. 0% = perfect.

t-DCF (tandem Detection Cost Function)
  ASVspoof standard. Penalizes false
  accepts 10x more than false rejects.
  Normalized: 0=perfect, 1=baseline.

AUC (Area Under ROC Curve)
  Rank-based performance metric.
  1.0 = perfect, 0.5 = random.

FAR (False Accept Rate)
  Human speech classified as synthetic.
  Direct usability impact.

FRR (False Reject Rate)
  Synthetic classified as human.
  Security impact (missed detection).
        </div>
        """, unsafe_allow_html=True)

    # System info
    st.markdown("---")
    st.markdown("#### SYSTEM PERFORMANCE TARGETS")
    perf_data = {
        "Metric": ["EER", "t-DCF", "AUC", "Latency (10s chunk)", "FAR", "FRR"],
        "Target": ["< 10%", "< 0.5", "> 0.90", "< 500ms", "< 5%", "< 15%"],
        "Notes": [
            "ASVspoof 2019 LA baseline ~10%",
            "Industry standard threshold",
            "Production-grade threshold",
            "Consumer hardware target",
            "Acceptable false alarm rate",
            "Max allowed miss rate",
        ]
    }
    import pandas as pd
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)
