"""
explainer.py — Visualization and explainability for detections.

Generates:
  - Mel spectrogram with anomaly overlay heatmap
  - Pitch track visualization
  - Spectral feature bar charts
  - Anomaly region highlighting
"""

import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
import matplotlib.cm as cm

from .features import AudioFeatures
from .detector import DetectionResult


# ── Color palette (phosphor green on dark) ────────────────────────────────────
BG_COLOR = "#050a05"
GRID_COLOR = "#1a2e1a"
GREEN = "#00ff41"
CYAN = "#00d4ff"
AMBER = "#ffb700"
RED_ALERT = "#ff3131"
DIM_GREEN = "#1a4d1a"
TEXT_COLOR = "#a8c8a8"


def plot_detection_dashboard(result: DetectionResult, sr: int) -> go.Figure:
    """
    Full forensics dashboard: mel + pitch + features + alert panel.
    Returns a Plotly figure.
    """
    feats = result.features

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "MEL SPECTROGRAM + ANOMALY OVERLAY",
            "SPECTRAL FEATURES",
            "PITCH TRACK (F0)",
            "FEATURE ANOMALY SCORES",
            "SPECTROGRAM PHASE",
            "DETECTION TIMELINE",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        row_heights=[0.4, 0.35, 0.25],
    )

    _add_mel_spectrogram(fig, feats, result, row=1, col=1)
    _add_spectral_features(fig, feats, row=1, col=2)
    _add_pitch_track(fig, feats, sr, row=2, col=1)
    _add_anomaly_bars(fig, result, row=2, col=2)
    _add_phase_plot(fig, feats, row=3, col=1)
    _add_timeline(fig, result, row=3, col=2)

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Courier New, monospace", size=10, color=TEXT_COLOR),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=780,
    )

    # Style all axes
    axis_style = dict(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        color=TEXT_COLOR,
        tickfont=dict(size=9, color=TEXT_COLOR),
    )
    for key in fig.layout:
        if key.startswith("xaxis") or key.startswith("yaxis"):
            fig.layout[key].update(axis_style)

    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.font.color = GREEN
        ann.font.size = 10
        ann.font.family = "Courier New, monospace"

    return fig


def _add_mel_spectrogram(fig, feats: AudioFeatures, result: DetectionResult, row, col):
    """Mel spectrogram with anomaly heatmap overlay."""
    if feats is None or feats.mel_db is None:
        return

    mel = feats.mel_db
    times = np.linspace(0, feats.times[-1] if feats.times is not None else mel.shape[1] * 0.032, mel.shape[1])
    freqs = np.linspace(0, 8000, mel.shape[0])

    fig.add_trace(
        go.Heatmap(
            z=mel,
            x=times,
            y=freqs,
            colorscale=[
                [0.0,  "#000000"],
                [0.25, "#001a00"],
                [0.5,  "#004d00"],
                [0.75, "#00a000"],
                [1.0,  "#00ff41"],
            ],
            showscale=False,
            zmin=-80, zmax=0,
            name="Mel",
        ),
        row=row, col=col,
    )

    # Anomaly overlay: highlight suspicious frequency bands
    if result.synthetic_probability > 0.5 and feats.spectral_contrast_mean is not None:
        # Highlight mid-frequency region (1-4kHz) if spectral flatness high
        if feats.spectral_flatness_mean > 0.07:
            fig.add_shape(
                type="rect",
                x0=times[0], x1=times[-1],
                y0=1000, y1=4000,
                fillcolor=f"rgba(255,49,49,0.12)",
                line=dict(color=RED_ALERT, width=1, dash="dot"),
                row=row, col=col,
            )

        # Phase anomaly band
        if feats.phase_randomness < 0.85:
            fig.add_shape(
                type="rect",
                x0=times[0], x1=times[-1],
                y0=4000, y1=8000,
                fillcolor="rgba(0,212,255,0.08)",
                line=dict(color=CYAN, width=1, dash="dot"),
                row=row, col=col,
            )

    fig.update_xaxes(title_text="Time (s)", row=row, col=col)
    fig.update_yaxes(title_text="Freq (Hz)", row=row, col=col)


def _add_spectral_features(fig, feats: AudioFeatures, row, col):
    """Bar chart of key spectral features with threshold markers."""
    if feats is None:
        return

    names = [
        "Spec. Flatness",
        "Pitch Regularity",
        "HNR (norm)",
        "Phase Random.",
        "Mel Smoothness",
        "Dynamic Range",
        "Silence Reg.",
        "Spec. Flux",
    ]
    values = [
        min(feats.spectral_flatness_mean * 10, 1.0),
        feats.f0_regularity,
        min(max(feats.hnr / 40.0, 0), 1),
        feats.phase_randomness / np.pi,
        min(feats.mel_smoothness / 10.0, 1.0),
        min(feats.dynamic_range / 60.0, 1.0),
        feats.silence_regularity,
        min(feats.spectral_flux * 100, 1.0),
    ]
    # Suspicious direction: high for flatness, pitch_reg, HNR, silence; low for phase, mel, dynamic, flux
    suspicious_if_high = [True, True, True, False, False, False, True, False]
    thresholds_norm = [0.7, 0.72, 0.45, 0.27, 0.35, 0.25, 0.68, 0.5]

    colors = []
    for v, sih, thr in zip(values, suspicious_if_high, thresholds_norm):
        if (sih and v > thr) or (not sih and v < thr):
            colors.append(RED_ALERT)
        else:
            colors.append(GREEN)

    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            marker_line=dict(color=DIM_GREEN, width=1),
            opacity=0.85,
        ),
        row=row, col=col,
    )

    # Threshold line
    fig.add_shape(
        type="line",
        x0=0.5, x1=0.5, y0=-0.5, y1=len(names) - 0.5,
        line=dict(color=AMBER, width=1, dash="dash"),
        row=row, col=col,
    )

    fig.update_xaxes(range=[0, 1], row=row, col=col)


def _add_pitch_track(fig, feats: AudioFeatures, sr: int, row, col):
    """F0 pitch track with stability visualization."""
    if feats is None or feats.f0_values is None or feats.f0_times is None:
        return

    f0 = feats.f0_values
    t = feats.f0_times
    valid = ~np.isnan(f0)

    if np.any(valid):
        fig.add_trace(
            go.Scatter(
                x=t[valid],
                y=f0[valid],
                mode="lines",
                line=dict(color=GREEN, width=1.5),
                name="F0",
                fill="tozeroy",
                fillcolor="rgba(0,255,65,0.08)",
            ),
            row=row, col=col,
        )

        # Moving average (shows regularity)
        if np.sum(valid) > 10:
            from scipy.ndimage import uniform_filter1d
            f0_filled = np.where(valid, f0, np.nan)
            ma_vals = f0_filled[valid]
            ma_times = t[valid]
            # Smooth version
            if len(ma_vals) > 5:
                window = min(15, len(ma_vals) // 3)
                smooth = np.convolve(ma_vals, np.ones(window) / window, mode="same")
                fig.add_trace(
                    go.Scatter(
                        x=ma_times,
                        y=smooth,
                        mode="lines",
                        line=dict(color=AMBER, width=1, dash="dot"),
                        name="F0 smooth",
                    ),
                    row=row, col=col,
                )

    fig.update_xaxes(title_text="Time (s)", row=row, col=col)
    fig.update_yaxes(title_text="Hz", row=row, col=col)


def _add_anomaly_bars(fig, result: DetectionResult, row, col):
    """Radar-style anomaly score breakdown."""
    categories = ["Spectral\nFlatness", "Pitch\nReg.", "HNR", "Phase", "Mel\nSmooth", "Silence\nReg."]

    feats = result.features
    if feats is None:
        return

    vals = [
        min(feats.spectral_flatness_mean * 10, 1.0),
        feats.f0_regularity,
        min(max(feats.hnr / 40.0, 0), 1),
        1.0 - (feats.phase_randomness / np.pi),  # invert: low randomness = high anomaly
        1.0 - min(feats.mel_smoothness / 10.0, 1.0),  # invert: low smoothness = high anomaly
        feats.silence_regularity,
    ]

    colors = [RED_ALERT if v > 0.5 else GREEN for v in vals]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=vals,
            marker_color=colors,
            marker_line=dict(color=BG_COLOR, width=1),
            opacity=0.9,
        ),
        row=row, col=col,
    )

    # Threshold line at 0.5
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(categories) - 0.5,
        y0=0.5, y1=0.5,
        line=dict(color=AMBER, width=1, dash="dash"),
        row=row, col=col,
    )

    fig.update_yaxes(range=[0, 1], title_text="Anomaly Score", row=row, col=col)


def _add_phase_plot(fig, feats: AudioFeatures, row, col):
    """Phase difference visualization — structured = vocoder artifact."""
    if feats is None or feats.stft_phase is None:
        return

    phase = feats.stft_phase
    # Phase difference across frequency bins (should be random for natural speech)
    phase_diff = np.diff(phase, axis=0)
    # Mean absolute phase diff per time frame
    mean_pd = np.mean(np.abs(phase_diff), axis=0)
    t = feats.times if feats.times is not None else np.arange(len(mean_pd))
    t = t[:len(mean_pd)]

    color = RED_ALERT if feats.phase_randomness < 0.85 else GREEN

    fig.add_trace(
        go.Scatter(
            x=t,
            y=mean_pd,
            mode="lines",
            line=dict(color=color, width=1),
            fill="tozeroy",
            fillcolor=f"rgba({'255,49,49' if color == RED_ALERT else '0,255,65'},0.1)",
        ),
        row=row, col=col,
    )

    fig.update_xaxes(title_text="Time (s)", row=row, col=col)
    fig.update_yaxes(title_text="Phase diff", row=row, col=col)


def _add_timeline(fig, result: DetectionResult, row, col):
    """Single-chunk probability gauge as horizontal bar."""
    prob = result.synthetic_probability
    color = RED_ALERT if prob > 0.65 else (AMBER if prob > 0.4 else GREEN)

    fig.add_trace(
        go.Bar(
            x=[prob],
            y=["SYNTHETIC PROB"],
            orientation="h",
            marker_color=color,
            opacity=0.9,
            text=[f"{prob:.1%}"],
            textposition="outside",
            textfont=dict(color=color, size=14, family="Courier New"),
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Bar(
            x=[1 - prob],
            y=["SYNTHETIC PROB"],
            orientation="h",
            marker_color=DIM_GREEN,
            opacity=0.3,
        ),
        row=row, col=col,
    )

    fig.update_xaxes(range=[0, 1], showticklabels=False, row=row, col=col)
    fig.update_yaxes(showticklabels=True, row=row, col=col)
    fig.update_layout(barmode="stack")


def plot_streaming_timeline(results: List[DetectionResult]) -> go.Figure:
    """
    Multi-chunk streaming view — shows probability over time.
    """
    if not results:
        return go.Figure()

    times = [r.chunk_index for r in results]
    probs = [r.synthetic_probability for r in results]
    confs = [r.confidence for r in results]

    fig = go.Figure()

    # Fill regions
    fig.add_trace(go.Scatter(
        x=times, y=probs,
        mode="lines+markers",
        line=dict(color=GREEN, width=2),
        marker=dict(
            size=8,
            color=[RED_ALERT if p > 0.65 else (AMBER if p > 0.4 else GREEN) for p in probs],
            line=dict(color=BG_COLOR, width=1),
        ),
        fill="tozeroy",
        fillcolor="rgba(0,255,65,0.06)",
        name="Synthetic Prob",
    ))

    # Alert threshold
    fig.add_shape(
        type="line",
        x0=min(times), x1=max(times),
        y0=0.65, y1=0.65,
        line=dict(color=RED_ALERT, width=1, dash="dash"),
    )

    fig.add_annotation(
        x=max(times), y=0.65,
        text="ALERT THRESHOLD",
        showarrow=False,
        font=dict(color=RED_ALERT, size=9, family="Courier New"),
        xanchor="right",
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Courier New, monospace", color=TEXT_COLOR),
        xaxis=dict(title="Chunk #", gridcolor=GRID_COLOR, color=TEXT_COLOR),
        yaxis=dict(title="P(synthetic)", range=[0, 1], gridcolor=GRID_COLOR, color=TEXT_COLOR),
        margin=dict(l=10, r=10, t=30, b=10),
        height=200,
    )
    return fig
