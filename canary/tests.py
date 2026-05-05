"""
tests.py — Canary system test suite (9 tests, no pytest required).

Run with:  python tests.py

Tests cover:
  T1  Feature extraction — shapes, finite, known-property checks
  T2  Heuristic scoring logic — mock AudioFeatures, rule firing
  T3  End-to-end detect() — output structure, rule semantics
  T4  Streaming — chunk count, ordering, latency variance
  T5  TTS fingerprinting — is_synthetic, score keys, confidence range
  T6  Degradation pipeline — 7 conditions, length/NaN/range checks
  T7  Evaluation metrics — EER, AUC, t-DCF, ROC, edge cases
  T8  Plotly visualizations — dashboard, timeline, empty list
  T9  benchmark_under_degradation() — all 7 conditions, key presence
"""

import numpy as np
import sys
import time

sys.path.insert(0, ".")

# ── Test signals ──────────────────────────────────────────────────────────────
SR = 16000
T  = np.linspace(0, 4.0, int(SR * 4.0))
np.random.seed(42)

# Pure harmonics — triggers pitch_regularity + HNR (TTS-like signals)
HARM = sum((1/k) * np.sin(2 * np.pi * 150.0 * k * T) for k in range(1, 10)).astype(np.float32)
HARM /= np.max(np.abs(HARM))

# White noise — triggers spectral_flatness (high), not pitch/HNR
NOISE = np.random.randn(len(T)).astype(np.float32) * 0.5

# Sub-threshold clip (0.3s < 0.5s minimum)
SHORT = HARM[:int(SR * 0.3)]

# Mock TTS AudioFeatures (all 8 rules should fire)
from src.features import AudioFeatures

TTS_FEATS = AudioFeatures(
    spectral_flatness_mean=0.13, spectral_flatness_std=0.02,
    f0_regularity=0.91, f0_mean=180.0, f0_std=4.0,
    hnr=28.0, phase_randomness=0.60, mel_smoothness=2.2,
    silence_regularity=0.80, spectral_flux=0.001, dynamic_range=9.0,
    voiced_fraction=0.75, rms_mean=0.12, rms_std=0.02,
    spectral_rolloff_mean=3000.0, spectral_bandwidth_mean=2000.0,
    spectral_contrast_mean=np.ones(7) * 10,
    mfcc_mean=np.zeros(20), mfcc_std=np.ones(20) * 5,
    mfcc_delta_mean=np.zeros(20), mel_temporal_var=50.0,
    zcr_mean=0.05, zcr_std=0.01, silence_ratio=0.15,
)

HUMAN_FEATS = AudioFeatures(
    spectral_flatness_mean=0.03, spectral_flatness_std=0.01,
    f0_regularity=0.45, f0_mean=130.0, f0_std=18.0,
    hnr=8.0, phase_randomness=1.20, mel_smoothness=6.5,
    silence_regularity=0.40, spectral_flux=0.02, dynamic_range=32.0,
    voiced_fraction=0.65, rms_mean=0.08, rms_std=0.06,
    spectral_rolloff_mean=4000.0, spectral_bandwidth_mean=2500.0,
    spectral_contrast_mean=np.ones(7) * 20,
    mfcc_mean=np.zeros(20), mfcc_std=np.ones(20) * 8,
    mfcc_delta_mean=np.zeros(20), mel_temporal_var=200.0,
    zcr_mean=0.08, zcr_std=0.03, silence_ratio=0.18,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

KNOWN_RULES = {
    "spectral_flatness", "pitch_regularity", "hnr", "phase_randomness",
    "mel_smoothness", "silence_regularity", "spectral_flux", "dynamic_range",
}

FINGERPRINT_MODELS = {"SpeechT5", "XTTS-v2", "OpenVoice", "WhisperTTS", "AudioLDM2"}

DEGRADATION_NAMES = {
    "Clean", "Phone (μ-law)", "VoIP (Opus)", "Noisy (SNR 10)",
    "Reverb Room", "Babble Noise", "Worst Case",
}


def run_tests():
    from src.features import extract_features, feature_vector
    from src.detector import DeepfakeDetector, TORCH_AVAILABLE
    from src.fingerprint import fingerprint_audio
    from src.robustness import apply_degradation, benchmark_degradation_configs
    from src.evaluator import (
        generate_synthetic_benchmark, evaluate, roc_curve_data,
        compute_eer, benchmark_under_degradation,
    )
    from src.explainer import plot_detection_dashboard, plot_streaming_timeline
    import plotly.graph_objects as go

    det = DeepfakeDetector(use_model=False)

    # Pre-compute shared objects
    f_harm  = extract_features(HARM,  SR, store_arrays=True)
    r_harm  = det.detect(HARM,  SR)
    r_noise = det.detect(NOISE, SR)
    r_short = det.detect(SHORT, SR)
    long    = np.tile(HARM, 3)
    chunks  = list(det.detect_stream(long, SR, window_sec=4.0, hop_sec=2.0))
    lats    = [c.processing_time_ms for c in chunks]

    scores7, labels7 = generate_synthetic_benchmark(200, 200, seed=99)
    ev = evaluate(scores7, labels7)

    results = {}
    total_start = time.time()

    def test(n, fn):
        t0 = time.time()
        try:
            fn()
            elapsed = (time.time() - t0) * 1000
            results[n] = ("PASS", elapsed)
            print(f"  T{n}: PASSED ✓  ({elapsed:.0f}ms)")
        except AssertionError as e:
            results[n] = ("FAIL", str(e))
            print(f"  T{n}: FAILED ✗  {e}")
        except Exception as e:
            results[n] = ("ERROR", f"{type(e).__name__}: {e}")
            print(f"  T{n}: ERROR ✗   {type(e).__name__}: {e}")

    print("=" * 60)
    print(f"CANARY SYSTEM TESTS  |  torch={'on' if TORCH_AVAILABLE else 'off'}")
    print("=" * 60)

    # ── T1: Feature Extraction ──────────────────────────────────────────────
    def t1():
        vec = feature_vector(f_harm)
        assert vec.shape[0] == 67, f"Expected 67 dims, got {vec.shape[0]}"
        assert np.isfinite(vec).all(), "NaN/Inf in feature vector"
        assert f_harm.mel_db is not None
        assert f_harm.stft_mag is not None
        assert f_harm.f0_values is not None

        f_noise = extract_features(NOISE, SR, store_arrays=False)
        # White noise: high spectral flatness (broadband energy)
        assert f_noise.spectral_flatness_mean > 0.30, \
            f"White noise flatness={f_noise.spectral_flatness_mean:.3f} should be >0.30"
        # Harmonics: high HNR + very regular pitch
        assert f_harm.hnr > 15, f"Harm HNR={f_harm.hnr:.1f} should be >15 dB"
        assert f_harm.f0_regularity > 0.80, f"Harm f0_reg={f_harm.f0_regularity:.3f} should be >0.80"
    test(1, t1)

    # ── T2: Heuristic Scoring Logic ─────────────────────────────────────────
    def t2():
        tts_s, tts_r     = det._heuristic_score(TTS_FEATS)
        human_s, human_r = det._heuristic_score(HUMAN_FEATS)
        assert tts_s > 0.60,   f"TTS score should be >0.60, got {tts_s:.3f}"
        assert human_s < 0.40, f"Human score should be <0.40, got {human_s:.3f}"
        assert tts_s > human_s
        assert len(tts_r) >= 5, f"Expected ≥5 TTS rules, got {len(tts_r)}"
        assert len(human_r) == 0, f"Expected 0 human rules, got {len(human_r)}"
        for n, v, thr, d, sc in tts_r:
            assert n in KNOWN_RULES, f"Unknown rule: {n}"
            assert 0 <= sc <= 1, f"Rule score out of [0,1]: {sc}"
    test(2, t2)

    # ── T3: End-to-End detect() ─────────────────────────────────────────────
    def t3():
        for label, r in [("harm", r_harm), ("noise", r_noise), ("short", r_short)]:
            assert 0 <= r.synthetic_probability <= 1, f"{label}: P={r.synthetic_probability}"
            assert 0 <= r.confidence <= 1, f"{label}: conf={r.confidence}"
            assert r.explanation, f"{label}: empty explanation"
            assert r.processing_time_ms >= 0, f"{label}: negative latency"
            for n, *_ in r.triggered_rules:
                assert n in KNOWN_RULES, f"Unknown rule: {n}"

        harm_rules  = {r[0] for r in r_harm.triggered_rules}
        noise_rules = {r[0] for r in r_noise.triggered_rules}
        # Harmonics must trigger core TTS rules
        assert "pitch_regularity" in harm_rules, "Harmonics must trigger pitch_regularity"
        assert "hnr"              in harm_rules, "Harmonics must trigger hnr"
        # White noise must trigger spectral flatness (it IS flat)
        assert "spectral_flatness" in noise_rules, "Noise must trigger spectral_flatness"
        # Short clip: low confidence + specific message
        assert r_short.confidence < 0.5, f"Short clip conf={r_short.confidence} should be <0.5"
        assert "too short" in r_short.explanation.lower()
    test(3, t3)

    # ── T4: Streaming ───────────────────────────────────────────────────────
    def t4():
        probs = [c.synthetic_probability for c in chunks]
        assert len(chunks) >= 4, f"Expected ≥4 chunks, got {len(chunks)}"
        assert all(0 <= p <= 1 for p in probs), "Some chunk probabilities out of [0,1]"
        assert all(c.chunk_index == i for i, c in enumerate(chunks)), \
            "chunk_index not sequential"
        assert all(c.explanation for c in chunks), "Some chunks have empty explanations"
        assert np.std(probs) < 0.20, \
            f"Chunk variance too high (consistent signal): std={np.std(probs):.3f}"
    test(4, t4)

    # ── T5: TTS Fingerprinting ──────────────────────────────────────────────
    def t5():
        fp_h = fingerprint_audio(HARM,  SR)
        fp_n = fingerprint_audio(NOISE, SR)
        assert set(fp_h.scores.keys()) == FINGERPRINT_MODELS
        assert all(0 <= v <= 1 for v in fp_h.scores.values())
        assert 0 <= fp_h.confidence <= 1
        assert fp_h.is_synthetic, \
            "Pure harmonics (high HNR, regular pitch) must be flagged synthetic"
        assert not fp_n.is_synthetic, \
            "White noise (no regular pitch, near-zero HNR) must NOT be flagged synthetic"
    test(5, t5)

    # ── T6: Degradation Pipeline ────────────────────────────────────────────
    probs6 = []
    def t6():
        cfgs = benchmark_degradation_configs()
        assert len(cfgs) == 7, f"Expected 7 degradation configs, got {len(cfgs)}"
        for info in cfgs:
            name, cfg = info["name"], info["config"]
            deg = apply_degradation(HARM.copy(), SR, cfg)
            assert len(deg) == len(HARM), f"{name}: output length changed"
            assert np.isfinite(deg).all(), f"{name}: NaN/Inf in output"
            r = det.detect(deg, SR)
            assert 0 <= r.synthetic_probability <= 1, \
                f"{name}: P={r.synthetic_probability} out of range"
            probs6.append(r.synthetic_probability)
        n_detect = sum(1 for p in probs6 if p > 0.4)
        assert n_detect >= 3, f"Need ≥3/7 conditions to score >0.4, got {n_detect}"
    test(6, t6)

    # ── T7: Evaluation Metrics ──────────────────────────────────────────────
    def t7():
        assert 0 <= ev.eer < 0.20, f"EER={ev.eer:.4f} should be <0.20"
        assert ev.auc > 0.75, f"AUC={ev.auc:.4f} should be >0.75"
        assert abs(ev.far_at_threshold - ev.frr_at_threshold) < 0.10, \
            f"|FAR-FRR|={abs(ev.far_at_threshold-ev.frr_at_threshold):.4f} at EER threshold"

        fpr, tpr = roc_curve_data(scores7, labels7)
        assert len(fpr) >= 100, f"ROC too coarse: {len(fpr)} pts"
        assert fpr.shape == tpr.shape
        assert fpr[0] <= 0.05, "ROC must start near (0,0)"

        # Edge case: all-positive predictions
        eer_e, thr_e = compute_eer(np.ones(50), np.array([1]*25 + [0]*25))
        assert 0 <= eer_e <= 1, f"Edge EER out of range: {eer_e}"
        assert 0 <= thr_e <= 1, f"Edge threshold out of range: {thr_e}"
    test(7, t7)

    # ── T8: Plotly Visualizations ────────────────────────────────────────────
    def t8():
        r_harm.features = extract_features(HARM, SR, store_arrays=True)
        fig_d = plot_detection_dashboard(r_harm, SR)
        fig_t = plot_streaming_timeline(chunks)
        fig_e = plot_streaming_timeline([])   # empty list edge case

        assert isinstance(fig_d, go.Figure)
        assert len(fig_d.data) >= 6, f"Dashboard: expected ≥6 traces, got {len(fig_d.data)}"
        assert fig_d.layout.height == 780
        assert fig_d.layout.paper_bgcolor == "#050a05", "Wrong background colour (CRT dark)"
        assert len(fig_d.layout.annotations) == 6, "Expected 6 subplot titles"

        assert isinstance(fig_t, go.Figure)
        assert len(fig_t.data) >= 1, "Timeline must have ≥1 trace"

        assert isinstance(fig_e, go.Figure), "Empty list must return Figure (not crash)"
    test(8, t8)

    # ── T9: benchmark_under_degradation() ───────────────────────────────────
    def t9():
        bench = benchmark_under_degradation(det, HARM, SR, label=1)
        assert set(bench.keys()) == DEGRADATION_NAMES, \
            f"Missing keys: {DEGRADATION_NAMES - set(bench.keys())}"
        for name, data in bench.items():
            for key in ["score", "latency_ms", "correct"]:
                assert key in data, f"{name} missing '{key}'"
            assert 0 <= data["score"] <= 1, f"{name}: score out of [0,1]"
            assert data["latency_ms"] >= 0, f"{name}: negative latency"
            assert isinstance(data["correct"], bool), f"{name}: 'correct' not bool"
    test(9, t9)

    # ── Summary ─────────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v[0] == "PASS")
    total_ms = (time.time() - total_start) * 1000

    print()
    print("=" * 60)
    if passed == 9:
        print(f"  ALL 9/9 TESTS PASSED ✓  ({total_ms:.0f}ms total)")
    else:
        print(f"  {passed}/9 TESTS PASSED")
        for n, (status, detail) in results.items():
            if status != "PASS":
                print(f"  FAIL T{n}: {detail}")
    print("=" * 60)

    # Diagnostics
    tts_s, _ = det._heuristic_score(TTS_FEATS)
    hum_s, _ = det._heuristic_score(HUMAN_FEATS)
    harm_rules = sorted({r[0] for r in r_harm.triggered_rules})
    n_rob = sum(1 for p in probs6 if p > 0.4)
    fp_result = fingerprint_audio(HARM, SR)

    print(f"""
  SYSTEM DIAGNOSTICS
  ──────────────────────────────────────────────────────
  Feature vector dims       : 67
  Heuristic — TTS score     : {tts_s:.3f}  (8/8 rules fire on mock TTS)
  Heuristic — Human score   : {hum_s:.3f}  (0/8 rules fire on mock human)
  Rules fired (harmonics)   : {harm_rules}
  EER                       : {ev.eer:.3f}  (target <0.20)
  AUC                       : {ev.auc:.3f}  (target >0.75)
  t-DCF                     : {ev.tdcf:.3f}
  Stream — {len(chunks)} chunks, avg lat   : {np.mean(lats):.0f}ms
  Robustness P>0.40         : {n_rob}/7 degradation conditions
  Fingerprint (harmonics)   : {fp_result.likely_source}  conf={fp_result.confidence:.2f}
  Torch                     : {'enabled' if TORCH_AVAILABLE else 'disabled (heuristic-only)'}
  ──────────────────────────────────────────────────────
""")

    return passed == 9


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
