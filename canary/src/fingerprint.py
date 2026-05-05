"""
fingerprint.py — TTS system fingerprint detection.

Identifies which TTS system likely generated the audio based on
characteristic spectral artifacts of each system.

Known TTS artifact signatures (from research literature):
  - SpeechT5:    Characteristic spectral peaks 3-5kHz, smooth mel transitions
  - XTTS/XTTS-v2: High-frequency artifacts >6kHz, vocoder phase patterns
  - OpenVoice:   Spectral envelope over-smoothing in 1-3kHz
  - WhisperTTS:  Low-frequency emphasis, compressed dynamic range
  - Generic TTS: High spectral flatness, regular pitch, low noise floor

This module uses rule-based fingerprinting (production would use
a trained classifier on labeled TTS outputs).
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

from .features import AudioFeatures, extract_features


@dataclass
class FingerprintResult:
    likely_source: str              # e.g. "SpeechT5", "XTTS-v2", "Unknown TTS"
    confidence: float               # 0-1
    scores: Dict[str, float]        # per-model scores
    artifacts: List[str]            # detected artifact descriptions
    is_synthetic: bool


# ── Spectral fingerprint profiles ─────────────────────────────────────────────
# Each entry: {feature: (expected_value, tolerance, weight)}
TTS_PROFILES = {
    "SpeechT5": {
        "spectral_flatness_mean": (0.12, 0.05, 2.0),   # high flatness
        "f0_regularity":          (0.82, 0.10, 2.5),   # very regular pitch
        "mel_smoothness":         (2.8,  1.0,  2.0),   # over-smooth
        "hnr":                    (22.0, 6.0,  1.5),   # clean harmonics
        "dynamic_range":          (12.0, 4.0,  1.0),   # compressed
        "spectral_flux":          (0.002, 0.001, 1.5), # very low flux
        # SpeechT5 characteristic: peak energy 3-5kHz
        "_band_3_5khz_ratio":     (0.35, 0.10, 1.5),
    },
    "XTTS-v2": {
        "spectral_flatness_mean": (0.09, 0.04, 1.5),
        "f0_regularity":          (0.78, 0.12, 2.0),
        "mel_smoothness":         (3.2,  1.2,  1.5),
        "hnr":                    (19.0, 5.0,  1.5),
        "phase_randomness":       (0.70, 0.10, 2.0),   # low phase randomness
        # XTTS: high-freq artifacts >6kHz
        "_band_6plus_ratio":      (0.18, 0.06, 2.0),
    },
    "OpenVoice": {
        "spectral_flatness_mean": (0.08, 0.04, 1.5),
        "f0_regularity":          (0.75, 0.12, 2.0),
        "mel_smoothness":         (3.8,  1.5,  2.0),
        # OpenVoice: over-smooth 1-3kHz
        "_band_1_3khz_ratio":     (0.42, 0.08, 2.5),
        "dynamic_range":          (14.0, 5.0,  1.0),
    },
    "WhisperTTS": {
        "spectral_flatness_mean": (0.06, 0.03, 1.5),
        "f0_regularity":          (0.80, 0.10, 2.0),
        "mel_smoothness":         (2.5,  1.0,  2.5),
        "dynamic_range":          (10.0, 3.0,  2.0),   # very compressed
        # Low-frequency emphasis
        "_band_low_ratio":        (0.55, 0.10, 2.0),
        "silence_regularity":     (0.75, 0.10, 1.5),
    },
    "AudioLDM2": {
        "spectral_flatness_mean": (0.10, 0.05, 1.5),
        "spectral_flux":          (0.001, 0.0005, 2.0), # extremely low
        "mel_smoothness":         (2.0,  0.8,  2.5),
        "phase_randomness":       (0.60, 0.15, 2.5),   # structured phase
        "hnr":                    (25.0, 8.0,  1.5),   # very clean
    },
}


def _extract_band_ratios(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract energy ratios in specific frequency bands."""
    n_fft = 2048
    D = np.abs(librosa.stft(audio, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total_energy = np.mean(D) + 1e-10

    def band_ratio(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.mean(D[mask]) / total_energy)

    return {
        "_band_low_ratio":    band_ratio(0, 1000),
        "_band_1_3khz_ratio": band_ratio(1000, 3000),
        "_band_3_5khz_ratio": band_ratio(3000, 5000),
        "_band_6plus_ratio":  band_ratio(6000, sr / 2),
    }


def _profile_score(feat_dict: Dict[str, float], profile: Dict) -> float:
    """
    Compute similarity between extracted features and a TTS profile.
    Returns score in [0, 1] — higher = more similar to this TTS.
    """
    total_weight = 0.0
    weighted_match = 0.0

    for feat_name, (expected, tolerance, weight) in profile.items():
        if feat_name not in feat_dict:
            continue

        value = feat_dict[feat_name]
        # Gaussian similarity: how close is value to expected?
        similarity = np.exp(-0.5 * ((value - expected) / (tolerance + 1e-8)) ** 2)
        weighted_match += similarity * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return float(weighted_match / total_weight)


def fingerprint_audio(audio: np.ndarray, sr: int) -> FingerprintResult:
    """
    Identify likely TTS source system.
    
    Args:
        audio: mono float32 waveform
        sr: sample rate
        
    Returns:
        FingerprintResult with likely source and confidence
    """
    # Extract features
    feats = extract_features(audio, sr, store_arrays=False)
    band_ratios = _extract_band_ratios(audio, sr)

    # Build flat feature dict
    feat_dict: Dict[str, float] = {
        "spectral_flatness_mean": feats.spectral_flatness_mean,
        "f0_regularity":          feats.f0_regularity,
        "mel_smoothness":         feats.mel_smoothness,
        "hnr":                    feats.hnr,
        "dynamic_range":          feats.dynamic_range,
        "spectral_flux":          feats.spectral_flux,
        "phase_randomness":       feats.phase_randomness,
        "silence_regularity":     feats.silence_regularity,
        **band_ratios,
    }

    # Score each profile
    scores = {}
    for model_name, profile in TTS_PROFILES.items():
        scores[model_name] = _profile_score(feat_dict, profile)

    # Is it even synthetic?
    # Natural speech has spectral flatness < 0.05, f0_regularity < 0.7
    # Core TTS signal: EITHER very regular pitch OR very high HNR.
    # White noise and music have neither — they fail this core gate.
    has_reg_pitch  = feats.f0_regularity   > 0.70
    has_clean_hnr  = feats.hnr             > 18.0
    has_flat_spec  = feats.spectral_flatness_mean > 0.06
    has_smooth_mel = feats.mel_smoothness  < 3.5
    has_low_dyn    = feats.dynamic_range   < 15.0
    # Must pass the core gate (pitch OR HNR) PLUS ≥1 additional signal
    core_tts    = has_reg_pitch or has_clean_hnr
    extra_signals = sum([has_flat_spec, has_smooth_mel, has_low_dyn])
    is_synthetic = core_tts and extra_signals >= 1

    if not is_synthetic:
        return FingerprintResult(
            likely_source="Human Speech",
            confidence=0.8,
            scores=scores,
            artifacts=[],
            is_synthetic=False,
        )

    # Best match
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]

    # Normalize confidence
    score_vals = list(scores.values())
    if len(score_vals) > 1:
        score_arr = np.array(score_vals)
        # Confidence = how dominant best score is
        second_best = sorted(score_vals)[-2]
        margin = best_score - second_best
        confidence = float(np.clip(0.5 + margin * 2, 0.3, 0.95))
    else:
        confidence = best_score

    # Low confidence → "Unknown TTS"
    if best_score < 0.3:
        best_model = "Unknown TTS"
        confidence = 0.4

    # Artifact descriptions
    artifacts = _describe_artifacts(feats, feat_dict, best_model)

    return FingerprintResult(
        likely_source=best_model,
        confidence=confidence,
        scores=scores,
        artifacts=artifacts,
        is_synthetic=True,
    )


def _describe_artifacts(feats: AudioFeatures, feat_dict: Dict, model: str) -> List[str]:
    """Generate artifact descriptions for detected TTS."""
    artifacts = []

    if feats.spectral_flatness_mean > 0.07:
        artifacts.append(
            f"Spectral flatness {feats.spectral_flatness_mean:.4f} — frequency bands over-smoothed"
        )
    if feats.f0_regularity > 0.72:
        artifacts.append(
            f"Pitch regularity {feats.f0_regularity:.2f} — unnaturally stable fundamental frequency"
        )
    if feats.hnr > 18:
        artifacts.append(
            f"HNR {feats.hnr:.1f} dB — harmonics too clean, insufficient noise floor"
        )
    if feats.phase_randomness < 0.80:
        artifacts.append(
            f"Phase randomness {feats.phase_randomness:.3f} — vocoder phase structure detected"
        )
    if feats.mel_smoothness < 3.5:
        artifacts.append(
            f"Mel smoothness {feats.mel_smoothness:.2f} dB — lacking micro-variation of natural speech"
        )
    if feats.dynamic_range < 15.0:
        artifacts.append(
            f"Dynamic range {feats.dynamic_range:.1f} dB — compressed loudness variation"
        )

    # Model-specific artifacts
    if model == "XTTS-v2" and feat_dict.get("_band_6plus_ratio", 0) > 0.12:
        artifacts.append("High-frequency content >6kHz — characteristic XTTS vocoder artifact")
    if model == "SpeechT5" and feat_dict.get("_band_3_5khz_ratio", 0) > 0.25:
        artifacts.append("Spectral peak 3-5kHz — characteristic SpeechT5 formant pattern")
    if model == "OpenVoice" and feat_dict.get("_band_1_3khz_ratio", 0) > 0.34:
        artifacts.append("Mid-band smoothing 1-3kHz — characteristic OpenVoice artifact")

    return artifacts
