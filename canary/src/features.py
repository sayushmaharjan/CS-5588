"""
features.py — Audio feature extraction for deepfake detection.

Extracts spectral, prosodic, and temporal features known to differ
between natural human speech and AI-generated (TTS) audio.

Key signals targeted:
  - Spectral flatness: TTS over-smooths frequency bands
  - Pitch regularity: TTS has unnaturally stable f0
  - Harmonic-to-Noise Ratio: TTS too clean
  - Mel smoothness: TTS lacks natural micro-variation
  - Phase randomness: vocoders introduce phase artifacts
  - Spectral flux: TTS transitions too smooth
"""

import numpy as np
import librosa
import scipy.signal as signal
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class AudioFeatures:
    # Spectral
    spectral_flatness_mean: float = 0.0
    spectral_flatness_std: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_contrast_mean: np.ndarray = field(default_factory=lambda: np.zeros(7))
    spectral_flux: float = 0.0

    # Mel / MFCC
    mfcc_mean: np.ndarray = field(default_factory=lambda: np.zeros(20))
    mfcc_std: np.ndarray = field(default_factory=lambda: np.zeros(20))
    mfcc_delta_mean: np.ndarray = field(default_factory=lambda: np.zeros(20))
    mel_smoothness: float = 0.0          # low variance across time = over-smooth
    mel_temporal_var: float = 0.0

    # Pitch / prosody
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_regularity: float = 0.0           # 1.0 = perfectly regular = suspicious
    voiced_fraction: float = 0.0

    # Energy / dynamics
    rms_mean: float = 0.0
    rms_std: float = 0.0
    dynamic_range: float = 0.0

    # Phase / harmonic
    hnr: float = 0.0                     # harmonic-to-noise ratio
    phase_randomness: float = 0.0        # low = vocoder artifact

    # Zero-crossing
    zcr_mean: float = 0.0
    zcr_std: float = 0.0

    # Temporal
    silence_ratio: float = 0.0
    silence_regularity: float = 0.0      # too-regular silence = TTS breath pattern

    # Raw arrays for visualization
    mel_db: Optional[np.ndarray] = None
    stft_mag: Optional[np.ndarray] = None
    stft_phase: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    freqs: Optional[np.ndarray] = None
    f0_times: Optional[np.ndarray] = None
    f0_values: Optional[np.ndarray] = None


def extract_features(audio: np.ndarray, sr: int, store_arrays: bool = True) -> AudioFeatures:
    """Full feature extraction pipeline."""
    feats = AudioFeatures()
    audio = audio.astype(np.float32)

    # ── STFT ──────────────────────────────────────────────────────────────
    n_fft = 2048
    hop = 512
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag = np.abs(D)
    phase = np.angle(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop)

    if store_arrays:
        feats.stft_mag = mag
        feats.stft_phase = phase
        feats.freqs = freqs
        feats.times = times

    # ── Spectral flatness ─────────────────────────────────────────────────
    sf = librosa.feature.spectral_flatness(S=mag)
    feats.spectral_flatness_mean = float(np.mean(sf))
    feats.spectral_flatness_std = float(np.std(sf))

    # ── Spectral rolloff ──────────────────────────────────────────────────
    sr_feat = librosa.feature.spectral_rolloff(S=mag, sr=sr)
    feats.spectral_rolloff_mean = float(np.mean(sr_feat))

    # ── Spectral bandwidth ────────────────────────────────────────────────
    bw = librosa.feature.spectral_bandwidth(S=mag, sr=sr)
    feats.spectral_bandwidth_mean = float(np.mean(bw))

    # ── Spectral contrast ─────────────────────────────────────────────────
    try:
        sc = librosa.feature.spectral_contrast(S=mag, sr=sr)
        feats.spectral_contrast_mean = np.mean(sc, axis=1)
    except Exception:
        feats.spectral_contrast_mean = np.zeros(7)

    # ── Spectral flux (change between frames) ─────────────────────────────
    flux = np.mean(np.diff(mag, axis=1) ** 2)
    feats.spectral_flux = float(flux)

    # ── Mel spectrogram ───────────────────────────────────────────────────
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if store_arrays:
        feats.mel_db = mel_db

    # Smoothness: mean std across time per band — low = over-smooth (TTS)
    per_band_std = np.std(mel_db, axis=1)
    feats.mel_smoothness = float(np.mean(per_band_std))
    feats.mel_temporal_var = float(np.var(np.mean(mel_db, axis=0)))

    # ── MFCC ──────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, hop_length=hop)
    mfcc_delta = librosa.feature.delta(mfcc)
    feats.mfcc_mean = np.mean(mfcc, axis=1)
    feats.mfcc_std = np.std(mfcc, axis=1)
    feats.mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    # ── Pitch (f0) ────────────────────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=hop
        )
        f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]

        if store_arrays:
            feats.f0_times = f0_times
            feats.f0_values = f0

        if len(voiced_f0) > 5:
            feats.f0_mean = float(np.nanmean(voiced_f0))
            feats.f0_std = float(np.nanstd(voiced_f0))
            # Coefficient of variation — lower = more regular = suspicious
            cv = feats.f0_std / (feats.f0_mean + 1e-8)
            # Normalize: cv < 0.05 → highly regular, cv > 0.2 → natural
            feats.f0_regularity = float(np.clip(1.0 - cv / 0.2, 0, 1))
            feats.voiced_fraction = float(np.sum(voiced_flag) / len(voiced_flag))
    except Exception:
        pass

    # ── RMS energy / dynamics ─────────────────────────────────────────────
    rms = librosa.feature.rms(y=audio, hop_length=hop)
    feats.rms_mean = float(np.mean(rms))
    feats.rms_std = float(np.std(rms))
    rms_db = librosa.amplitude_to_db(rms)
    feats.dynamic_range = float(np.max(rms_db) - np.min(rms_db))

    # ── Harmonic-to-Noise Ratio (HNR) ────────────────────────────────────
    try:
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_power = np.mean(harmonic ** 2)
        noise_power = np.mean((audio - harmonic) ** 2) + 1e-10
        feats.hnr = float(10 * np.log10(harmonic_power / noise_power + 1e-10))
    except Exception:
        feats.hnr = 0.0

    # ── Phase randomness ──────────────────────────────────────────────────
    # Vocoders produce structured phase; natural speech phase is more random
    phase_diff = np.diff(phase, axis=1)
    phase_diff_unwrapped = np.abs(np.angle(np.exp(1j * phase_diff)))
    feats.phase_randomness = float(np.mean(phase_diff_unwrapped))

    # ── Zero-crossing rate ────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop)
    feats.zcr_mean = float(np.mean(zcr))
    feats.zcr_std = float(np.std(zcr))

    # ── Silence analysis ──────────────────────────────────────────────────
    energy = rms[0]
    threshold = np.percentile(energy, 20)
    is_silent = energy < threshold
    feats.silence_ratio = float(np.mean(is_silent))

    # Regularity of silence intervals — TTS breath patterns too even
    if np.any(is_silent) and np.any(~is_silent):
        # Run-length encode silent stretches
        runs = _run_lengths(is_silent)
        silent_runs = [r for r, s in runs if s]
        if len(silent_runs) > 2:
            run_arr = np.array(silent_runs, dtype=float)
            feats.silence_regularity = float(1.0 - (np.std(run_arr) / (np.mean(run_arr) + 1e-8)))
        else:
            feats.silence_regularity = 0.5

    return feats


def _run_lengths(arr: np.ndarray):
    """Return list of (length, value) run-length pairs."""
    if len(arr) == 0:
        return []
    runs = []
    current = arr[0]
    count = 1
    for v in arr[1:]:
        if v == current:
            count += 1
        else:
            runs.append((count, current))
            current = v
            count = 1
    runs.append((count, current))
    return runs


def feature_vector(feats: AudioFeatures) -> np.ndarray:
    """Flatten features into a fixed-length vector for ML."""
    vec = [
        feats.spectral_flatness_mean,
        feats.spectral_flatness_std,
        feats.spectral_rolloff_mean / 1e4,   # normalize Hz
        feats.spectral_bandwidth_mean / 1e4,
        feats.spectral_flux,
        feats.mel_smoothness / 10.0,
        feats.mel_temporal_var / 100.0,
        feats.f0_mean / 500.0,
        feats.f0_std / 100.0,
        feats.f0_regularity,
        feats.voiced_fraction,
        feats.rms_mean * 100,
        feats.rms_std * 100,
        feats.dynamic_range / 60.0,
        feats.hnr / 40.0,
        feats.phase_randomness / np.pi,
        feats.zcr_mean * 10,
        feats.zcr_std * 10,
        feats.silence_ratio,
        feats.silence_regularity,
    ]
    vec += list(feats.mfcc_mean / 100.0)
    vec += list(feats.mfcc_std / 50.0)
    vec += list(feats.spectral_contrast_mean / 50.0)
    return np.array(vec, dtype=np.float32)
