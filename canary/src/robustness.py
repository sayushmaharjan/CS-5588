"""
robustness.py — Adversarial robustness simulation.

Simulates real-world audio degradation:
  - Codec compression: μ-law (telephony 8kHz), simulated Opus/AAC artifacts
  - Downsampling to 8kHz and back (phone network)
  - Background noise (white, pink, babble)
  - Reverb simulation (room impulse response convolution)
  - Packet loss simulation (zeroing random frames)
  - Bandwidth limiting

Used for adversarial robustness testing: ensure detector works
even when audio is degraded by real-world transmission.
"""

import numpy as np
import librosa
import scipy.signal as signal
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class DegradationConfig:
    codec: Optional[str] = None      # 'ulaw', 'opus_sim', 'aac_sim'
    target_sr: Optional[int] = None  # downsample then upsample
    noise_type: Optional[str] = None # 'white', 'pink', 'babble'
    noise_snr_db: float = 20.0       # signal-to-noise ratio
    reverb_rt60: Optional[float] = None  # reverb time in seconds
    packet_loss_pct: float = 0.0     # % of frames to zero out
    bandwidth_limit_hz: Optional[float] = None


# ── μ-law codec (ITU-T G.711) ────────────────────────────────────────────────

def encode_ulaw(audio: np.ndarray, mu: int = 255) -> np.ndarray:
    """μ-law companding (quantize to 8-bit)."""
    audio = np.clip(audio, -1.0, 1.0)
    encoded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    # Quantize to 8-bit
    quantized = np.round(encoded * 127).astype(np.int8)
    return quantized


def decode_ulaw(encoded: np.ndarray, mu: int = 255) -> np.ndarray:
    """Inverse μ-law."""
    x = encoded.astype(np.float32) / 127.0
    decoded = np.sign(x) * (1 / mu) * ((1 + mu) ** np.abs(x) - 1)
    return decoded


def apply_ulaw_codec(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply μ-law encode/decode (G.711 telephony simulation)."""
    # First downsample to 8kHz (telephone bandwidth)
    audio_8k = librosa.resample(audio, orig_sr=sr, target_sr=8000)
    # Apply μ-law quantization
    encoded = encode_ulaw(audio_8k)
    decoded = decode_ulaw(encoded)
    # Upsample back
    audio_out = librosa.resample(decoded, orig_sr=8000, target_sr=sr)
    return audio_out.astype(np.float32)


# ── Opus/AAC simulation (spectral quantization artifacts) ────────────────────

def simulate_opus_artifacts(audio: np.ndarray, sr: int, bitrate_kbps: float = 24.0) -> np.ndarray:
    """
    Simulate Opus codec artifacts:
    - Frequency masking (zero low-energy subbands)
    - Temporal noise shaping (add quantization noise)
    - High-freq rolloff at low bitrates
    """
    # Compute MDCT-like subbands
    n_bands = 32
    band_size = len(audio) // n_bands

    out = audio.copy()

    # Quantization noise proportional to bitrate
    quality = min(bitrate_kbps / 128.0, 1.0)
    q_noise_level = (1.0 - quality) * 0.02
    noise = np.random.randn(len(audio)) * q_noise_level
    out = out + noise

    # High-frequency rolloff at low bitrates (< 32 kbps)
    if bitrate_kbps < 32:
        cutoff_ratio = 0.6 + (bitrate_kbps / 128.0) * 0.4
        cutoff_hz = cutoff_ratio * (sr / 2)
        b, a = signal.butter(4, cutoff_hz / (sr / 2), btype='low')
        out = signal.filtfilt(b, a, out)

    # Temporal noise shaping: add pre-echo artifact
    if bitrate_kbps < 16:
        pre_echo_delay = int(0.005 * sr)
        pre_echo = np.roll(out, -pre_echo_delay) * 0.05
        out = out + pre_echo

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def simulate_aac_artifacts(audio: np.ndarray, sr: int, bitrate_kbps: float = 64.0) -> np.ndarray:
    """
    Simulate AAC codec artifacts:
    - Spectral band replication approximation
    - Modified discrete cosine transform quantization
    """
    quality = min(bitrate_kbps / 192.0, 1.0)

    # Quantization noise
    q_noise = np.random.randn(len(audio)) * (1 - quality) * 0.015
    out = audio + q_noise

    # AAC psychoacoustic masking: reduce energy in low-energy frequency regions
    n_fft = 1024
    hop = 256
    D = librosa.stft(out, n_fft=n_fft, hop_length=hop)
    mag = np.abs(D)
    phase = np.angle(D)

    # Mask threshold: quantize low-energy bands
    mask_threshold = np.percentile(mag, 20 * (1 - quality))
    mag_quantized = np.where(mag < mask_threshold, 0, mag)

    D_out = mag_quantized * np.exp(1j * phase)
    out = librosa.istft(D_out, hop_length=hop)
    out = out[:len(audio)]

    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ── Noise addition ────────────────────────────────────────────────────────────

def add_noise(audio: np.ndarray, sr: int, noise_type: str = "white",
              snr_db: float = 20.0) -> np.ndarray:
    """Add realistic noise at specified SNR."""
    signal_power = np.mean(audio ** 2) + 1e-10

    if noise_type == "white":
        noise = np.random.randn(len(audio))

    elif noise_type == "pink":
        # 1/f noise via spectral shaping
        n = len(audio)
        f = np.fft.rfftfreq(n)
        f[0] = 1e-10  # avoid divide by zero
        psd = 1.0 / f ** 0.5
        phase = np.random.uniform(0, 2 * np.pi, len(f))
        spectrum = psd * np.exp(1j * phase)
        noise = np.fft.irfft(spectrum, n=n)

    elif noise_type == "babble":
        # Babble noise: sum of multiple synthetic voices (simplified)
        n_voices = 8
        noise = np.zeros(len(audio))
        for _ in range(n_voices):
            # Random modulated sinusoids (approximate speech babble)
            freq = np.random.uniform(100, 3000)
            env_freq = np.random.uniform(3, 8)
            t = np.arange(len(audio)) / sr
            voice = np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * env_freq * t))
            voice += np.random.randn(len(audio)) * 0.3
            noise += voice / n_voices

    else:
        noise = np.random.randn(len(audio))

    # Normalize noise to target SNR
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    noise = noise * scale

    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


# ── Reverb simulation ─────────────────────────────────────────────────────────

def add_reverb(audio: np.ndarray, sr: int, rt60: float = 0.3, room_size: float = 0.5) -> np.ndarray:
    """
    Simulate room reverb using exponentially decaying noise IR.
    
    Args:
        rt60: reverberation time (seconds to decay 60dB)
        room_size: 0-1 scale factor for room character
    """
    # Generate synthetic room impulse response
    ir_len = int(rt60 * sr)
    t = np.linspace(0, rt60, ir_len)

    # Exponential decay
    decay = np.exp(-6.91 * t / rt60)  # 6.91 = -60dB decay constant

    # Modulate with room modes
    ir = np.random.randn(ir_len) * decay
    ir[0] = 1.0  # Direct path

    # Early reflections (discrete delays)
    n_reflections = int(5 * room_size)
    for i in range(n_reflections):
        delay = int(np.random.uniform(0.002, 0.05) * sr)
        amplitude = np.random.uniform(0.3, 0.8) * (0.8 ** i)
        if delay < ir_len:
            ir[delay] += amplitude * np.random.choice([-1, 1])

    # Normalize IR
    ir = ir / (np.max(np.abs(ir)) + 1e-8)

    # Convolve
    reverbed = signal.fftconvolve(audio, ir, mode='full')[:len(audio)]
    return np.clip(reverbed, -1.0, 1.0).astype(np.float32)


# ── Packet loss ───────────────────────────────────────────────────────────────

def simulate_packet_loss(audio: np.ndarray, sr: int, loss_pct: float = 5.0,
                         packet_ms: float = 20.0) -> np.ndarray:
    """
    Zero out random audio packets (VoIP packet loss simulation).
    
    Args:
        loss_pct: percentage of packets to drop
        packet_ms: packet duration in milliseconds
    """
    packet_samples = int(packet_ms * sr / 1000)
    n_packets = len(audio) // packet_samples
    out = audio.copy()

    for i in range(n_packets):
        if np.random.random() < loss_pct / 100.0:
            start = i * packet_samples
            end = start + packet_samples
            # Concealment: interpolate or zero
            out[start:end] = np.linspace(
                out[start] if start > 0 else 0,
                out[end] if end < len(out) else 0,
                packet_samples
            ) * 0.1  # Mostly zero with slight interpolation artifact

    return out.astype(np.float32)


# ── Bandwidth limiting ────────────────────────────────────────────────────────

def limit_bandwidth(audio: np.ndarray, sr: int, cutoff_hz: float = 3400.0) -> np.ndarray:
    """Telephone bandwidth: 300-3400 Hz (POTS network)."""
    # High-pass at 300 Hz
    b_hp, a_hp = signal.butter(4, 300 / (sr / 2), btype='high')
    out = signal.filtfilt(b_hp, a_hp, audio)
    # Low-pass at cutoff
    b_lp, a_lp = signal.butter(4, cutoff_hz / (sr / 2), btype='low')
    out = signal.filtfilt(b_lp, a_lp, out)
    return out.astype(np.float32)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def apply_degradation(audio: np.ndarray, sr: int, config: DegradationConfig) -> np.ndarray:
    """Apply configured degradation chain to audio."""
    out = audio.copy()

    # Bandwidth limit first (before codec)
    if config.bandwidth_limit_hz:
        out = limit_bandwidth(out, sr, config.bandwidth_limit_hz)

    # Downsample / upsample
    if config.target_sr and config.target_sr != sr:
        out = librosa.resample(out, orig_sr=sr, target_sr=config.target_sr)
        out = librosa.resample(out, orig_sr=config.target_sr, target_sr=sr)

    # Codec
    if config.codec == "ulaw":
        out = apply_ulaw_codec(out, sr)
    elif config.codec == "opus_sim":
        out = simulate_opus_artifacts(out, sr, bitrate_kbps=24.0)
    elif config.codec == "aac_sim":
        out = simulate_aac_artifacts(out, sr, bitrate_kbps=64.0)

    # Packet loss
    if config.packet_loss_pct > 0:
        out = simulate_packet_loss(out, sr, config.packet_loss_pct)

    # Reverb
    if config.reverb_rt60 is not None:
        out = add_reverb(out, sr, config.reverb_rt60)

    # Noise (last, to not get double-degraded)
    if config.noise_type:
        out = add_noise(out, sr, config.noise_type, config.noise_snr_db)

    return out


def benchmark_degradation_configs() -> List[Dict]:
    """Return standard test degradation configs."""
    return [
        {"name": "Clean",          "config": DegradationConfig()},
        {"name": "Phone (μ-law)",  "config": DegradationConfig(codec="ulaw", target_sr=8000, bandwidth_limit_hz=3400)},
        {"name": "VoIP (Opus)",    "config": DegradationConfig(codec="opus_sim", packet_loss_pct=3.0)},
        {"name": "Noisy (SNR 10)", "config": DegradationConfig(noise_type="white", noise_snr_db=10.0)},
        {"name": "Reverb Room",    "config": DegradationConfig(reverb_rt60=0.5, noise_type="pink", noise_snr_db=25.0)},
        {"name": "Babble Noise",   "config": DegradationConfig(noise_type="babble", noise_snr_db=15.0)},
        {"name": "Worst Case",     "config": DegradationConfig(
            codec="ulaw", target_sr=8000, noise_type="babble",
            noise_snr_db=8.0, reverb_rt60=0.4, packet_loss_pct=5.0,
            bandwidth_limit_hz=3400
        )},
    ]
