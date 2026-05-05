import numpy as np
import librosa
import soundfile as sf
from typing import Tuple


def load_audio(audio_path: str, sr: int = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load audio file with librosa."""
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=mono)
    return y, sr_loaded


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or trim audio to target length in samples."""
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio[:target_length]


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to prevent clipping."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * target_peak
    return audio


def mix_tracks(
    track1: np.ndarray,
    track2: np.ndarray,
    track1_volume: float = 1.0,
    track2_volume: float = 1.0,
    target_sr: int = 44100
) -> np.ndarray:
    """Mix two audio tracks together."""
    # Ensure same length
    target_length = max(len(track1), len(track2))
    track1 = pad_or_trim(track1, target_length)
    track2 = pad_or_trim(track2, target_length)
    
    # Apply volumes
    track1 = track1 * track1_volume
    track2 = track2 * track2_volume
    
    # Mix
    mixed = track1 + track2
    
    # Normalize
    mixed = normalize_audio(mixed)
    
    return mixed


def clip_audio(audio_path: str, start_s: float, end_s: float, output_path: str, sr: int = 16000):
    """Clip a segment from an audio file."""
    y, sr_loaded = librosa.load(audio_path, sr=sr)
    start_sample = int(start_s * sr_loaded)
    end_sample = int(end_s * sr_loaded)
    segment = y[start_sample:end_sample]
    sf.write(output_path, segment, samplerate=sr_loaded)
    return output_path


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
