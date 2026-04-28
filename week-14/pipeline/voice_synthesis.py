"""
Cinematic Memory — Voice Synthesis Module
Uses Chatterbox TTS (Resemble AI) for narrator voice.
Generates timestamped narration audio per script beat.
"""
from __future__ import annotations
import os, logging, torch
from typing import List, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level model cache ──────────────────────────────────────────────
_chatterbox_model = None
_chatterbox_device = None


def _get_chatterbox_model():
    """Lazy-load and cache the Chatterbox TTS model."""
    global _chatterbox_model, _chatterbox_device

    if _chatterbox_model is not None:
        return _chatterbox_model

    from chatterbox.tts import ChatterboxTTS

    # Detect best device (Apple Silicon → mps, NVIDIA → cuda, else cpu)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    _chatterbox_device = device
    logger.info(f"Loading Chatterbox TTS model on device: {device}")

    # Mac MPS compatibility patch (from official example_for_mac.py)
    if device == "mps":
        map_location = torch.device(device)
        _original_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = map_location
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

    _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    logger.info("Chatterbox TTS model loaded successfully")
    return _chatterbox_model


def _generate_chatterbox_audio(text: str, output_path: str, reference_audio_path: Optional[str] = None) -> float:
    """
    Text-to-speech using Chatterbox TTS (Resemble AI).
    Saves WAV to output_path and returns duration in seconds.
    """
    try:
        import torchaudio as ta

        model = _get_chatterbox_model()

        # Generate speech — default voice, or use reference clip if provided
        if reference_audio_path and os.path.exists(reference_audio_path):
            wav = model.generate(text, audio_prompt_path=reference_audio_path)
        else:
            wav = model.generate(text)

        # Save as WAV at the model's native sample rate
        ta.save(output_path, wav, model.sr)

        # Compute duration from tensor shape
        duration_s = wav.shape[-1] / model.sr
        logger.info(f"Chatterbox TTS: saved {duration_s:.1f}s audio to {output_path}")
        return duration_s

    except Exception as e:
        logger.error(f"Chatterbox TTS failed: {e}")
        # Fallback: generate silence with estimated duration
        words = text.split()
        estimated_duration_s = max(2.0, len(words) / 130 * 60)
        n_samples = int(estimated_duration_s * 16000)

        # Save silence
        try:
            import scipy.io.wavfile as wav
            audio_int16 = np.zeros(n_samples, dtype=np.int16)
            wav.write(output_path, 16000, audio_int16)
        except ImportError:
            import wave
            audio_int16 = np.zeros(n_samples, dtype=np.int16)
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())

        return estimated_duration_s


def synthesize_beat(
    beat_id:    str,
    text:       str,
    output_dir: str,
    emotion:    str = "neutral",
    voice_reference_path: Optional[str] = None,
) -> "NarrationAudio":
    """
    Synthesize narration for a single script beat.
    Args:
        beat_id:    unique beat identifier
        text:       narration text to speak
        output_dir: directory to save audio file
        emotion:    emotion tag
        voice_reference_path: optional path to a reference audio file for voice cloning
    Returns:
        NarrationAudio with path and duration
    """
    from utils.data_schemas import NarrationAudio, EmotionTag

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{beat_id}_narration.wav")
    emo_tag = EmotionTag(emotion) if emotion in EmotionTag._value2member_map_ else EmotionTag.NEUTRAL

    logger.info(f"Synthesizing narration for beat {beat_id} ({len(text)} chars)…")
    duration_s = _generate_chatterbox_audio(text, output_path, reference_audio_path=voice_reference_path)
    
    return NarrationAudio(
        beat_id=beat_id, audio_path=output_path,
        duration_s=duration_s, emotion=emo_tag, text=text,
    )


def synthesize_all_beats(
    script: "DocumentaryScript",
    output_dir: str,
    voice_reference_path: Optional[str] = None,
) -> Dict[str, "NarrationAudio"]:
    """
    Synthesize narration for all beats in a documentary script.
    Returns dict: beat_id → NarrationAudio
    """
    narrations = {}
    for beat in script.beats:
        narration = synthesize_beat(
            beat_id              = beat.beat_id,
            text                 = beat.narration_text,
            output_dir           = output_dir,
            emotion              = beat.emotion.value,
            voice_reference_path = voice_reference_path,
        )
        narrations[beat.beat_id] = narration
        logger.info(f"  Beat {beat.beat_id}: {narration.duration_s:.1f}s synthesized")
    return narrations