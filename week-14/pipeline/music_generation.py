"""
Cinematic Memory — Music Generation Module
Uses facebook/musicgen-small for emotion-conditioned adaptive soundtrack.
Generates one music segment per script beat (segment-level conditioning).
Includes global-prompt vs segment-level A/B experiment.
"""
from __future__ import annotations
import os, logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

_musicgen_model     = None
_musicgen_processor = None
_device             = None


def _load_musicgen():
    global _musicgen_model, _musicgen_processor, _device
    if _musicgen_model is not None:
        return
    try:
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        from config import MUSICGEN_MODEL_ID
        logger.info(f"Loading MusicGen on {_device}…")
        _musicgen_processor = AutoProcessor.from_pretrained(MUSICGEN_MODEL_ID)
        _musicgen_model     = MusicgenForConditionalGeneration.from_pretrained(
            MUSICGEN_MODEL_ID
        ).to(_device)
        logger.info("MusicGen loaded ✓")
    except Exception as e:
        logger.warning(f"MusicGen load failed: {e}. Will use silence fallback.")


def _generate_music_clip(prompt: str, duration_s: float) -> Optional[np.ndarray]:
    """Generate music audio array for given prompt and duration."""
    import torch
    from config import MUSICGEN_DURATION

    # MusicGen-small max tokens ≈ 1503 ≈ 30s at 32 tokens/s
    # Clamp duration
    effective_dur = min(duration_s, 30.0)
    max_new_tokens = int(effective_dur * 50)  # ~50 tokens/s for musicgen-small

    inputs = _musicgen_processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        audio_values = _musicgen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )

    # audio_values: [batch, channels, samples]
    sample_rate = _musicgen_model.config.audio_encoder.sampling_rate
    audio = audio_values[0, 0].cpu().numpy().astype(np.float32)
    return audio, sample_rate


def _save_audio(audio: np.ndarray, path: str, sample_rate: int = 32000):
    """Save audio array to WAV. Multi-backend fallback."""
    try:
        import soundfile as sf
        sf.write(path, audio, sample_rate)
        return
    except ImportError:
        pass
    try:
        import scipy.io.wavfile as wav
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wav.write(path, sample_rate, audio_int16)
        return
    except ImportError:
        pass
    import wave
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def _generate_procedural_music(emotion: str, duration_s: float, sample_rate: int = 32000) -> np.ndarray:
    """
    Procedural music fallback when MusicGen is unavailable.
    Generates emotion-matched chord tones with soft pad texture.
    """
    # Emotion → root freq + chord intervals (semitones from root)
    emotion_params = {
        "joyful":       (261.63, [0, 4, 7, 12], 0.9),    # C major, bright
        "nostalgic":    (220.00, [0, 3, 7, 10], 0.6),    # A minor 7, warm
        "reflective":   (196.00, [0, 3, 7,  9], 0.5),    # G minor, contemplative
        "sad":          (185.00, [0, 3, 6, 10], 0.45),   # F# dim 7, melancholic
        "excited":      (293.66, [0, 4, 7, 11], 1.0),    # D major 7, energetic
        "celebratory":  (349.23, [0, 4, 7, 12], 1.0),   # F major, triumphant
        "tender":       (246.94, [0, 4, 7,  9], 0.55),  # B major 6, intimate
        "neutral":      (261.63, [0, 4, 7],     0.5),   # C major, plain
    }
    root, intervals, vol = emotion_params.get(emotion, emotion_params["neutral"])

    n_samples = int(duration_s * sample_rate)
    t         = np.linspace(0, duration_s, n_samples, endpoint=False)
    audio     = np.zeros(n_samples, dtype=np.float32)

    # Add each chord tone as a soft sine with slight detuning (chorus effect)
    for i, semitones in enumerate(intervals):
        freq      = root * (2 ** (semitones / 12))
        detune    = 1.0 + (i * 0.0015)  # tiny detuning for richness
        amplitude = vol / (len(intervals) * 1.4)
        audio    += (amplitude * np.sin(2 * np.pi * freq * detune * t)).astype(np.float32)

    # Slow amplitude modulation (tremolo) for organic feel
    if emotion in ("nostalgic", "reflective", "sad", "tender"):
        lfo    = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * t)  # 0.25Hz tremolo
        audio *= lfo.astype(np.float32)

    # Soft fade in/out
    fade = min(int(sample_rate * 1.5), n_samples // 4)
    audio[:fade]  *= np.linspace(0, 1, fade).astype(np.float32)
    audio[-fade:] *= np.linspace(1, 0, fade).astype(np.float32)

    return audio


def _silence_fallback(duration_s: float, path: str, sample_rate: int = 32000):
    audio = _generate_procedural_music("neutral", duration_s, sample_rate)
    _save_audio(audio, path, sample_rate)


def generate_music_for_beat(
    beat_id:    str,
    prompt:     str,
    duration_s: float,
    output_dir: str,
    emotion:    str = "neutral",
) -> "MusicSegment":
    """
    Generate adaptive music for a single script beat.
    Args:
        beat_id:    beat identifier
        prompt:     MusicGen text prompt (emotion-specific)
        duration_s: target music duration
        output_dir: save directory
        emotion:    emotion tag
    Returns:
        MusicSegment with path and metadata
    """
    from utils.data_schemas import MusicSegment, EmotionTag
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{beat_id}_music.wav")

    _load_musicgen()

    if _musicgen_model is None:
        logger.warning(f"MusicGen unavailable — using procedural music for beat {beat_id}")
        audio = _generate_procedural_music(emotion, duration_s, sample_rate=32000)
        _save_audio(audio, output_path, 32000)
        return MusicSegment(
            beat_id=beat_id, audio_path=output_path,
            duration_s=len(audio)/32000, prompt_used=prompt,
            emotion=EmotionTag(emotion) if emotion in EmotionTag._value2member_map_ else EmotionTag.NEUTRAL,
        )

    try:
        logger.info(f"Generating music for beat {beat_id}: '{prompt[:60]}…'")
        audio, sample_rate = _generate_music_clip(prompt, duration_s)
        _save_audio(audio, output_path, sample_rate)
        actual_dur = len(audio) / sample_rate

        return MusicSegment(
            beat_id=beat_id, audio_path=output_path,
            duration_s=actual_dur, prompt_used=prompt,
            emotion=EmotionTag(emotion) if emotion in EmotionTag._value2member_map_ else EmotionTag.NEUTRAL,
        )
    except Exception as e:
        logger.error(f"MusicGen failed for beat {beat_id}: {e} — using procedural music")
        audio = _generate_procedural_music(emotion, duration_s, sample_rate=32000)
        _save_audio(audio, output_path, 32000)
        return MusicSegment(
            beat_id=beat_id, audio_path=output_path,
            duration_s=len(audio)/32000, prompt_used=prompt,
            emotion=EmotionTag.NEUTRAL,
        )


def generate_music_global(
    script:     "DocumentaryScript",
    output_dir: str,
    global_prompt: str = "cinematic documentary soundtrack, warm orchestral, emotional journey",
) -> Dict[str, "MusicSegment"]:
    """
    EXPERIMENT MODE A: Single global prompt for entire film.
    Less varied but more coherent. Compare vs segment-level.
    """
    music_segs = {}
    for beat in script.beats:
        seg = generate_music_for_beat(
            beat_id=beat.beat_id, prompt=global_prompt,
            duration_s=beat.duration_hint_s,
            output_dir=output_dir, emotion=beat.emotion.value,
        )
        music_segs[beat.beat_id] = seg
    return music_segs


def generate_music_segment_level(
    script:     "DocumentaryScript",
    output_dir: str,
) -> Dict[str, "MusicSegment"]:
    """
    EXPERIMENT MODE B: Segment-level emotion-conditioned prompts.
    Each beat gets its own music prompt → better arc alignment.
    """
    music_segs = {}
    for beat in script.beats:
        seg = generate_music_for_beat(
            beat_id=beat.beat_id, prompt=beat.music_prompt,
            duration_s=beat.duration_hint_s,
            output_dir=output_dir, emotion=beat.emotion.value,
        )
        music_segs[beat.beat_id] = seg
    return music_segs


def generate_all_music(
    script:     "DocumentaryScript",
    output_dir: str,
    mode:       str = "segment",   # "segment" | "global"
    global_prompt: str = "",
) -> Dict[str, "MusicSegment"]:
    """Main entry point. mode='segment' (default) or 'global' (experiment)."""
    if mode == "global":
        return generate_music_global(script, output_dir, global_prompt or
            "cinematic documentary soundtrack, warm orchestral, emotional journey")
    else:
        return generate_music_segment_level(script, output_dir)


# ── Single Global Music Track (new unified approach) ─────────────────────────

MUSIC_MOODS = {
    "joyful": {
        "label": "😄 Joyful",
        "prompt": "uplifting happy acoustic guitar, warm bright major key, sunny day",
        "freesound_url": "https://cdn.freesound.org/previews/612/612094_5674468-lq.mp3",
        "description": "Bright and warm — perfect for happy memories",
    },
    "nostalgic": {
        "label": "🥺 Nostalgic",
        "prompt": "nostalgic slow piano melody, emotional warmth, bittersweet strings, memories",
        "freesound_url": "https://cdn.freesound.org/previews/612/612095_5674468-lq.mp3",
        "description": "Gentle and bittersweet — looking back with warmth",
    },
    "adventurous": {
        "label": "🏔️ Adventurous",
        "prompt": "epic adventurous orchestral music, rising strings, triumphant horns, exploration",
        "freesound_url": "https://cdn.freesound.org/previews/612/612096_5674468-lq.mp3",
        "description": "Bold and sweeping — grand cinematic journey",
    },
    "romantic": {
        "label": "💕 Romantic",
        "prompt": "romantic soft piano and strings, tender intimate melody, love story soundtrack",
        "freesound_url": "https://cdn.freesound.org/previews/612/612097_5674468-lq.mp3",
        "description": "Tender and warm — for love and connection",
    },
    "sad": {
        "label": "😢 Melancholic",
        "prompt": "slow melancholic strings, minor key, soft and tender, reflective sorrow",
        "freesound_url": "https://cdn.freesound.org/previews/612/612098_5674468-lq.mp3",
        "description": "Quiet and sorrowful — honest and real",
    },
    "excited": {
        "label": "🤩 Excited",
        "prompt": "energetic upbeat music, driving rhythm, vibrant dynamic, full band",
        "freesound_url": "https://cdn.freesound.org/previews/612/612099_5674468-lq.mp3",
        "description": "High-energy and vibrant — for peak moments",
    },
    "calm": {
        "label": "🌿 Calm",
        "prompt": "calm ambient music, soft pads, nature sounds, peaceful meditation",
        "freesound_url": "https://cdn.freesound.org/previews/612/612100_5674468-lq.mp3",
        "description": "Peaceful and grounding — gentle and serene",
    },
    "cinematic": {
        "label": "🎬 Cinematic",
        "prompt": "cinematic orchestral documentary soundtrack, emotional depth, sweeping strings",
        "freesound_url": "https://cdn.freesound.org/previews/612/612101_5674468-lq.mp3",
        "description": "Epic and movie-like — professional documentary feel",
    },
}

# Procedural music params for each mood (used as fallback)
_MOOD_PROCEDURAL = {
    "joyful":       ("joyful",      261.63, [0, 4, 7, 12], 0.9),
    "nostalgic":    ("nostalgic",   220.00, [0, 3, 7, 10], 0.6),
    "adventurous":  ("excited",     293.66, [0, 4, 7, 11], 0.85),
    "romantic":     ("tender",      246.94, [0, 4, 7,  9], 0.55),
    "sad":          ("sad",         185.00, [0, 3, 6, 10], 0.45),
    "excited":      ("excited",     293.66, [0, 4, 7, 11], 1.0),
    "calm":         ("reflective",  196.00, [0, 3, 7,  9], 0.4),
    "cinematic":    ("nostalgic",   196.00, [0, 3, 7, 12], 0.7),
}


def generate_single_music_track(
    mood:       str,
    duration_s: float,
    output_dir: str,
    output_name: str = "global_music.wav",
) -> str:
    """
    Generate a single music track for the entire video.
    Tries MusicGen → FreeSound download → procedural fallback.
    Returns the output WAV path.
    """
    from utils.data_schemas import EmotionTag
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    mood_info = MUSIC_MOODS.get(mood, MUSIC_MOODS["cinematic"])
    prompt    = mood_info["prompt"]

    # 1. Try MusicGen
    _load_musicgen()
    if _musicgen_model is not None:
        try:
            logger.info(f"Generating global music ({mood}, {duration_s:.0f}s) via MusicGen…")
            audio, sr = _generate_music_clip(prompt, duration_s)
            # Loop if MusicGen returned shorter than needed
            while len(audio) / sr < duration_s * 0.9:
                audio = np.concatenate([audio, audio])
            audio = audio[:int(duration_s * sr)]
            _save_audio(audio, output_path, sr)
            return output_path
        except Exception as e:
            logger.warning(f"MusicGen failed for global track: {e}")

    # 2. FreeSound download fallback
    try:
        import requests
        from pydub import AudioSegment
        url = mood_info.get("freesound_url", "")
        if url:
            logger.info(f"Downloading music from FreeSound for mood '{mood}'…")
            temp_mp3 = output_path.replace(".wav", "_temp.mp3")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(temp_mp3, "wb") as f:
                    f.write(r.content)
                audio = AudioSegment.from_mp3(temp_mp3)
                audio = audio.set_frame_rate(32000).set_channels(1)
                target_ms = int(duration_s * 1000)
                while len(audio) < target_ms:
                    audio += audio
                audio = audio[:target_ms].fade_in(2000).fade_out(3000)
                audio.export(output_path, format="wav")
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
                return output_path
    except Exception as e:
        logger.warning(f"FreeSound download failed for music: {e}")

    # 3. Procedural fallback
    logger.info(f"Using procedural music fallback for mood '{mood}'")
    emo_key, root, intervals, vol = _MOOD_PROCEDURAL.get(mood, _MOOD_PROCEDURAL["cinematic"])
    sr = 32000
    n  = int(duration_s * sr)
    t  = np.linspace(0, duration_s, n, endpoint=False)
    audio = np.zeros(n, dtype=np.float32)
    for i, semitones in enumerate(intervals):
        freq   = root * (2 ** (semitones / 12))
        detune = 1.0 + i * 0.0015
        amplitude = vol / (len(intervals) * 1.4)
        audio += (amplitude * np.sin(2 * np.pi * freq * detune * t)).astype(np.float32)
    if mood in ("nostalgic", "sad", "calm", "romantic"):
        lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
        audio *= lfo.astype(np.float32)
    fade = min(int(sr * 2.0), n // 4)
    audio[:fade]  *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    _save_audio(audio, output_path, sr)
    return output_path