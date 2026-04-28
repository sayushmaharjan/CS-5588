"""
Cinematic Memory — Ambient Sound Design Module
Uses AudioLDM2 (cvssp/audioldm2) for environmental scene-matched audio.
Generates ambient sound per beat, mixed under narration.
"""
from __future__ import annotations
import os, logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

_audioldm2_pipe = None
_device         = None


def _load_audioldm2():
    global _audioldm2_pipe, _device
    if _audioldm2_pipe is not None:
        return
    try:
        import torch
        from diffusers import AudioLDM2Pipeline
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        from config import AUDIOLDM2_MODEL_ID
        logger.info(f"Loading AudioLDM2 on {_device}… (may take a minute)")
        _audioldm2_pipe = AudioLDM2Pipeline.from_pretrained(
            AUDIOLDM2_MODEL_ID,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        ).to(_device)
        logger.info("AudioLDM2 loaded ✓")
    except Exception as e:
        logger.warning(f"AudioLDM2 load failed: {e}. Using silence fallback.")


def _generate_ambient(prompt: str, duration_s: float) -> Optional[tuple]:
    """
    Generate ambient audio clip.
    Returns (audio_array, sample_rate) or None.
    """
    # AudioLDM2 default output: 16kHz, ~10s
    effective_dur = min(duration_s, 10.0)

    audio_output = _audioldm2_pipe(
        prompt,
        num_inference_steps=20,    # balance quality vs speed
        audio_length_in_s=effective_dur,
        num_waveforms_per_prompt=1,
        guidance_scale=3.5,
    )
    audio = audio_output.audios[0]  # shape: (samples,)
    sample_rate = 16000
    return audio, sample_rate


def _save_audio(audio: np.ndarray, path: str, sample_rate: int = 16000):
    """Save audio to WAV. Multi-backend fallback."""
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


def _generate_procedural_ambient(scene_type: str, duration_s: float,
                                  sample_rate: int = 16000) -> np.ndarray:
    """
    Procedural ambient audio fallback when AudioLDM2 is unavailable.
    Generates barely-audible, smooth room-tone noise (no harsh static).
    Amplitude is kept very low so it sits under narration without distraction.
    """
    n_samples = int(duration_s * sample_rate)
    t         = np.linspace(0, duration_s, n_samples, endpoint=False)
    rng       = np.random.default_rng(sum(ord(c) for c in scene_type))

    # Very soft base noise — max 2% amplitude (well below static threshold)
    noise_vol = {"beach":    0.018, "wedding":  0.008, "city":     0.014,
        "indoors":  0.005, "nature":   0.010, "party":    0.015,
        "travel":   0.012, "portrait": 0.004, "unknown":  0.006,
    }.get(scene_type, 0.006)
    

def _generate_real_ambient_fallback(scene_type: str, duration_s: float, output_path: str):
    """
    Downloads real ambient audio from FreeSound (preview links), loops to duration,
    and saves to output_path. Replaces white noise procedural fallback.
    """
    try:
        import requests
        import tempfile
        from pydub import AudioSegment
        
        # Real audio preview links scraped from FreeSound
        scene_urls = {
            "beach": "https://cdn.freesound.org/previews/789/789503_16787921-lq.mp3",
            "city": "https://cdn.freesound.org/previews/345/345313_6140900-lq.mp3",
            "nature": "https://cdn.freesound.org/previews/462/462137_8192401-lq.mp3",
            "wedding": "https://cdn.freesound.org/previews/609/609204_2282212-lq.mp3",
            "party": "https://cdn.freesound.org/previews/609/609204_2282212-lq.mp3",
            "travel": "https://cdn.freesound.org/previews/345/345313_6140900-lq.mp3",
            "unknown": "https://cdn.freesound.org/previews/736/736518_1648170-lq.mp3",
            "indoors": "https://cdn.freesound.org/previews/736/736518_1648170-lq.mp3",
            "portrait": "https://cdn.freesound.org/previews/736/736518_1648170-lq.mp3",
        }
        
        url = scene_urls.get(scene_type, scene_urls["unknown"])
        
        # Download the MP3
        temp_mp3 = output_path.replace(".wav", "_temp.mp3")
        r = requests.get(url, timeout=10)
        with open(temp_mp3, "wb") as f:
            f.write(r.content)
            
        # Load and process with pydub
        audio = AudioSegment.from_mp3(temp_mp3)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Loop until target duration
        target_ms = int(duration_s * 1000)
        while len(audio) < target_ms:
            audio += audio
            
        # Trim and apply fade
        audio = audio[:target_ms]
        audio = audio.fade_in(1000).fade_out(1000)
        
        # Save
        audio.export(output_path, format="wav")
        
        # Cleanup
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
            
    except Exception as e:
        logger.error(f"Real ambient fallback failed: {e}")
        # Ultimate silent fallback
        import numpy as np
        audio_arr = np.zeros(int(duration_s * 16000), dtype=np.float32)
        _save_audio(audio_arr, output_path, 16000)


def generate_ambient_for_beat(
    beat_id:    str,
    prompt:     str,
    duration_s: float,
    output_dir: str,
    scene_type: str = "unknown",
) -> "AmbientSegment":
    """
    Generate ambient sound for a single script beat.
    Args:
        beat_id:    beat identifier
        prompt:     AudioLDM2 text prompt
        duration_s: target duration
        output_dir: save directory
        scene_type: scene type tag
    Returns:
        AmbientSegment with path and metadata
    """
    from utils.data_schemas import AmbientSegment, SceneType
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{beat_id}_ambient.wav")

    _load_audioldm2()

    if _audioldm2_pipe is None:
        logger.warning(f"AudioLDM2 unavailable — using real ambient fallback for beat {beat_id}")
        _generate_real_ambient_fallback(scene_type, duration_s, output_path)
        return AmbientSegment(
            beat_id=beat_id, audio_path=output_path,
            duration_s=duration_s, scene_type=SceneType(scene_type) if scene_type in SceneType._value2member_map_ else SceneType.UNKNOWN, prompt_used="real ambient fallback",
        )

    try:
        logger.info(f"Generating ambient for beat {beat_id}: '{prompt[:60]}…'")
        audio, sr = _generate_ambient(prompt, duration_s)
        _save_audio(audio, output_path, sr)
        actual_dur = len(audio) / sr

        return AmbientSegment(
            beat_id=beat_id, audio_path=output_path,
            duration_s=actual_dur, prompt_used=prompt,
            scene_type=SceneType(scene_type) if scene_type in SceneType._value2member_map_ else SceneType.UNKNOWN,
        )
    except Exception as e:
        logger.error(f"AudioLDM2 failed for beat {beat_id}: {e}")
        _generate_real_ambient_fallback(scene_type, duration_s, output_path)
        return AmbientSegment(
            beat_id=beat_id, audio_path=output_path,
            duration_s=duration_s, scene_type=SceneType.UNKNOWN, prompt_used="real ambient fallback",
        )


def generate_all_ambient(
    script:     "DocumentaryScript",
    visual_meta: Dict[str, "VisualMetadata"],
    output_dir: str,
) -> Dict[str, "AmbientSegment"]:
    """
    Generate ambient sounds for all beats.
    Picks dominant scene_type from beat's media_ids to auto-select prompt.
    """
    from config import SCENE_AMBIENT_PROMPTS
    ambient_segs = {}

    for beat in script.beats:
        # Determine dominant scene from assigned media
        scene_counts: Dict[str, int] = {}
        for mid in beat.media_ids:
            if mid in visual_meta:
                st = visual_meta[mid].scene_type.value
                scene_counts[st] = scene_counts.get(st, 0) + 1

        dominant_scene = max(scene_counts, key=scene_counts.get) if scene_counts else "unknown"

        # Use beat's ambient_prompt (from LLM) or fallback to scene map
        prompt = beat.ambient_prompt or SCENE_AMBIENT_PROMPTS.get(dominant_scene,
            "gentle neutral ambiance, soft room tone")

        seg = generate_ambient_for_beat(
            beat_id=beat.beat_id, prompt=prompt,
            duration_s=beat.duration_hint_s,
            output_dir=output_dir, scene_type=dominant_scene,
        )
        ambient_segs[beat.beat_id] = seg

    return ambient_segs


# ── Single Global Ambient Track ──────────────────────────────────────────────

AMBIENT_SCENES = {
    "nature": {
        "label": "🌿 Nature",
        "description": "Birds, rustling leaves, gentle breeze",
        "url": "https://cdn.freesound.org/previews/462/462137_8192401-lq.mp3",
    },
    "beach": {
        "label": "🌊 Beach",
        "description": "Ocean waves, seagulls, sea breeze",
        "url": "https://cdn.freesound.org/previews/789/789503_16787921-lq.mp3",
    },
    "city": {
        "label": "🌆 City",
        "description": "Traffic hum, distant voices, urban energy",
        "url": "https://cdn.freesound.org/previews/345/345313_6140900-lq.mp3",
    },
    "cafe": {
        "label": "☕ Café",
        "description": "Soft chatter, clinking cups, cozy indoors",
        "url": "https://cdn.freesound.org/previews/736/736518_1648170-lq.mp3",
    },
    "crowd": {
        "label": "🎉 Crowd",
        "description": "Lively crowd, celebration, energy",
        "url": "https://cdn.freesound.org/previews/609/609204_2282212-lq.mp3",
    },
    "rain": {
        "label": "🌧️ Rain",
        "description": "Soft rain on windows, meditative",
        "url": "https://cdn.freesound.org/previews/346/346642_5121236-lq.mp3",
    },
    "silence": {
        "label": "🔇 Minimal",
        "description": "Very subtle room tone, nearly silent",
        "url": None,
    },
}


def generate_single_ambient_track(
    scene:      str,
    duration_s: float,
    output_dir: str,
    output_name: str = "global_ambient.wav",
) -> str:
    """
    Generate a single ambient track for the entire video.
    Downloads from FreeSound, loops to fit, adds crossfades.
    Returns output WAV path.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    scene_info  = AMBIENT_SCENES.get(scene, AMBIENT_SCENES["nature"])
    url         = scene_info.get("url")

    if url:
        try:
            import requests
            from pydub import AudioSegment
            temp_mp3 = output_path.replace(".wav", "_temp.mp3")
            logger.info(f"Downloading ambient '{scene}' from FreeSound…")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(temp_mp3, "wb") as f:
                    f.write(r.content)
                audio = AudioSegment.from_mp3(temp_mp3)
                audio = audio.set_frame_rate(16000).set_channels(1)
                # Lower volume to -18 dB so it sits softly under narration
                audio = audio - 18
                target_ms = int(duration_s * 1000)
                while len(audio) < target_ms:
                    audio += audio
                audio = audio[:target_ms].fade_in(2000).fade_out(3000)
                audio.export(output_path, format="wav")
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
                return output_path
        except Exception as e:
            logger.warning(f"FreeSound ambient download failed: {e}")

    # Silent/minimal fallback
    logger.info(f"Using silent ambient fallback for scene '{scene}'")
    n = int(duration_s * 16000)
    audio_arr = np.zeros(n, dtype=np.float32)
    _save_audio(audio_arr, output_path, 16000)
    return output_path