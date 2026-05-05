"""
sound_engine.py
~~~~~~~~~~~~~~~
Generates ambient background music and mixes per-page audio.

Reuses:
  - MusicVideoAudioGenerator.generate_instrumental() for MusicGen ambient tracks
  - audio_utils.mix_tracks() for narration + SFX + music mixing

New:
  - MOOD_TO_MUSIC_PROMPT mapping (calm → soft piano, etc.)
  - SFX_MAP for file-based sound effects
  - Per-page audio mixing (narration is primary, music + SFX are background)
"""

import os
import logging
import numpy as np
import soundfile as sf
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mood → MusicGen prompt
# ---------------------------------------------------------------------------

MOOD_TO_MUSIC_PROMPT: Dict[str, str] = {
    "calm":      "soft piano melody, gentle, slow, peaceful, ambient, instrumental, children's lullaby",
    "happy":     "cheerful xylophone, upbeat gentle melody, playful, warm, orchestral, children's",
    "magical":   "harp and bells, twinkling, magical, sparkle, fairy tale, soft ambient, orchestral",
    "adventure": "light orchestral adventure, gentle strings, hopeful, children's storybook",
    "sleepy":    "very slow ambient, soft piano, lullaby, dreamy, calming, bedtime, fading",
}

# ---------------------------------------------------------------------------
# SFX map: key → filename inside assets/sfx/
# Users drop in their own WAVs; missing files are silently skipped.
# ---------------------------------------------------------------------------

SFX_MAP: Dict[str, str] = {
    "forest":  "forest_ambience.wav",
    "magic":   "sparkle.wav",
    "night":   "crickets.wav",
    "ocean":   "waves.wav",
    "wind":    "wind.wav",
    "rain":    "rain.wav",
    "birds":   "birds.wav",
    "cozy":    "fireplace.wav",
    "stars":   "stars_chime.wav",
    "sparkle": "sparkle.wav",
}

# ---------------------------------------------------------------------------
# Short MusicGen prompts
# Shorter = fewer tokens = less likely to produce NaN/inf on float32 CUDA.
# ---------------------------------------------------------------------------

_SHORT_MUSIC_PROMPTS: Dict[str, str] = {
    "calm":      "soft piano, gentle, slow, peaceful",
    "happy":     "cheerful piano, playful, warm, upbeat",
    "magical":   "harp and bells, twinkling, fairy tale",
    "adventure": "gentle strings, hopeful, orchestral",
    "sleepy":    "slow piano, lullaby, dreamy, soft",
}


def _audio_is_valid(path: str) -> bool:
    """
    Post-write audio sanity check. Note: soundfile clips float32 to ±1.0 on write,
    so amplitude checks are not useful here. The real defense against bad MusicGen
    output (NaN/inf/huge values) is the numpy array check inside _generate_chunk.

    This function just checks that the file is non-empty and readable.
    """
    try:
        data, _ = sf.read(path)
        if data is None or len(data) == 0:
            return False
        return True
    except Exception:
        return False


class SoundEngine:
    """
    Generates ambient music via MusicGen and mixes per-page audio tracks.

    Reuses MusicVideoAudioGenerator and audio_utils from the original pipeline.
    """

    def __init__(
        self,
        device: str = "cuda",
        sfx_dir: str = "storybook_generator/assets/sfx",
        narration_volume: float = 1.0,
        music_volume: float = 0.35,
        sfx_volume: float = 0.25,
        target_sr: int = 44100,
    ):
        self.device = device
        self.sfx_dir = sfx_dir
        self.narration_volume = narration_volume
        self.music_volume = music_volume
        self.sfx_volume = sfx_volume
        self.target_sr = target_sr

        from storybook_generator.pipeline.music_generator import MusicVideoAudioGenerator
        from storybook_generator.utils.audio_utils import mix_tracks, load_audio, pad_or_trim, normalize_audio

        self._audio_gen = MusicVideoAudioGenerator(device=device)
        self._mix_tracks = mix_tracks
        self._load_audio = load_audio
        self._pad_or_trim = pad_or_trim
        self._normalize = normalize_audio

    # ── Ambient music generation ─────────────────────────────────────────────

    def generate_ambient_music(
        self,
        music_prompt: str,
        total_duration_s: float,
        output_path: str,
        dominant_mood: str = "calm",
    ) -> str:
        """
        Generate a loopable ambient music track using MusicGen.

        If music_prompt is not set, falls back to the mood-based default.
        Uses a simpler, shorter prompt and lower guidance_scale to reduce
        NaN/inf probability tensor errors on CUDA float32.
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Use short, simple prompts — long prompts increase NaN probability with MusicGen float32
        prompt = _SHORT_MUSIC_PROMPTS.get(dominant_mood, _SHORT_MUSIC_PROMPTS["calm"])
        logger.info(f"Generating ambient music ({total_duration_s:.0f}s, mood={dominant_mood}): {prompt}")

        generated_ok = False
        try:
            self._audio_gen.generate_instrumental(
                prompt=prompt,
                duration_s=total_duration_s,
                output_path=output_path,
                guidance_scale=1.5,    # lower = fewer NaN issues on float32
                temperature=1.0,
            )
            # Validate the output — MusicGen can write NaN/inf as static even after greedy fallback
            generated_ok = _audio_is_valid(output_path)
            if not generated_ok:
                print(f"[SoundEngine] MusicGen produced invalid audio (NaN/inf/clipped). Replacing with silence.")
        except Exception as e:
            print(f"[SoundEngine] MusicGen failed: {e}. Writing silence.")

        if not generated_ok:
            # Write smooth silence so mixing downstream doesn't produce static
            sr = 32000
            silent = np.zeros(int(total_duration_s * sr), dtype=np.float32)
            sf.write(output_path, silent, samplerate=sr)

        return output_path

    # ── Per-page mixing ──────────────────────────────────────────────────────

    def mix_page_audio(
        self,
        narration_path: str,
        ambient_path: str,
        sound_effects: List[str],
        duration_s: float,
        output_path: str,
    ) -> str:
        """
        Mix narration + ambient music + SFX into a single WAV for one page.

        Narration is the primary track; music and SFX are ducked under it.
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        sr = self.target_sr
        target_samples = int(duration_s * sr)

        # 1. Load narration
        try:
            narration, _ = self._load_audio(narration_path, sr=sr)
        except Exception as e:
            logger.warning(f"Could not load narration {narration_path}: {e}")
            narration = np.zeros(target_samples, dtype=np.float32)

        narration = self._pad_or_trim(narration, target_samples) * self.narration_volume

        # 2. Load ambient music (clip to page duration)
        try:
            music, music_sr = self._load_audio(ambient_path, sr=sr)
            # Find a good starting position (avoid always starting at 0)
            import hashlib
            start_offset = int(
                (int(hashlib.md5(output_path.encode()).hexdigest(), 16) % 30) * sr
            )
            if start_offset + target_samples < len(music):
                music = music[start_offset:start_offset + target_samples]
            music = self._pad_or_trim(music, target_samples) * self.music_volume
        except Exception as e:
            logger.warning(f"Could not load ambient music: {e}")
            music = np.zeros(target_samples, dtype=np.float32)

        # 3. Load and mix SFX
        sfx_mix = np.zeros(target_samples, dtype=np.float32)
        for sfx_key in sound_effects:
            sfx_filename = SFX_MAP.get(sfx_key)
            if not sfx_filename:
                continue
            sfx_path = os.path.join(self.sfx_dir, sfx_filename)
            if not os.path.exists(sfx_path):
                logger.debug(f"SFX file not found (skipped): {sfx_path}")
                continue
            try:
                sfx_audio, _ = self._load_audio(sfx_path, sr=sr)
                # Loop SFX to fill page duration
                reps = (target_samples // len(sfx_audio)) + 1
                sfx_looped = np.tile(sfx_audio, reps)[:target_samples]
                sfx_mix += sfx_looped * self.sfx_volume
            except Exception as e:
                logger.warning(f"SFX '{sfx_key}' load failed: {e}")

        # 4. Final mix: narration + music + sfx
        mixed = narration + music + sfx_mix
        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.95

        sf.write(output_path, mixed, samplerate=sr)
        logger.info(f"Page audio mix saved: {output_path}")
        return output_path

    def mix_all_pages(
        self,
        pages,                        # List[StoryPage]
        narration_paths: List[str],
        ambient_music_path: str,
        output_dir: str,
    ) -> List[str]:
        """
        Mix audio for all pages.  Returns list of mixed WAV paths (one per page).
        """
        os.makedirs(output_dir, exist_ok=True)
        mixed_paths = []

        for page, narr_path in zip(pages, narration_paths):
            out = os.path.join(output_dir, f"mixed_page_{page.page_number:02d}.wav")
            self.mix_page_audio(
                narration_path=narr_path,
                ambient_path=ambient_music_path,
                sound_effects=page.sound_effects,
                duration_s=page.duration_s if page.duration_s > 0 else 5.0,
                output_path=out,
            )
            mixed_paths.append(out)

        return mixed_paths
