"""
narration_generator.py
~~~~~~~~~~~~~~~~~~~~~~
Per-page narration audio using Chatterbox TTS.

This is a thin wrapper around MusicVideoAudioGenerator.generate_vocals_chatterbox()
from the original pipeline — zero TTS code is reimplemented here.

Key changes vs. music video vocals:
  - Called once per page (not once per whole song)
  - Text prefixed with slow/warm narrator instruction
  - Returns per-page WAV paths + measured durations
"""

import os
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class NarrationGenerator:
    """
    Generates spoken narration for each storybook page using Chatterbox TTS.

    Delegates entirely to the existing MusicVideoAudioGenerator — no TTS code
    is duplicated here.
    """


    def __init__(self, device: str = "cuda", voice_reference_path: Optional[str] = None):
        self.device = device
        self.voice_reference_path = voice_reference_path

        # Import the bundled audio generator from storybook_generator/pipeline/
        from storybook_generator.pipeline.music_generator import MusicVideoAudioGenerator

        self._audio_gen = MusicVideoAudioGenerator(device=device)
        logger.info(f"NarrationGenerator ready (device={device}, chatterbox={self._audio_gen._chatterbox_available})")

    def generate_audio(
        self,
        pages,                       # List[StoryPage]
        output_dir: str,
    ) -> Tuple[List[str], List[float]]:
        """
        Generate narration WAV for each page.

        Args:
            pages: list of StoryPage objects (must have .text set)
            output_dir: directory to save WAV files

        Returns:
            (narration_paths, durations_s)
            Both lists have one entry per page, in order.
        """
        os.makedirs(output_dir, exist_ok=True)

        narration_paths: List[str] = []
        durations: List[float] = []

        for page in pages:
            out_path = os.path.join(output_dir, f"narration_page_{page.page_number:02d}.wav")

            try:
                self._audio_gen.generate_vocals_chatterbox(
                    lyrics=page.text,
                    vocal_style="",  # MUST be empty: music_generator.py prepends "[Singing in {style} style]" when non-empty
                    output_path=out_path,
                    voice_reference_path=self.voice_reference_path,
                )
                duration = self._measure_duration(out_path)
                logger.info(f"Page {page.page_number} narration: {duration:.1f}s → {out_path}")
            except Exception as e:
                logger.error(f"Narration failed for page {page.page_number}: {e}")
                # Write 4-second silence fallback
                self._write_silence(out_path, duration_s=4.0)
                duration = 4.0

            narration_paths.append(out_path)
            durations.append(duration)

        return narration_paths, durations

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _measure_duration(wav_path: str) -> float:
        """Return duration of a WAV file in seconds."""
        try:
            import soundfile as sf
            info = sf.info(wav_path)
            return info.duration
        except Exception:
            try:
                import librosa
                y, sr = librosa.load(wav_path, sr=None)
                return len(y) / sr
            except Exception:
                return 4.0  # Safe default

    @staticmethod
    def _write_silence(path: str, duration_s: float = 4.0, sr: int = 44100):
        """Write a silent WAV file as fallback."""
        import numpy as np
        import soundfile as sf
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        silent = np.zeros(int(duration_s * sr), dtype=np.float32)
        sf.write(path, silent, samplerate=sr)
