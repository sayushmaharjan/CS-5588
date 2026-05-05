import os
import torch
import numpy as np
import soundfile as sf
import warnings
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── Module-level Chatterbox cache (same pattern as week-14) ──────────────────
_chatterbox_model = None
_chatterbox_device = None


def _get_chatterbox_model():
    """Lazy-load and cache the Chatterbox TTS model."""
    global _chatterbox_model, _chatterbox_device
    if _chatterbox_model is not None:
        return _chatterbox_model

    from chatterbox.tts import ChatterboxTTS

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    _chatterbox_device = device
    logger.info(f"Loading Chatterbox TTS on {device}…")

    # Apple Silicon MPS compatibility patch
    if device == "mps":
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device(device))
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load

    _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    logger.info("Chatterbox TTS loaded.")
    return _chatterbox_model


class MusicVideoAudioGenerator:
    def __init__(self, device: str = "cuda"):
        # MusicGen has numerical instability on MPS causing NaN/inf and static audio.
        # Force CPU on Apple Silicon.
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.musicgen_model = None
        self.musicgen_processor = None
        self._chatterbox_available = False

        # Check Chatterbox availability (soft check — lazy load later)
        try:
            import chatterbox  # noqa
            self._chatterbox_available = True
            logger.info("Chatterbox TTS detected.")
        except ImportError:
            warnings.warn(
                "chatterbox-tts not installed. "
                "Install with: pip install chatterbox-tts  "
                "Vocals will fall back to gTTS/silence."
            )

    def load_musicgen(self):
        """Load MusicGen for instrumental generation."""
        if self.musicgen_model is not None:
            return

        print("Loading MusicGen-small...")
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            self.musicgen_processor = AutoProcessor.from_pretrained(
                "facebook/musicgen-small",
                torch_dtype=torch.float32
            )
            self.musicgen_model = MusicgenForConditionalGeneration.from_pretrained(
                "facebook/musicgen-small",
                torch_dtype=torch.float32
            ).to(self.device)
            print("MusicGen loaded successfully.")
        except Exception as e:
            print(f"Error loading MusicGen: {e}")
            raise

    def generate_instrumental(
        self,
        prompt: str,
        duration_s: float,
        output_path: str,
        guidance_scale: float = 3.0,
        temperature: float = 1.0
    ) -> str:
        """Generate instrumental music track using MusicGen."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if self.musicgen_model is None:
            self.load_musicgen()

        # MusicGen can do max ~30s per generation
        if duration_s <= 30:
            audio = self._generate_chunk(prompt, duration_s, guidance_scale, temperature)
        else:
            audio = self._generate_long_audio(prompt, duration_s, guidance_scale, temperature)

        # MusicGen outputs at 32kHz
        sf.write(output_path, audio, samplerate=32000)
        print(f"Instrumental saved: {output_path}")
        return output_path

    def _generate_chunk(
        self,
        prompt: str,
        duration_s: float,
        guidance_scale: float = 3.0,
        temperature: float = 1.0
    ) -> np.ndarray:
        """Generate a single chunk of audio."""
        inputs = self.musicgen_processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # MusicGen generates ~50 tokens per second
        max_tokens = int(duration_s * 50)

        # Classifier-free guidance (guidance_scale > 1) causes NaN on CPU;
        # disable it automatically on non-GPU devices.
        safe_guidance = guidance_scale if self.device in ("cuda", "mps") else 1.0

        generate_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=True,
            guidance_scale=safe_guidance,
            temperature=temperature,
        )

        try:
            with torch.no_grad():
                audio_values = self.musicgen_model.generate(**inputs, **generate_kwargs)
        except RuntimeError as e:
            if "inf" in str(e) or "nan" in str(e) or "element < 0" in str(e):
                print(f"MusicGen sampling error ({e}); retrying with greedy decode.")
                generate_kwargs["do_sample"] = False
                generate_kwargs.pop("temperature", None)
                generate_kwargs["guidance_scale"] = 1.0
                with torch.no_grad():
                    audio_values = self.musicgen_model.generate(**inputs, **generate_kwargs)
            else:
                raise

        audio = audio_values[0, 0].cpu().numpy().astype(np.float32)

        # Validate — greedy decode can still produce NaN/inf/huge values on CUDA float32
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)) or np.max(np.abs(audio)) > 2.0:
            print(f"MusicGen generated invalid audio array (max={np.max(np.abs(audio)):.2f}); substituting silence.")
            audio = np.zeros(int(duration_s * 32000), dtype=np.float32)

        # Trim to exact duration
        target_samples = int(duration_s * 32000)
        if len(audio) > target_samples:
            audio = audio[:target_samples]

        return audio

    def _generate_long_audio(
        self,
        prompt: str,
        duration_s: float,
        guidance_scale: float = 3.0,
        temperature: float = 1.0
    ) -> np.ndarray:
        """Generate audio longer than 30s by concatenating chunks with crossfade."""
        chunk_duration = 14.0  # Safe chunk size to avoid VRAM OOM
        chunks = []
        remaining = duration_s

        # First chunk with full prompt
        first_chunk_dur = min(chunk_duration, remaining)
        chunk = self._generate_chunk(prompt, first_chunk_dur, guidance_scale, temperature)
        chunks.append(chunk)
        remaining -= first_chunk_dur

        # Subsequent chunks with continuation prompt
        while remaining > 0:
            continuation_prompt = f"{prompt}, continuing seamlessly, same style and tempo"
            this_chunk_dur = min(chunk_duration, remaining)
            next_chunk = self._generate_chunk(continuation_prompt, this_chunk_dur, guidance_scale, temperature)

            # Crossfade last 0.5s
            fade_s = 0.5
            fade_samples = int(fade_s * 32000)

            if len(chunks[-1]) > fade_samples and len(next_chunk) > fade_samples:
                fade_out = np.linspace(1.0, 0.0, fade_samples)
                fade_in = np.linspace(0.0, 1.0, fade_samples)

                prev_tail = chunks[-1][-fade_samples:] * fade_out
                next_head = next_chunk[:fade_samples] * fade_in
                overlap = prev_tail + next_head

                chunks[-1] = chunks[-1][:-fade_samples]
                next_chunk = np.concatenate([overlap, next_chunk[fade_samples:]])

            chunks.append(next_chunk)
            remaining -= this_chunk_dur

        return np.concatenate(chunks)

    # ── Vocals: Chatterbox TTS ────────────────────────────────────────────────
    def generate_vocals_chatterbox(
        self,
        lyrics: str,
        vocal_style: str,
        output_path: str,
        voice_reference_path: Optional[str] = None,
    ) -> str:
        """
        Generate vocals using Chatterbox TTS (Resemble AI).
        Supports voice cloning via an optional reference audio clip.
        Falls back to gTTS → silence if Chatterbox is unavailable.
        """
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True
        )

        if not self._chatterbox_available:
            logger.warning("Chatterbox unavailable — using gTTS fallback.")
            return self._generate_fallback_vocals(lyrics, output_path)

        try:
            import torchaudio as ta

            model = _get_chatterbox_model()
            clean_lyrics = re.sub(r'\[.*?\]', '', lyrics).strip()

            # Build a short style prefix so the model reads with musical intent
            if vocal_style:
                text_input = f"[Singing in {vocal_style} style] {clean_lyrics}"
            else:
                text_input = clean_lyrics

            # Chunk long lyrics so Chatterbox doesn't time-out
            chunks = self._split_lyrics_into_chunks(text_input, max_chars=300)
            audio_segments = []

            for chunk in chunks:
                if voice_reference_path and os.path.exists(voice_reference_path):
                    wav = model.generate(chunk, audio_prompt_path=voice_reference_path)
                else:
                    wav = model.generate(chunk)
                audio_segments.append(wav.squeeze(0).cpu())

            # Concatenate segments
            if audio_segments:
                combined = torch.cat(audio_segments, dim=-1).unsqueeze(0)
            else:
                combined = torch.zeros((1, 1, model.sr), dtype=torch.float32)

            ta.save(output_path, combined, model.sr)
            duration_s = combined.shape[-1] / model.sr
            logger.info(f"Chatterbox vocals saved ({duration_s:.1f}s): {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Chatterbox vocal generation failed: {e}. Falling back to gTTS.")
            return self._generate_fallback_vocals(lyrics, output_path)

    # Keep the old name as an alias so orchestrator.py doesn't break
    def generate_vocals_bark(
        self,
        lyrics: str,
        vocal_style: str,
        output_path: str,
        speaker: str = "en_speaker_6",
        voice_reference_path: Optional[str] = None,
    ) -> str:
        """Alias → now delegates to Chatterbox TTS."""
        return self.generate_vocals_chatterbox(
            lyrics=lyrics,
            vocal_style=vocal_style,
            output_path=output_path,
            voice_reference_path=voice_reference_path,
        )

    def _split_lyrics_into_chunks(self, text: str, max_chars: int = 300) -> list:
        """Split long lyrics into sentence-safe chunks for Chatterbox."""
        lines = text.split('\n')
        chunks, current = [], ""
        for line in lines:
            if len(current) + len(line) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = line
            else:
                current = (current + "\n" + line).strip()
        if current:
            chunks.append(current)
        return chunks if chunks else [text]

    def _generate_fallback_vocals(self, lyrics: str, output_path: str) -> str:
        """Fallback vocal generation: gTTS → silence."""
        try:
            from gtts import gTTS
            import tempfile

            clean_lyrics = re.sub(r'\[.*?\]', '', lyrics).strip()
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tts = gTTS(text=clean_lyrics, lang='en', slow=False)
                tts.save(tmp.name)
                import librosa
                y, sr = librosa.load(tmp.name, sr=44100)
                sf.write(output_path, y, samplerate=44100)
                os.unlink(tmp.name)
            logger.info(f"Fallback vocals (gTTS) saved: {output_path}")
            return output_path
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"gTTS failed: {e}")

        # Last resort: silence
        logger.warning("No TTS available. Writing silent audio.")
        silent = np.zeros(int(30.0 * 44100), dtype=np.float32)
        sf.write(output_path, silent, samplerate=44100)
        return output_path

    def mix_audio(
        self,
        instrumental_path: str,
        vocals_path: str,
        output_path: str,
        vocal_volume: float = 1.0,
        instrumental_volume: float = 0.6,
        target_sr: int = 44100
    ) -> str:
        """Mix vocals and instrumental together."""
        import librosa

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        instrumental, _ = librosa.load(instrumental_path, sr=target_sr, mono=True)
        vocals, _ = librosa.load(vocals_path, sr=target_sr, mono=True)

        target_length = max(len(instrumental), len(vocals))

        if len(instrumental) < target_length:
            instrumental = np.pad(instrumental, (0, target_length - len(instrumental)))
        else:
            instrumental = instrumental[:target_length]

        if len(vocals) < target_length:
            vocals = np.pad(vocals, (0, target_length - len(vocals)))
        else:
            vocals = vocals[:target_length]

        instrumental = instrumental * instrumental_volume
        vocals = vocals * vocal_volume

        mixed = instrumental + vocals

        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.95

        sf.write(output_path, mixed, samplerate=target_sr)
        print(f"Mixed audio saved: {output_path}")
        return output_path

    def unload_models(self):
        """Unload models to free GPU VRAM."""
        self.musicgen_model = None
        self.musicgen_processor = None
        global _chatterbox_model
        _chatterbox_model = None
