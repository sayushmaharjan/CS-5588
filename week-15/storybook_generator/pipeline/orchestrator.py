"""
orchestrator.py
~~~~~~~~~~~~~~~
StorybookOrchestrator — the main entry point for the AI Children's Story Creator.

Mirrors the structure of MusicVideoOrchestrator exactly:
  - Same constructor signature pattern
  - Same progress_cb callback pattern
  - Same fallback-at-every-step robustness
  - Same return dict structure

Steps:
  1. Generate story   (StoryGenerator  ← replaces LyricsProcessor)
  2. Generate images  (StorybookIllustrator ← replaces BackgroundGenerator)
  3. Generate speech  (NarrationGenerator ← wraps Chatterbox TTS)
  4. Generate music   (SoundEngine ← wraps MusicGen)
  5. Mix audio        (SoundEngine.mix_all_pages ← wraps audio_utils)
  6. Render PDF       (PDFRenderer ← Pillow-based)
  7. Render video     (VideoRenderer ← reuses MoviePy compositor)
  8. Generate SRT     (subtitle_generator ← adapted from original)
"""

import os
import logging
from typing import Optional, Callable, Dict, Any

from storybook_generator.utils.story_schemas import Storybook, GeneratedStoryAssets
from storybook_generator.pipeline.story_generator   import StoryGenerator
from storybook_generator.pipeline.illustrator       import StorybookIllustrator
from storybook_generator.pipeline.narration_generator import NarrationGenerator
from storybook_generator.pipeline.sound_engine      import SoundEngine, MOOD_TO_MUSIC_PROMPT
from storybook_generator.pipeline.pdf_renderer      import PDFRenderer
from storybook_generator.pipeline.video_renderer    import VideoRenderer
from storybook_generator.pipeline.subtitle_generator import generate_storybook_srt

logger = logging.getLogger(__name__)


class StorybookOrchestrator:
    """
    End-to-end storybook generation pipeline.

    Usage:
        orch = StorybookOrchestrator(api_key="...", device="mps")
        result = orch.generate_storybook(
            child_name="Liam",
            age=5,
            theme="dinosaurs and friendship",
            num_pages=4,
        )
        print(result["pdf_path"], result["video_path"])
    """

    def __init__(
        self,
        api_key: str = "",
        device: str = "cuda",
        output_dir: str = "outputs",
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        voice_reference_path: Optional[str] = None,
        sd_model: str = "runwayml/stable-diffusion-v1-5",
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Resolve device
        try:
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = device

        # ── Initialise pipeline components ───────────────────────────────────
        self.story_gen = StoryGenerator(
            api_key=api_key,
            base_url=llm_base_url,
            model=llm_model or "meta-llama/Llama-3.1-8B-Instruct",
        )

        self.illustrator = StorybookIllustrator(
            device=self.device,
            sd_model=sd_model,
        )

        self.narrator = NarrationGenerator(
            device=self.device,
            voice_reference_path=voice_reference_path,
        )

        self.sound_engine = SoundEngine(
            device=self.device,
            sfx_dir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "assets", "sfx",
            ),
        )

        self.pdf_renderer   = PDFRenderer()
        self.video_renderer = VideoRenderer(
            output_dir=os.path.join(output_dir, "video")
        )

    # ── Main generate method ─────────────────────────────────────────────────

    def generate_storybook(
        self,
        child_name: str,
        age: int,
        theme: str,
        num_pages: int = 4,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Full end-to-end storybook generation.

        Args:
            child_name: The child's name (woven into story + illustrations)
            age: Child's age (used to calibrate vocabulary)
            theme: Short story idea ("dinosaurs and friendship")
            num_pages: Number of story pages (2–8)
            progress_cb: Optional callback(message: str, percent: float)

        Returns dict with keys:
            pdf_path, video_path, subtitle_path, storybook, assets
        """

        def update(msg: str, pct: float):
            print(f"[{pct:5.1f}%] {msg}")
            if progress_cb:
                progress_cb(msg, pct)

        assets = GeneratedStoryAssets()

        # ── STEP 1: Generate story ──────────────────────────────────────────
        update("Generating story with LLM…", 5.0)
        try:
            storybook: Storybook = self.story_gen.generate(
                child_name=child_name,
                age=age,
                theme=theme,
                num_pages=num_pages,
            )
        except Exception as e:
            raise RuntimeError(f"Story generation failed: {e}") from e

        update(f"Story generated: '{storybook.title}' ({len(storybook.pages)} pages)", 12.0)

        # ── STEP 2: Generate illustrations ─────────────────────────────────
        update("Generating illustrations…", 15.0)
        images_dir = os.path.join(self.output_dir, "images")
        seed = StorybookIllustrator.seed_from_title(storybook.title)
        try:
            image_paths = self.illustrator.generate_pages(
                pages=storybook.pages,
                output_dir=images_dir,
                story_seed=seed,
            )
            for page, path in zip(storybook.pages, image_paths):
                page.image_path = path
                assets.illustration_paths[page.page_number] = path
        except Exception as e:
            logger.error(f"Illustration generation failed: {e}")
            for page in storybook.pages:
                page.image_path = ""

        update("Illustrations ready.", 40.0)

        # ── STEP 3: Generate narration audio ────────────────────────────────
        update("Generating narration with Chatterbox TTS…", 42.0)
        narration_dir = os.path.join(self.output_dir, "audio", "narration")
        try:
            narration_paths, durations = self.narrator.generate_audio(
                pages=storybook.pages,
                output_dir=narration_dir,
            )
            for page, narr_path, dur in zip(storybook.pages, narration_paths, durations):
                page.narration_path = narr_path
                page.duration_s     = dur
                assets.narration_paths[page.page_number] = narr_path
        except Exception as e:
            logger.error(f"Narration generation failed: {e}")
            for page in storybook.pages:
                page.duration_s = 5.0  # default

        storybook.total_duration_s = sum(p.duration_s for p in storybook.pages)
        update(f"Narration done. Total: {storybook.total_duration_s:.0f}s", 60.0)

        # ── STEP 3.5: Whisper Word Timestamps ───────────────────────────────
        update("Aligning subtitles with Whisper…", 62.0)
        try:
            import whisper
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_model = whisper.load_model("tiny")
            
            for page in storybook.pages:
                if page.narration_path and os.path.exists(page.narration_path):
                    res = w_model.transcribe(page.narration_path, word_timestamps=True)
                    words = []
                    for seg in res.get("segments", []):
                        for w in seg.get("words", []):
                            words.append({
                                "word": w["word"].strip(),
                                "start": w["start"],
                                "end": w["end"],
                            })
                    page.word_timestamps = words
        except Exception as e:
            logger.warning(f"Whisper alignment failed: {e}")

        # ── STEP 4: Generate ambient music ──────────────────────────────────
        update("Generating ambient background music with MusicGen…", 63.0)
        ambient_path = os.path.join(self.output_dir, "audio", "ambient_music.wav")
        os.makedirs(os.path.dirname(ambient_path), exist_ok=True)

        # Dominant mood = most common across pages
        moods = [p.mood for p in storybook.pages]
        dominant_mood = max(set(moods), key=moods.count)

        try:
            self.sound_engine.generate_ambient_music(
                music_prompt=storybook.music_prompt,
                total_duration_s=storybook.total_duration_s + 10,   # extra buffer
                output_path=ambient_path,
                dominant_mood=dominant_mood,
            )
            assets.ambient_music_path = ambient_path
        except Exception as e:
            logger.error(f"Music generation failed: {e}")
            ambient_path = ""

        # ── STEP 5: Mix per-page audio ──────────────────────────────────────
        update("Mixing narration + music + SFX per page…", 72.0)
        mix_dir = os.path.join(self.output_dir, "audio", "mixed")
        try:
            narration_paths_list = [p.narration_path for p in storybook.pages]
            mixed_audio_paths = self.sound_engine.mix_all_pages(
                pages=storybook.pages,
                narration_paths=narration_paths_list,
                ambient_music_path=ambient_path if ambient_path else "",
                output_dir=mix_dir,
            )
            for page, mix_path in zip(storybook.pages, mixed_audio_paths):
                page.mixed_audio_path = mix_path
                assets.mixed_audio_paths[page.page_number] = mix_path
        except Exception as e:
            logger.error(f"Audio mixing failed: {e}")
            mixed_audio_paths = [p.narration_path for p in storybook.pages]

        # ── STEP 6: Render PDF ──────────────────────────────────────────────
        update("Rendering storybook PDF…", 78.0)
        pdf_path = os.path.join(self.output_dir, "pdf", f"{_safe_filename(storybook.title)}.pdf")
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        try:
            self.pdf_renderer.render(storybook=storybook, output_path=pdf_path)
            assets.output_pdf_path = pdf_path
        except Exception as e:
            logger.error(f"PDF rendering failed: {e}")
            pdf_path = ""

        # ── STEP 7: Render video ────────────────────────────────────────────
        update("Rendering storybook video…", 82.0)
        video_path = os.path.join(self.output_dir, "video", f"{_safe_filename(storybook.title)}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        try:
            self.video_renderer.render(
                storybook=storybook,
                mixed_audio_paths=mixed_audio_paths,
                output_path=video_path,
            )
            assets.output_video_path = video_path
        except Exception as e:
            logger.error(f"Video rendering failed: {e}")
            video_path = ""

        # ── STEP 8: Generate subtitles ──────────────────────────────────────
        update("Generating SRT subtitles…", 96.0)
        srt_path = os.path.join(self.output_dir, "video", f"{_safe_filename(storybook.title)}.srt")
        try:
            generate_storybook_srt(pages=storybook.pages, output_path=srt_path)
            assets.subtitle_path = srt_path
        except Exception as e:
            logger.error(f"Subtitle generation failed: {e}")
            srt_path = ""

        update("✅ Storybook complete!", 100.0)

        return {
            "pdf_path":     pdf_path,
            "video_path":   video_path,
            "subtitle_path": srt_path,
            "ambient_path": ambient_path,
            "storybook":    storybook,
            "assets":       assets,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_filename(title: str, max_len: int = 40) -> str:
    """Convert a story title to a filesystem-safe filename."""
    import re
    safe = re.sub(r"[^\w\s-]", "", title).strip().lower()
    safe = re.sub(r"[\s_-]+", "_", safe)
    return safe[:max_len] or "storybook"
