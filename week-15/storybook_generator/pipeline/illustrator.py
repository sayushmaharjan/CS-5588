"""
illustrator.py
~~~~~~~~~~~~~~
Replaces background_generator.py.

Generates per-page storybook illustrations using Stable Diffusion (local),
with a consistent style prefix and seed locking for character consistency.

Falls back to gradient+text images if SD is not available.
"""

import os
import logging
import hashlib
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from storybook_generator.utils.story_schemas import StoryPage

logger = logging.getLogger(__name__)

# Mood → background color palette (used in fallback gradient images)
MOOD_COLORS = {
    "calm":      [(135, 180, 255), (200, 230, 255)],   # soft blue
    "happy":     [(255, 220, 100), (255, 180, 80)],    # warm yellow
    "magical":   [(180, 120, 255), (230, 180, 255)],   # purple
    "adventure": [(100, 200, 150), (60, 160, 120)],    # green
    "sleepy":    [(100, 120, 180), (60, 80, 140)],     # deep blue
}


class StorybookIllustrator:
    """
    Generates one illustration per story page via Stable Diffusion.

    Reuses the same SD loading pattern as the original BackgroundGenerator
    but applies a fixed children's-book style prefix and consistent seed.
    """

    STYLE_PREFIX = (
        "children's book illustration, soft pastel colors, Pixar style, "
        "warm lighting, bedtime story style, kid-friendly, watercolor texture, "
    )
    # Generate at 512×512 (same as original BackgroundGenerator) then upscale
    WIDTH  = 512
    HEIGHT = 512
    OUT_WIDTH  = 768   # final saved size
    OUT_HEIGHT = 512

    def __init__(self, device: str = "cuda", sd_model: str = "runwayml/stable-diffusion-v1-5"):
        import torch
        # Mirror the device detection from music_generator.py
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.sd_model_name = sd_model
        self._pipe = None           # Lazy-loaded

    # ── SD loading (same lazy pattern as BackgroundGenerator) ───────────────

    def _load_sd(self):
        if self._pipe is not None:
            return

        print(f"Loading SD model '{self.sd_model_name}' on {self.device}…")
        try:
            import torch
            
            # Monkeypatch for diffusers < 0.31.0 compatibility with transformers >= 4.45.0
            import transformers.utils
            if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
                transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
                
            from diffusers import StableDiffusionPipeline

            # Mirror original BackgroundGenerator: float16 on CUDA, float32 everywhere else
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.sd_model_name,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(self.device)

            try:
                self._pipe.enable_attention_slicing()
            except Exception:
                pass

            print(f"SD pipeline ready on {self.device}.")
        except Exception as e:
            print(f"[Illustrator] Could not load SD ({type(e).__name__}: {e}). "
                  f"Using gradient fallback for illustrations.")
            self._pipe = None

    # ── Illustration generation ──────────────────────────────────────────────

    def generate_pages(
        self,
        pages: List[StoryPage],
        output_dir: str,
        story_seed: Optional[int] = None,
    ) -> List[str]:
        """
        Generate one illustration per page.

        Args:
            pages: list of StoryPage objects with illustration_prompt set
            output_dir: directory to save PNGs into
            story_seed: base seed for consistent character look across pages.
                        If None, derived from story title hash.

        Returns list of image file paths (one per page).
        """
        os.makedirs(output_dir, exist_ok=True)

        # Attempt to load SD; if it fails we'll use fallback per page
        self._load_sd()

        paths = []
        for page in pages:
            path = os.path.join(output_dir, f"page_{page.page_number:02d}.png")
            try:
                if self._pipe is not None:
                    img = self._generate_with_sd(page, story_seed)
                else:
                    img = self._gradient_fallback(page)
                img.save(path)
                logger.info(f"Illustration saved: {path}")
            except Exception as e:
                logger.warning(f"Illustration failed for page {page.page_number}: {e}. Using fallback.")
                img = self._gradient_fallback(page)
                img.save(path)

            paths.append(path)

        return paths

    def _generate_with_sd(self, page: StoryPage, story_seed: Optional[int]) -> Image.Image:
        """Generate illustration using Stable Diffusion."""
        import torch

        full_prompt = self.STYLE_PREFIX + page.illustration_prompt
        negative_prompt = (
            "ugly, scary, dark, violent, adult content, realistic photo, "
            "blurry, bad anatomy, deformed, watermark, signature"
        )

        # Per-page seed: base + page_number so each page is unique but deterministic
        seed = (story_seed or 42) + page.page_number
        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self._pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=self.WIDTH,
            height=self.HEIGHT,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        )
        img = result.images[0]
        # Upscale to output dimensions (same pattern as original BackgroundGenerator)
        return img.resize((self.OUT_WIDTH, self.OUT_HEIGHT), Image.LANCZOS)

    def _gradient_fallback(self, page: StoryPage) -> Image.Image:
        """
        Create a simple gradient image with page text overlay.
        Used when SD is unavailable (CPU-only environments).
        """
        colors = MOOD_COLORS.get(page.mood, [(150, 180, 220), (200, 220, 255)])
        img = Image.new("RGB", (self.WIDTH, self.HEIGHT))

        # Vertical gradient
        for y in range(self.HEIGHT):
            t = y / self.HEIGHT
            r = int(colors[0][0] * (1 - t) + colors[1][0] * t)
            g = int(colors[0][1] * (1 - t) + colors[1][1] * t)
            b = int(colors[0][2] * (1 - t) + colors[1][2] * t)
            for x in range(self.WIDTH):
                img.putpixel((x, y), (r, g, b))

        # Draw page text as placeholder illustration caption
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except Exception:
            font = ImageFont.load_default()
            small_font = font

        # Semi-transparent overlay
        overlay = Image.new("RGBA", (self.WIDTH, 120), (0, 0, 0, 100))
        img_rgba = img.convert("RGBA")
        img_rgba.paste(overlay, (0, self.HEIGHT // 2 - 60), overlay)
        img = img_rgba.convert("RGB")
        draw = ImageDraw.Draw(img)

        # Page number
        draw.text((20, 20), f"Page {page.page_number}", fill=(255, 255, 255, 200), font=small_font)

        # Mood tag
        mood_label = f"✨ {page.mood.title()}"
        draw.text((self.WIDTH - 150, 20), mood_label, fill=(255, 255, 255, 180), font=small_font)

        # Short caption from prompt
        caption = page.illustration_prompt[:80] + ("…" if len(page.illustration_prompt) > 80 else "")
        draw.text(
            (self.WIDTH // 2, self.HEIGHT // 2 - 10),
            caption,
            fill=(255, 255, 255),
            font=small_font,
            anchor="mm",
        )

        return img

    @staticmethod
    def seed_from_title(title: str) -> int:
        """Derive a deterministic seed from a story title for character consistency."""
        return int(hashlib.md5(title.encode()).hexdigest(), 16) % (2 ** 32)
