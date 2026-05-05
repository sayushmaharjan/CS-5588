"""
pdf_renderer.py
~~~~~~~~~~~~~~~
Generates a multi-page PDF storybook using Pillow only (no reportlab).

Layout per page:
  ┌───────────────────────────────────────────┐
  │              Illustration (top 65%)        │
  ├───────────────────────────────────────────┤
  │         Story text (bottom 35%)           │
  │                 Page N                    │
  └───────────────────────────────────────────┘

The cover page contains the title and a decorative background.
Pillow's save(..., save_all=True, append_images=[...]) writes a real PDF.
"""

import os
import logging
import textwrap
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

from storybook_generator.utils.story_schemas import Storybook, StoryPage

logger = logging.getLogger(__name__)

# PDF page size (A4-ish in pixels at 96 DPI)
PAGE_W = 794
PAGE_H = 1123

ILLUSTRATION_H = int(PAGE_H * 0.60)   # top 60%
TEXT_AREA_H    = PAGE_H - ILLUSTRATION_H

MARGIN = 40
TEXT_COLOR    = (30, 30, 50)
BG_COLOR      = (255, 252, 245)        # warm off-white
TITLE_COLOR   = (80, 60, 130)          # purple
ACCENT_COLOR  = (220, 180, 100)        # golden


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Try to load a nice system font; fall back to default."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ]
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ] + candidates

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


class PDFRenderer:
    """
    Renders a Storybook to a multi-page Pillow PDF.
    """

    def render(
        self,
        storybook: Storybook,
        output_path: str,
    ) -> str:
        """
        Generate the PDF file.

        Args:
            storybook: Storybook with pages that have image_path and text set
            output_path: destination path (e.g. outputs/pdf/storybook.pdf)

        Returns:
            output_path
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        pages_pil: List[Image.Image] = []

        # Cover page
        cover = self._make_cover(storybook)
        pages_pil.append(cover)

        # Story pages
        for page in storybook.pages:
            pil_page = self._make_story_page(page, storybook)
            pages_pil.append(pil_page)

        # Save as PDF (Pillow multi-page PDF)
        if not pages_pil:
            logger.warning("No pages to render!")
            return output_path

        first = pages_pil[0].convert("RGB")
        rest  = [p.convert("RGB") for p in pages_pil[1:]]

        first.save(
            output_path,
            format="PDF",
            save_all=True,
            append_images=rest,
            resolution=96,
        )
        logger.info(f"PDF saved ({len(pages_pil)} pages): {output_path}")
        return output_path

    # ── Cover page ───────────────────────────────────────────────────────────

    def _make_cover(self, storybook: Storybook) -> Image.Image:
        img = Image.new("RGB", (PAGE_W, PAGE_H), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Decorative gradient strip at top
        for y in range(200):
            t = y / 200
            r = int(180 * (1 - t) + 245 * t)
            g = int(140 * (1 - t) + 240 * t)
            b = int(230 * (1 - t) + 245 * t)
            draw.line([(0, y), (PAGE_W, y)], fill=(r, g, b))

        # Stars decoration
        import random
        rng = random.Random(42)
        for _ in range(30):
            x = rng.randint(20, PAGE_W - 20)
            y = rng.randint(10, 190)
            r = rng.randint(2, 5)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 230, 100))

        # Try to use first page illustration as cover art
        cover_art_y = 210
        cover_art_h = int(PAGE_H * 0.40)
        first_page_with_img = next((p for p in storybook.pages if p.image_path and os.path.exists(p.image_path)), None)
        if first_page_with_img:
            try:
                art = Image.open(first_page_with_img.image_path).convert("RGB")
                art = art.resize((PAGE_W - 2 * MARGIN, cover_art_h), Image.LANCZOS)
                img.paste(art, (MARGIN, cover_art_y))
            except Exception as e:
                logger.debug(f"Cover art load failed: {e}")

        # Title
        title_font = _load_font(52, bold=True)
        subtitle_font = _load_font(28)
        by_font = _load_font(22)

        text_y = cover_art_y + cover_art_h + 40
        self._draw_centered_text(draw, storybook.title, title_font, TITLE_COLOR, text_y)

        text_y += 80
        subtitle = f"A bedtime story for {storybook.child_name}"
        self._draw_centered_text(draw, subtitle, subtitle_font, (120, 90, 170), text_y)

        text_y += 50
        by_line = f"Created with AI Children's Story Creator"
        self._draw_centered_text(draw, by_line, by_font, (160, 160, 160), text_y)

        # Bottom decorative strip
        for y in range(PAGE_H - 60, PAGE_H):
            t = (y - (PAGE_H - 60)) / 60
            r = int(180 * (1 - t) + 220 * t)
            g = int(140 * (1 - t) + 180 * t)
            b = int(230 * (1 - t) + 255 * t)
            draw.line([(0, y), (PAGE_W, y)], fill=(r, g, b))

        return img

    # ── Story page ───────────────────────────────────────────────────────────

    def _make_story_page(self, page: StoryPage, storybook: Storybook) -> Image.Image:
        img = Image.new("RGB", (PAGE_W, PAGE_H), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # ── Illustration (top 60%) ────────────────────────────────────────
        if page.image_path and os.path.exists(page.image_path):
            try:
                art = Image.open(page.image_path).convert("RGB")
                art = art.resize((PAGE_W, ILLUSTRATION_H), Image.LANCZOS)
                img.paste(art, (0, 0))
            except Exception as e:
                logger.debug(f"Page {page.page_number} image load failed: {e}")
                self._draw_placeholder_art(img, draw, page)
        else:
            self._draw_placeholder_art(img, draw, page)

        # Separator line
        draw.line([(MARGIN, ILLUSTRATION_H + 8), (PAGE_W - MARGIN, ILLUSTRATION_H + 8)],
                  fill=ACCENT_COLOR, width=2)

        # ── Story text (bottom 40%) ───────────────────────────────────────
        text_font  = _load_font(26)
        page_font  = _load_font(18)
        title_font = _load_font(16)

        text_y = ILLUSTRATION_H + 24

        # Wrap text to fit page width
        max_chars_per_line = 62
        wrapped = textwrap.fill(page.text, width=max_chars_per_line)
        lines   = wrapped.split("\n")

        for line in lines:
            if text_y + 34 > PAGE_H - 50:  # Prevent overflow
                break
            draw.text((MARGIN, text_y), line, fill=TEXT_COLOR, font=text_font)
            text_y += 34

        # Mood emoji
        mood_icons = {"calm": "🌙", "happy": "☀️", "magical": "✨", "adventure": "🌟", "sleepy": "💤"}
        mood_icon = mood_icons.get(page.mood, "📖")

        # Page number (bottom center)
        page_text = f"{mood_icon}  Page {page.page_number}  {mood_icon}"
        self._draw_centered_text(draw, page_text, page_font, (160, 140, 190), PAGE_H - 36)

        # Book title (bottom left, very small)
        draw.text((MARGIN, PAGE_H - 36), storybook.title, fill=(190, 190, 190), font=title_font)

        return img

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _draw_centered_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        font: ImageFont.FreeTypeFont,
        color: Tuple,
        y: int,
    ):
        """Draw text horizontally centered on the page."""
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
        except Exception:
            w = len(text) * 10   # rough estimate
        x = (PAGE_W - w) // 2
        draw.text((x, y), text, fill=color, font=font)

    def _draw_placeholder_art(self, img: Image.Image, draw: ImageDraw.Draw, page: StoryPage):
        """Draw a colored gradient as placeholder when image is missing."""
        MOOD_COLORS = {
            "calm":      [(135, 180, 255), (200, 230, 255)],
            "happy":     [(255, 220, 100), (255, 180, 80)],
            "magical":   [(180, 120, 255), (230, 180, 255)],
            "adventure": [(100, 200, 150), (60, 160, 120)],
            "sleepy":    [(100, 120, 180), (60, 80, 140)],
        }
        colors = MOOD_COLORS.get(page.mood, [(150, 180, 220), (200, 220, 255)])
        for y in range(ILLUSTRATION_H):
            t = y / ILLUSTRATION_H
            r = int(colors[0][0] * (1 - t) + colors[1][0] * t)
            g = int(colors[0][1] * (1 - t) + colors[1][1] * t)
            b = int(colors[0][2] * (1 - t) + colors[1][2] * t)
            draw.line([(0, y), (PAGE_W, y)], fill=(r, g, b))

        note_font = _load_font(20)
        note = f"Illustration: {page.illustration_prompt[:70]}…"
        draw.text((MARGIN, ILLUSTRATION_H // 2 - 20), note, fill=(255, 255, 255), font=note_font)
