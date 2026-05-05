"""
video_renderer.py
~~~~~~~~~~~~~~~~~
Renders the storybook as a narrated MP4 video.

Reuses ~90% of MusicVideoCompositor logic:
  - MoviePy + FFmpeg fallback (verbatim)
  - H.264 transcoding helper (verbatim)
  - Text clip creation (adapted for page text)
  - FFmpeg concatenation (verbatim)

Replaces:
  - Beat-synced scene cuts → page-based timeline
  - Outfit/lipsync layers → illustration images with Ken Burns zoom
  - Lyric overlays → page text overlays
"""

import os
import subprocess
import numpy as np
import logging
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont

from storybook_generator.utils.story_schemas import Storybook, StoryPage

logger = logging.getLogger(__name__)


class VideoRenderer:
    """
    Assembles the storybook into a narrated MP4 video.

    Architecture:
        for each page:
            Ken-Burns-zoomed illustration → narration audio → page text subtitle
        → fade transitions between pages
        → title card at start, end card at finish
        → single mixed audio track
    """

    VIDEO_WIDTH  = 1280
    VIDEO_HEIGHT = 720
    FPS          = 24

    def __init__(self, output_dir: str = "outputs/video"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────────────

    def render(
        self,
        storybook: Storybook,
        mixed_audio_paths: List[str],   # one WAV per page
        output_path: str,
    ) -> str:
        """
        Render the complete storybook video.

        Args:
            storybook: Storybook with pages that have image_path, text, duration_s set
            mixed_audio_paths: per-page mixed audio (narration + music + SFX)
            output_path: destination MP4 path

        Returns:
            output_path
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        try:
            from moviepy.editor import (
                VideoFileClip, ImageClip, AudioFileClip,
                CompositeVideoClip, concatenate_videoclips,
                TextClip, ColorClip,
            )
            return self._render_moviepy(storybook, mixed_audio_paths, output_path)
        except ImportError:
            logger.warning("MoviePy not available. Using FFmpeg fallback.")
            return self._render_ffmpeg(storybook, mixed_audio_paths, output_path)

    # ── MoviePy rendering ────────────────────────────────────────────────────

    def _render_moviepy(self, storybook: Storybook, mixed_audio_paths: List[str], output_path: str) -> str:
        try:
            from moviepy import (
                ImageClip, AudioFileClip, CompositeVideoClip,
                concatenate_videoclips, ColorClip
            )
        except ImportError:
            from moviepy.editor import (
                ImageClip, AudioFileClip, CompositeVideoClip,
                concatenate_videoclips, ColorClip
            )

        all_clips = []

        # Title card
        try:
            title_clip = self._make_title_card_moviepy(storybook, duration=4.0)
            all_clips.append(title_clip)
        except Exception as e:
            logger.warning(f"Title card failed: {e}")

        # Per-page clips
        for i, (page, audio_path) in enumerate(zip(storybook.pages, mixed_audio_paths)):
            try:
                page_clip = self._make_page_clip_moviepy(
                    page=page,
                    audio_path=audio_path,
                    duration=max(page.duration_s, 3.0),
                )
                if i > 0:
                    page_clip = page_clip.crossfadein(0.5)
                all_clips.append(page_clip)
            except Exception as e:
                logger.warning(f"Page {page.page_number} clip failed: {e}. Using fallback.")
                all_clips.append(self._static_fallback_clip_moviepy(page, audio_path))

        # End card
        try:
            end_clip = self._make_end_card_moviepy(storybook, duration=3.0)
            all_clips.append(end_clip)
        except Exception as e:
            logger.warning(f"End card failed: {e}")

        if not all_clips:
            logger.error("No clips to assemble!")
            return output_path

        # Concatenate
        try:
            final = concatenate_videoclips(all_clips, method="compose")
        except Exception as e:
            logger.warning(f"Concatenation failed: {e}. Trying sequential.")
            final = concatenate_videoclips(all_clips, method="chain")

        # Write
        try:
            final.write_videofile(
                output_path,
                fps=self.FPS,
                codec="libx264",
                audio_codec="aac",
                bitrate="5000k",
                audio_bitrate="192k",
                threads=4,
                preset="medium",
                logger=None,
            )
            logger.info(f"Storybook video saved: {output_path}")
        except Exception as e:
            logger.error(f"MoviePy write failed: {e}. Trying FFmpeg fallback.")
            return self._render_ffmpeg(storybook, mixed_audio_paths, output_path)

        return output_path

    def _make_page_clip_moviepy(self, page: StoryPage, audio_path: str, duration: float):
        """Create one page's video clip with Ken Burns zoom and text overlay."""
        try:
            from moviepy import ImageClip, AudioFileClip, CompositeVideoClip
        except ImportError:
            from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip

        W, H = self.VIDEO_WIDTH, self.VIDEO_HEIGHT

        # 1. Illustration with Ken Burns effect
        if page.image_path and os.path.exists(page.image_path):
            try:
                img = Image.open(page.image_path).convert("RGB")
                img = img.resize((W, H), Image.LANCZOS)
                img_array = np.array(img)
            except Exception:
                img_array = self._gradient_array(page.mood)
        else:
            img_array = self._gradient_array(page.mood)

        # Ken Burns: gentle 1.0→1.08 zoom over the page duration
        def make_frame(t):
            zoom = 1.0 + 0.08 * (t / max(duration, 1.0))
            h, w = img_array.shape[:2]
            new_w, new_h = int(w * zoom), int(h * zoom)
            pil = Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - w) // 2
            top  = (new_h - h) // 2
            cropped = pil.crop((left, top, left + w, top + h))
            
            # Draw dynamic text chunk on the frame!
            chunk = self._get_subtitle_chunk(page, t)
            cropped = self._draw_text_on_pil(cropped, chunk)
            
            return np.array(cropped)

        try:
            from moviepy import VideoClip
        except ImportError:
            from moviepy.editor import VideoClip
            
        bg_clip = VideoClip(make_frame, duration=duration).with_fps(self.FPS) if hasattr(VideoClip, "with_fps") else VideoClip(make_frame, duration=duration).set_fps(self.FPS)

        # 3. Page number overlay (top-right)
        try:
            pg_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            pg_img = self._draw_text_on_pil(pg_img, f"Page {page.page_number}", y_pos=20)
            pg_num = ImageClip(np.array(pg_img)).with_duration(duration) if hasattr(ImageClip, "with_duration") else ImageClip(np.array(pg_img)).set_duration(duration)
            layers = [bg_clip, pg_num]
        except Exception:
            layers = [bg_clip]

        composite = CompositeVideoClip(layers, size=(W, H)).with_duration(duration) if hasattr(CompositeVideoClip, "with_duration") else CompositeVideoClip(layers, size=(W, H)).set_duration(duration)

        # 4. Audio
        if os.path.exists(audio_path):
            try:
                audio = AudioFileClip(audio_path)
                if hasattr(composite, "with_audio"):
                    composite = composite.with_audio(audio)
                else:
                    composite = composite.set_audio(audio)
            except Exception as e:
                logger.debug(f"Audio attach failed: {e}")

        return composite

    def _make_title_card_moviepy(self, storybook: Storybook, duration: float = 4.0):
        """Title card: dark background + title + child name."""
        try:
            from moviepy import ColorClip, ImageClip, CompositeVideoClip
        except ImportError:
            from moviepy.editor import ColorClip, ImageClip, CompositeVideoClip

        W, H = self.VIDEO_WIDTH, self.VIDEO_HEIGHT

        bg = ColorClip(size=(W, H), color=[20, 15, 40], duration=duration)

        layers = [bg]
        try:
            # Render title with PIL
            title_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            title_img = self._draw_text_on_pil(title_img, storybook.title, y_pos=H//2 - 50)
            sub_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sub_img = self._draw_text_on_pil(sub_img, f"A bedtime story for {storybook.child_name} ✨", y_pos=H//2 + 30)
            
            t_clip = ImageClip(np.array(title_img)).with_duration(duration) if hasattr(ImageClip, "with_duration") else ImageClip(np.array(title_img)).set_duration(duration)
            s_clip = ImageClip(np.array(sub_img)).with_duration(duration) if hasattr(ImageClip, "with_duration") else ImageClip(np.array(sub_img)).set_duration(duration)
            
            layers.append(t_clip)
            layers.append(s_clip)
        except Exception as e:
            logger.debug(f"Title card text failed: {e}")

        return CompositeVideoClip(layers, size=(W, H)).with_duration(duration) if hasattr(CompositeVideoClip, "with_duration") else CompositeVideoClip(layers, size=(W, H)).set_duration(duration)

    def _make_end_card_moviepy(self, storybook: Storybook, duration: float = 3.0):
        """End card: 'The End'."""
        try:
            from moviepy import ColorClip, ImageClip, CompositeVideoClip
        except ImportError:
            from moviepy.editor import ColorClip, ImageClip, CompositeVideoClip

        W, H = self.VIDEO_WIDTH, self.VIDEO_HEIGHT
        bg = ColorClip(size=(W, H), color=[20, 15, 40], duration=duration)
        layers = [bg]

        try:
            end_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            end_img = self._draw_text_on_pil(end_img, "The End 🌙", y_pos=H//2 - 20)
            end_clip = ImageClip(np.array(end_img)).with_duration(duration) if hasattr(ImageClip, "with_duration") else ImageClip(np.array(end_img)).set_duration(duration)
            layers.append(end_clip)
        except Exception:
            pass

        return CompositeVideoClip(layers, size=(W, H)).with_duration(duration) if hasattr(CompositeVideoClip, "with_duration") else CompositeVideoClip(layers, size=(W, H)).set_duration(duration)

    def _static_fallback_clip_moviepy(self, page: StoryPage, audio_path: str):
        """Static gradient clip when page clip generation fails."""
        from moviepy.editor import ImageClip, AudioFileClip
        arr = self._gradient_array(page.mood)
        duration = max(page.duration_s, 3.0)
        clip = ImageClip(arr).set_duration(duration)
        if os.path.exists(audio_path):
            try:
                clip = clip.set_audio(AudioFileClip(audio_path))
            except Exception:
                pass
        return clip

    # ── FFmpeg fallback (mirrors MusicVideoCompositor._assemble_with_ffmpeg) ─

    def _render_ffmpeg(self, storybook: Storybook, mixed_audio_paths: List[str], output_path: str) -> str:
        """FFmpeg-based rendering when MoviePy is unavailable or fails."""
        import tempfile, shutil, cv2

        logger.info("Assembling storybook with FFmpeg fallback...")
        temp_dir = tempfile.mkdtemp()
        scene_paths = []

        for i, (page, audio_path) in enumerate(zip(storybook.pages, mixed_audio_paths)):
            scene_path = os.path.join(temp_dir, f"page_{i:03d}.mp4")
            duration   = max(page.duration_s, 3.0)

            # Base illustration frame
            if page.image_path and os.path.exists(page.image_path):
                try:
                    pil = Image.open(page.image_path).convert("RGB")
                    pil = pil.resize((self.VIDEO_WIDTH, self.VIDEO_HEIGHT), Image.LANCZOS)
                except Exception:
                    pil = Image.fromarray(self._gradient_array(page.mood))
            else:
                pil = Image.fromarray(self._gradient_array(page.mood))

            base_array = np.array(pil)
            h, w = base_array.shape[:2]

            # Write silent video frame-by-frame for Ken Burns + Dynamic Subtitles
            silent_path = os.path.join(temp_dir, f"silent_{i:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(silent_path, fourcc, self.FPS, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
            
            total_frames = int(duration * self.FPS)
            for f in range(total_frames):
                t = f / self.FPS
                # Ken Burns zoom: 1.0 to 1.08
                zoom = 1.0 + 0.08 * (t / max(duration, 1.0))
                new_w, new_h = int(w * zoom), int(h * zoom)
                
                # Resize and center crop
                resized = cv2.resize(base_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                left = (new_w - w) // 2
                top  = (new_h - h) // 2
                cropped = resized[top:top+h, left:left+w]
                
                # Draw subtitle chunk using PIL
                frame_pil = Image.fromarray(cropped)
                chunk = self._get_subtitle_chunk(page, t)
                frame_pil = self._draw_text_on_pil(frame_pil, chunk)
                
                out.write(cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR))
                
            out.release()

            # Combine with audio via FFmpeg
            if os.path.exists(audio_path):
                cmd = [
                    "ffmpeg", "-y",
                    "-i", silent_path,
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-shortest",
                    "-pix_fmt", "yuv420p",
                    scene_path,
                ]
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    shutil.copy(silent_path, scene_path)
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", silent_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    scene_path,
                ]
                subprocess.run(cmd, capture_output=True)

            if os.path.exists(scene_path):
                scene_paths.append(scene_path)

        if not scene_paths:
            logger.error("No scene videos generated.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return output_path

        # Concatenate all pages
        concat_list = os.path.join(temp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in scene_paths:
                f.write(f"file '{p}'\n")

        concat_video = os.path.join(temp_dir, "concat.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list, "-c", "copy", concat_video,
        ], capture_output=True)

        final_input = concat_video if os.path.exists(concat_video) else scene_paths[0]
        subprocess.run([
            "ffmpeg", "-y",
            "-i", final_input,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-b:v", "5000k",
            output_path,
        ], capture_output=True)

        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"FFmpeg storybook video saved: {output_path}")
        return output_path

    # ── Gradient helpers ─────────────────────────────────────────────────────

    MOOD_COLORS = {
        "calm":      ([135, 180, 255], [200, 230, 255]),
        "happy":     ([255, 220, 100], [255, 180, 80]),
        "magical":   ([180, 120, 255], [230, 180, 255]),
        "adventure": ([100, 200, 150], [60, 160, 120]),
        "sleepy":    ([100, 120, 180], [60, 80, 140]),
    }

    def _gradient_array(self, mood: str) -> np.ndarray:
        """RGB numpy array for a mood gradient (for MoviePy ImageClip)."""
        colors = self.MOOD_COLORS.get(mood, ([150, 180, 220], [200, 220, 255]))
        arr = np.zeros((self.VIDEO_HEIGHT, self.VIDEO_WIDTH, 3), dtype=np.uint8)
        for y in range(self.VIDEO_HEIGHT):
            t = y / self.VIDEO_HEIGHT
            for c in range(3):
                arr[y, :, c] = int(colors[0][c] * (1 - t) + colors[1][c] * t)
        return arr

    def _gradient_bgr(self, mood: str) -> np.ndarray:
        """BGR numpy array for a mood gradient (for OpenCV VideoWriter)."""
        arr = self._gradient_array(mood)
        return arr[:, :, ::-1].copy()   # RGB → BGR

    def _draw_text_on_pil(self, img: Image.Image, text: str, y_pos: Optional[int] = None) -> Image.Image:
        """Draw text perfectly centered on a PIL image."""
        from PIL import ImageDraw, ImageFont
        import textwrap

        draw = ImageDraw.Draw(img)
        try:
            # Mac system font
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except Exception:
            try:
                # Windows / Linux standard font
                font = ImageFont.truetype("Arial.ttf", 40)
            except Exception:
                font = ImageFont.load_default()

        # Simple text wrapping
        lines = textwrap.wrap(text, width=50)
        line_height = 48
        total_height = len(lines) * line_height

        if y_pos is None:
            # Subtitle at bottom
            y = self.VIDEO_HEIGHT - total_height - 60
        else:
            y = y_pos

        for line in lines:
            try:
                bbox = font.getbbox(line)
                w = bbox[2] - bbox[0]
            except Exception:
                w = len(line) * 20
            
            x = (self.VIDEO_WIDTH - w) // 2

            # Black outline
            outline_color = (0, 0, 0, 255)
            draw.text((x-2, y-2), line, font=font, fill=outline_color)
            draw.text((x+2, y-2), line, font=font, fill=outline_color)
            draw.text((x-2, y+2), line, font=font, fill=outline_color)
            draw.text((x+2, y+2), line, font=font, fill=outline_color)
            
            # White text
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
            y += line_height

        return img

    def _get_subtitle_chunk(self, page: StoryPage, t: float) -> str:
        """Get the active 3-word chunk using exact Whisper timestamps, or fallback to linear timing."""
        chunk_size = 3

        # 1. Exact timing via Whisper
        if hasattr(page, "word_timestamps") and page.word_timestamps:
            words = page.word_timestamps
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i+chunk_size]
                start_t = chunk_words[0]["start"]
                end_t = chunk_words[-1]["end"]
                # Display slightly before and after so it doesn't flicker during pauses
                if start_t - 0.2 <= t <= end_t + 0.4:
                    return " ".join(w["word"] for w in chunk_words)
            return ""

        # 2. Linear fallback
        text = page.text
        words = text.split()
        if not words:
            return ""
            
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        progress = max(0.0, min(1.0, t / max(page.duration_s, 1.0)))
        idx = int(progress * len(chunks))
        if idx >= len(chunks):
            idx = len(chunks) - 1
            
        return chunks[idx]
