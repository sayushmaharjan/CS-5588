"""
config.py — StorybookConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extends the original MusicVideoConfig pattern with storybook-specific fields.
Keeps all API key / device / LLM / SD / Chatterbox / MusicGen settings intact.
"""

import os
from dataclasses import dataclass
from typing import Optional


def _load_dotenv():
    """Load .env from project root — identical to original config.py."""
    try:
        from dotenv import load_dotenv
        # Walk up until we find a .env (handles running from subdirs)
        search = os.path.dirname(os.path.abspath(__file__))
        for _ in range(3):
            env_path = os.path.join(search, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path, override=False)
                return
            search = os.path.dirname(search)
        load_dotenv(override=False)
    except ImportError:
        pass


_load_dotenv()


@dataclass
class StorybookConfig:
    """
    Configuration for the AI Children's Story Creator.

    Preserves every field from MusicVideoConfig and adds story-specific ones.
    """
    # ── API keys (same as original) ──────────────────────────────────────────
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    groq_api_key: Optional[str] = None

    # ── LLM routing (same as original) ──────────────────────────────────────
    llm_base_url: Optional[str] = None
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # ── Hardware (same as original) ──────────────────────────────────────────
    device: str = "cuda"
    output_dir: str = "outputs"

    # ── Video settings (reused by VideoRenderer) ─────────────────────────────
    video_width: int = 1280
    video_height: int = 720
    fps: int = 24

    # ── Audio (reused) ───────────────────────────────────────────────────────
    target_sr: int = 44100
    narration_volume: float = 1.0
    music_volume: float = 0.35   # Quieter than vocals in music video (was 0.6)
    sfx_volume: float = 0.25

    # ── Models (same as original) ─────────────────────────────────────────────
    musicgen_model: str = "facebook/musicgen-small"
    sd_model: str = "runwayml/stable-diffusion-v1-5"

    # ── Story-specific ────────────────────────────────────────────────────────
    default_pages: int = 4          # Default number of story pages
    max_pages: int = 8
    min_pages: int = 2
    sd_style_prefix: str = (
        "children's book illustration, soft pastel colors, Pixar style, "
        "warm lighting, bedtime story style, kid-friendly, "
    )
    narration_speed_hint: str = (
        "Speak slowly, warmly, and clearly like a bedtime storyteller. "
    )

    @classmethod
    def from_env(cls) -> "StorybookConfig":
        """Build config from environment variables (identical pattern to original)."""
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            hf_token=os.environ.get("HF_TOKEN"),
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            llm_base_url=os.environ.get("LLM_BASE_URL"),
            llm_model=os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            device=os.environ.get("DEVICE", "cuda"),
            output_dir=os.environ.get("OUTPUT_DIR", "outputs"),
        )

    def best_api_key(self) -> str:
        """Return the first available API key (same priority as LyricsProcessor)."""
        return (
            self.hf_token
            or self.groq_api_key
            or self.openai_api_key
            or self.anthropic_api_key
            or ""
        )
