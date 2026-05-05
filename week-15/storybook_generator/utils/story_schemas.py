"""
story_schemas.py
~~~~~~~~~~~~~~~~
Data models for the AI Children's Story Creator.

Mirrors the role of utils/data_schemas.py from the music video pipeline,
but replaces MusicVideoScene/MusicVideoScript with story-specific structures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Core story structure
# ---------------------------------------------------------------------------

@dataclass
class StoryPage:
    """
    One page of the storybook.

    Equivalent to MusicVideoScene in the original pipeline, but page-based
    instead of beat-based.
    """
    page_number: int          # 1-indexed
    text: str                 # Actual story text for this page (2–4 sentences)
    illustration_prompt: str  # SD text-to-image prompt
    mood: str                 # calm | happy | magical | adventure | sleepy
    sound_effects: List[str]  # e.g. ["forest", "magic"] — keys into SFX_MAP

    # Filled in after generation
    image_path: str = ""      # Path to generated illustration PNG
    narration_path: str = ""  # Path to Chatterbox TTS audio WAV
    mixed_audio_path: str = ""# Narration + SFX + music mix
    duration_s: float = 0.0   # Duration of narration audio (computed)
    word_timestamps: List[Dict] = field(default_factory=list) # Whisper word-level timestamps


@dataclass
class Storybook:
    """
    Complete storybook, equivalent to MusicVideoScript.
    """
    title: str
    child_name: str
    age: int
    theme: str
    pages: List[StoryPage]

    # Music / ambient
    music_prompt: str    # MusicGen prompt for the ambient track
    total_duration_s: float = 0.0  # Filled after narration generation

    # Visual style
    color_palette: str = "soft pastels, warm, dreamy"
    art_style: str = "Pixar-style children's book illustration"


# ---------------------------------------------------------------------------
# Asset tracking (mirrors GeneratedAssets)
# ---------------------------------------------------------------------------

@dataclass
class GeneratedStoryAssets:
    """Tracks paths to all generated files."""
    # Per page
    illustration_paths: Dict[int, str] = field(default_factory=dict)   # page_number -> path
    narration_paths: Dict[int, str] = field(default_factory=dict)       # page_number -> path
    mixed_audio_paths: Dict[int, str] = field(default_factory=dict)     # page_number -> path

    # Global
    ambient_music_path: str = ""
    output_video_path: str = ""
    output_pdf_path: str = ""
    subtitle_path: str = ""
