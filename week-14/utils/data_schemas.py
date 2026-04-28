"""
Cinematic Memory — Data Schemas
Pydantic models for all inter-module contracts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json


# ── Enums ──────────────────────────────────────────────────────────────────

class EmotionTag(str, Enum):
    JOYFUL      = "joyful"
    NOSTALGIC   = "nostalgic"
    REFLECTIVE  = "reflective"
    SAD         = "sad"
    EXCITED     = "excited"
    NEUTRAL     = "neutral"
    CELEBRATORY = "celebratory"
    TENDER      = "tender"

class SceneType(str, Enum):
    BEACH       = "beach"
    WEDDING     = "wedding"
    CITY        = "city"
    INDOORS     = "indoors"
    NATURE      = "nature"
    PARTY       = "party"
    TRAVEL      = "travel"
    PORTRAIT    = "portrait"
    UNKNOWN     = "unknown"

class ActPhase(str, Enum):
    SETUP       = "setup"
    PEAK        = "peak"
    REFLECTION  = "reflection"

class MediaType(str, Enum):
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"


# ── Visual Understanding ───────────────────────────────────────────────────

@dataclass
class VisualMetadata:
    """Output from Visual Understanding Module per media item."""
    media_id:       str
    file_path:      str
    media_type:     MediaType
    scene_type:     SceneType
    objects:        List[str]          = field(default_factory=list)
    emotions:       List[EmotionTag]   = field(default_factory=list)
    salience_score: float              = 0.5   # 0–1, higher = more cinematic
    clip_features:  Optional[List[float]] = None
    exif_timestamp: Optional[str]      = None
    description:    str                = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "clip_features"}
        d["media_type"] = self.media_type.value
        d["scene_type"] = self.scene_type.value
        d["emotions"]   = [e.value for e in self.emotions]
        return d


# ── Audio Understanding ────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    start:   float
    end:     float
    text:    str
    speaker: str = "narrator"
    emotion: EmotionTag = EmotionTag.NEUTRAL

@dataclass
class AudioMetadata:
    """Output from Audio Understanding Module per voice memo."""
    audio_id:    str
    file_path:   str
    duration_s:  float
    transcript:  str
    segments:    List[TranscriptSegment] = field(default_factory=list)
    overall_emotion: EmotionTag          = EmotionTag.NEUTRAL
    language:    str                     = "en"

    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments) if self.segments else self.transcript


# ── Narrative Engine ───────────────────────────────────────────────────────

@dataclass
class NarrationBeat:
    beat_id:       str
    act_phase:     ActPhase
    narration_text: str
    media_ids:     List[str]           # which photos/clips to use
    emotion:       EmotionTag
    duration_hint_s: float             # suggested screen time
    cut_speed:     str                 # "slow" | "medium" | "fast"
    music_prompt:  str                 # prompt for MusicGen
    ambient_prompt: str                # prompt for AudioLDM2

@dataclass
class DocumentaryScript:
    """Full structured script output from Narrative Engine."""
    title:          str
    total_duration_s: float
    beats:          List[NarrationBeat]
    arc_summary:    str
    raw_llm_output: str = ""

    def to_json(self) -> str:
        beats_dicts = []
        for b in self.beats:
            beats_dicts.append({
                "beat_id":        b.beat_id,
                "act_phase":      b.act_phase.value,
                "narration_text": b.narration_text,
                "media_ids":      b.media_ids,
                "emotion":        b.emotion.value,
                "duration_hint_s": b.duration_hint_s,
                "cut_speed":      b.cut_speed,
                "music_prompt":   b.music_prompt,
                "ambient_prompt": b.ambient_prompt,
            })
        return json.dumps({
            "title": self.title,
            "total_duration_s": self.total_duration_s,
            "arc_summary": self.arc_summary,
            "beats": beats_dicts,
        }, indent=2)


# ── Voice Synthesis ────────────────────────────────────────────────────────

@dataclass
class NarrationAudio:
    beat_id:     str
    audio_path:  str
    duration_s:  float
    emotion:     EmotionTag
    text:        str


# ── Music Generation ───────────────────────────────────────────────────────

@dataclass
class MusicSegment:
    beat_id:     str
    audio_path:  str
    duration_s:  float
    prompt_used: str
    emotion:     EmotionTag


# ── Ambient Sound ──────────────────────────────────────────────────────────

@dataclass
class AmbientSegment:
    beat_id:     str
    audio_path:  str
    duration_s:  float
    prompt_used: str
    scene_type:  SceneType


# ── Final Assembly ─────────────────────────────────────────────────────────

@dataclass
class AssemblyManifest:
    """Complete manifest passed to Video Assembly Engine."""
    script:          DocumentaryScript
    visual_meta:     Dict[str, VisualMetadata]   # media_id → metadata
    narration_audio: Dict[str, NarrationAudio]   # beat_id  → audio
    music_segments:  Dict[str, MusicSegment]     # beat_id  → music
    ambient_segments: Dict[str, AmbientSegment]  # beat_id  → ambient
    output_path:     str = "outputs/documentary.mp4"
