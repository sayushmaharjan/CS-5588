"""
Cinematic Memory — Configuration
Central config for all model IDs, hyperparams, paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent /".env", override=True)

# ── API Keys ───────────────────────────────────────────────────────────────
# Set your API key here. Only one is needed depending on LLM_PROVIDER.
GROQ_API_KEY            = os.environ.get("GROQ_API_KEY", "")       # free at console.groq.com
ANTHROPIC_API_KEY       = os.environ.get("ANTHROPIC_API_KEY", "")  # paid at anthropic.com

# ── LLM Provider ──────────────────────────────────────────────────────────
# Options: "groq"  →  uses GROQ_API_KEY  (free, fast — recommended)
#          "anthropic" → uses ANTHROPIC_API_KEY (paid)
#          "template"  → no LLM, uses built-in template fallback
LLM_PROVIDER            = "groq"

# ── Model IDs (all free / HuggingFace) ───────────────────────────────────
WHISPER_MODEL_ID        = "openai/whisper-small"
CLIP_MODEL_ID           = "openai/clip-vit-base-patch32"
MUSICGEN_MODEL_ID       = "facebook/musicgen-small"
AUDIOLDM2_MODEL_ID      = "cvssp/audioldm2"

# ── LLM Model IDs ─────────────────────────────────────────────────────────
# ANTHROPIC_MODEL         = "claude-sonnet-4-20250514"
GROQ_MODEL              = "llama-3.1-8b-instant"   # Free via groq.com (higher rate limit)
# GROQ_MODEL              = "llama-3.3-70b-versatile"   # Free via groq.com

# ── Audio settings ─────────────────────────────────────────────────────────
SAMPLE_RATE          = 16000
MUSICGEN_DURATION    = 10      # seconds per segment
AUDIOLDM2_DURATION   = 5 
TTS_SAMPLE_RATE      = 16000

# ── Video settings ─────────────────────────────────────────────────────────
VIDEO_FPS            = 24
VIDEO_RESOLUTION     = (1280, 720)
CROSSFADE_DURATION   = 0.5     # seconds
SLOW_CUT_DURATION    = 4.0     # nostalgia
MEDIUM_CUT_DURATION  = 2.5
FAST_CUT_DURATION    = 1.2     # excitement

# ── Audio mix levels (0–1) ─────────────────────────────────────────────────
NARRATION_LEVEL = 1.0
MUSIC_LEVEL     = 0.35
AMBIENT_LEVEL   = 0.20

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "/Users/sayush/Documents/cs5588/CS-5588/week-14/outputs"
TEMP_DIR     = "/Users/sayush/Documents/cs5588/CS-5588/week-14/temp"
UPLOAD_DIR   = "/Users/sayush/Documents/cs5588/CS-5588/week-14/uploads"

# ── Scene → ambient prompt mapping ────────────────────────────────────────
SCENE_AMBIENT_PROMPTS = {
    "beach":    "ocean waves crashing gently on a sandy beach, seagulls, wind",
    "wedding":  "soft indoor ambiance, gentle crowd murmur, clinking glasses",
    "city":     "urban street ambiance, distant traffic, city sounds",
    "indoors":  "quiet indoor room, subtle HVAC hum, soft footsteps",
    "nature":   "forest ambiance, birds chirping, gentle wind through leaves",
    "party":    "festive crowd laughter, music in background, celebration chatter",
    "travel":   "airport ambiance, train station sounds, adventure soundscape",
    "portrait": "quiet intimate room, soft ambient tone",
    "unknown":  "gentle neutral ambiance, soft room tone",
}

# ── Emotion → music prompt mapping ────────────────────────────────────────
EMOTION_MUSIC_PROMPTS = {
    "joyful":       "uplifting happy acoustic guitar, warm and bright, major key",
    "nostalgic":    "nostalgic slow piano melody, emotional warmth, bittersweet strings",
    "reflective":   "gentle introspective piano, soft pads, slow tempo, contemplative",
    "sad":          "slow melancholic strings, minor key, soft and tender",
    "excited":      "energetic upbeat music, driving rhythm, vibrant and dynamic",
    "neutral":      "calm cinematic underscore, soft orchestral texture",
    "celebratory":  "joyful orchestral celebration, brass fanfare, triumphant and warm",
    "tender":       "intimate soft guitar and strings, gentle melodic, heartfelt",
}

# ── Emotion → cut speed ───────────────────────────────────────────────────
EMOTION_CUT_SPEED = {
    "joyful":       "medium",
    "nostalgic":    "slow",
    "reflective":   "slow",
    "sad":          "slow",
    "excited":      "fast",
    "neutral":      "medium",
    "celebratory":  "fast",
    "tender":       "slow",
}
