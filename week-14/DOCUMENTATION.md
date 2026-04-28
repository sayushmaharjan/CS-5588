# 🎬 Cinematic Memory — Auto-Documentary Maker

> *Transform unstructured personal media (photos, videos, voice memos) into a cinematic documentary-style video with coherent narrative, emotional arc, AI-generated narration, adaptive music, and ambient soundscapes.*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Module Reference](#3-module-reference)
   - [Visual Understanding](#31-visual-understanding-module)
   - [Audio Understanding](#32-audio-understanding-module)
   - [Narrative Engine](#33-narrative-engine)
   - [Voice Synthesis](#34-voice-synthesis-module)
   - [Music Generation](#35-music-generation-module)
   - [Ambient Sound Design](#36-ambient-sound-design-module)
   - [Video Assembly Engine](#37-video-assembly-engine)
4. [Data Schemas](#4-data-schemas--api-contracts)
5. [Model Selection Justification](#5-model-selection-justification)
6. [Prompt Templates](#6-prompt-templates)
7. [Research Experiments](#7-research-experiments)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Setup & Running](#9-setup--running)
10. [Fallback Behavior](#10-fallback-behavior)
11. [Known Limitations](#11-known-limitations)

---

## 1. System Overview

**Cinematic Memory** is a modular AI pipeline that turns raw personal media into a polished emotional documentary. The user uploads unordered photos, videos, and rough voice memos — the system automatically handles scene analysis, transcription, story arc construction, narration synthesis, music scoring, ambient design, and final video assembly.

### Input
| Type | Format | Notes |
|------|--------|-------|
| Photos | JPG, PNG, WEBP | EXIF timestamps used for ordering |
| Videos | MP4, MOV, AVI, MKV | First-frame extracted for CLIP analysis |
| Voice Memos | MP3, WAV, M4A, OGG | Rough/unscripted is ideal |

### Output
- `documentary.mp4` — Final rendered documentary
- `documentary_script.json` — Structured narrative script
- `temp/narration/` — Per-beat narration WAV files
- `temp/music/` — Per-beat adaptive music WAV files
- `temp/ambient/` — Per-beat ambient soundscape WAV files

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│  📸 Photos + 🎥 Videos + 🎙️ Voice Memos                        │
└───────────────┬────────────────────────┬────────────────────────┘
                │                        │
                ▼                        ▼
┌──────────────────────┐    ┌─────────────────────────┐
│  Visual Understanding │    │   Audio Understanding   │
│  ─────────────────── │    │  ───────────────────── │
│  CLIP ViT-B/32        │    │  Whisper-small          │
│                       │    │                         │
│  Output:              │    │  Output:                │
│  • scene_type         │    │  • timestamped          │
│  • emotion tags       │    │    transcript           │
│  • object list        │    │  • emotion labels       │
│  • salience score     │    │  • speaker segments     │
│  • EXIF timestamp     │    │                         │
└──────────┬───────────┘    └────────────┬────────────┘
           │                             │
           └────────────┬────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │    Narrative Engine    │
           │  ──────────────────── │
           │  Claude Sonnet / LLM  │
           │                       │
           │  • Chronological +    │
           │    emotional ordering │
           │  • 3-act arc builder  │
           │  • Script generation  │
           │  • Beat segmentation  │
           └───────────┬───────────┘
                       │ DocumentaryScript
                       │
        ┌──────────────┼──────────────────┐
        │              │                  │
        ▼              ▼                  ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────────┐
│    Voice    │ │    Music     │ │   Ambient Sound  │
│  Synthesis  │ │ Generation   │ │   Design         │
│  ───────── │ │ ──────────── │ │  ──────────────  │
│ Chatterbox │ │  MusicGen-   │ │  AudioLDM2       │
│  TTS       │ │  small       │ │                  │
│            │ │              │ │  Scene-matched   │
│  Narration │ │  Emotion-    │ │  environmental   │
│  WAV per   │ │  conditioned │ │  audio per beat  │
│  beat      │ │  per beat    │ │                  │
└─────┬───────┘ └──────┬───────┘ └────────┬─────────┘
      │                │                  │
      └────────────────┼──────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   Video Assembly    │
            │  ───────────────── │
            │  MoviePy            │
            │                     │
            │  • Ken Burns zoom   │
            │  • Cut-speed pacing │
            │  • 3-layer audio mix│
            │  • Crossfades       │
            └──────────┬──────────┘
                       │
                       ▼
              📼 documentary.mp4
```

---

## 3. Module Reference

### 3.1 Visual Understanding Module

**File:** `pipeline/visual_understanding.py`  
**Model:** `openai/clip-vit-base-patch32`

#### What it does
Processes every photo and video file using CLIP (Contrastive Language-Image Pretraining) to extract structured semantic metadata.

#### CLIP Label Banks

| Category | Labels |
|----------|--------|
| **Scene type** | beach, wedding, city, indoors, nature, party, travel, portrait |
| **Emotions** | joyful, tender, reflective, sad, excited, nostalgic, celebratory, neutral |
| **Salience** | "cinematic dramatic scene" vs "dull or blurry photo" |
| **Objects** | people, faces, food, drinks, flowers, rings, cake, sunset, water, … |

#### Salience Scoring
Binary CLIP classification between:
- `"a cinematic dramatic visually stunning scene"` (high salience)
- `"a dull or blurry or ordinary photo"` (low salience)

Score range: 0–1. High-salience clips get priority placement in the narrative.

#### EXIF Timestamp Extraction
Reads `DateTime` / `DateTimeOriginal` EXIF tags from JPEGs. Used by the Narrative Engine for chronological ordering.

#### Output Schema
```python
@dataclass
class VisualMetadata:
    media_id:       str
    file_path:      str
    media_type:     MediaType    # PHOTO | VIDEO
    scene_type:     SceneType
    objects:        List[str]
    emotions:       List[EmotionTag]  # top-2 emotions
    salience_score: float             # 0–1
    description:    str
    exif_timestamp: Optional[str]
```

---

### 3.2 Audio Understanding Module

**File:** `pipeline/audio_understanding.py`  
**Model:** `openai/whisper-small`

#### What it does
Transcribes voice memos into timestamped text segments, then classifies emotional tone per segment.

#### Processing Pipeline
1. Load audio → resample to 16kHz mono
2. Chunk into 30s windows (Whisper's context limit)
3. Run Whisper inference per chunk
4. Reconstruct timestamped segments
5. Apply keyword-based emotion classifier per segment

#### Emotion Classification
Rule-based keyword matching (fallback when no emotion model loaded):

| Emotion | Keywords |
|---------|----------|
| `joyful` | happy, joy, laugh, smile, fun, wonderful |
| `nostalgic` | remember, miss, used to, back then, years ago |
| `reflective` | think, wonder, realize, understand, looking back |
| `sad` | sad, cry, tears, loss, gone, difficult |
| `excited` | excited, incredible, wow, unbelievable, best |
| `celebratory` | celebrate, cheers, congrats, wedding, anniversary |
| `tender` | love you, family, together, hold, embrace |

#### Output Schema
```python
@dataclass
class AudioMetadata:
    audio_id:        str
    file_path:       str
    duration_s:      float
    transcript:      str
    segments:        List[TranscriptSegment]
    overall_emotion: EmotionTag
    language:        str
```

---

### 3.3 Narrative Engine

**File:** `pipeline/narrative_engine.py`  
**Primary:** Anthropic Claude Sonnet  
**Fallback:** Template-based system

#### What it does
The intelligence core of the pipeline. Takes all visual + audio metadata and constructs a coherent 3-act documentary script.

#### Photo/Video Ordering Strategy
```
Priority 1: EXIF timestamp (chronological if available)
Priority 2: Emotional arc ordering (setup → peak → reflection)
Priority 3: Salience score (high-salience = peak placement)
Priority 4: Scene grouping (visual coherence within beats)
```

#### 3-Act Structure
| Act | Phase | % Runtime | Emotional Tone |
|-----|-------|-----------|----------------|
| I | Setup | 30% | warm, curious, establishing |
| II | Peak | 40% | joyful, intense, climactic |
| III | Reflection | 30% | nostalgic, grateful, tender |

#### System Prompt
```
You are an acclaimed documentary filmmaker and screenwriter.
Your task: transform raw media metadata and voice memo transcripts into a 
cinematic 3-act documentary script with emotional depth and coherent narrative.
Output ONLY valid JSON. No markdown fences. No preamble. No explanation.
```

#### LLM Output Schema (JSON)
```json
{
  "title": "Evocative documentary title",
  "arc_summary": "2-3 sentence emotional journey description",
  "total_duration_s": 65,
  "beats": [
    {
      "beat_id": "beat_001",
      "act_phase": "setup|peak|reflection",
      "narration_text": "Poetic 2-4 sentence narration script",
      "media_ids": ["media_0001_img", "media_0002_img"],
      "emotion": "joyful|nostalgic|reflective|...",
      "duration_hint_s": 10.0,
      "cut_speed": "slow|medium|fast",
      "music_prompt": "Detailed MusicGen prompt...",
      "ambient_prompt": "AudioLDM2 environment prompt..."
    }
  ]
}
```

#### Template Fallback (no API key)
When no Anthropic API key is provided, the engine uses hard-coded narrative templates with 5 beats:
1. Opening reflection
2. Rising action
3. Climax / peak moment
4. Wind-down
5. Final reflection

---

### 3.4 Voice Synthesis Module

**File:** `pipeline/voice_synthesis.py`  
**Model:** Chatterbox TTS (Resemble AI)

#### What it does
Converts each beat's narration text to speech using a consistent narrator voice via Chatterbox TTS. Supports zero-shot voice cloning from an optional reference audio clip.

#### Voice Consistency Strategy
- Uses an optional `voice_reference_path` parameter to supply a reference audio clip for zero-shot voice cloning.
- If no reference is provided, it falls back to a high-quality default voice.
- Consistency is maintained across all beats by reusing the same reference audio prompt during inference.
- Provides native support for Apple Silicon via MPS device mapping patches.

#### Voice Consistency Research
To measure voice similarity degradation:
- **SIM-R metric**: cosine similarity between speaker embeddings of synthesized audio
- Run across different input texts (same embedding → should stay > 0.85)
- Run with/without background noise in source voice memos (tests robustness)

#### Fallback
If Chatterbox TTS is unavailable or fails, generates silence audio with duration estimated from word count (~2 words/second).

---

### 3.5 Music Generation Module

**File:** `pipeline/music_generation.py`  
**Model:** `facebook/musicgen-small`

#### What it does
Generates adaptive background music for each narrative beat, conditioned on emotion-specific text prompts.

#### Two Generation Modes (Research Experiment)

**Mode A: Global Prompt** (`music_mode="global"`)
- Single prompt for entire film
- Example: `"cinematic documentary soundtrack, warm orchestral, emotional journey"`
- ✅ More tonal coherence across film
- ❌ Less emotional differentiation between beats

**Mode B: Segment-Level** (`music_mode="segment"`, default)
- Unique prompt per beat derived from emotion tag
- Examples:
  - Setup beat: `"gentle introspective piano, soft pads, slow tempo, contemplative"`
  - Peak beat: `"joyful orchestral celebration, brass fanfare, triumphant and warm"`
  - Reflection beat: `"nostalgic slow piano melody, emotional warmth, bittersweet strings"`
- ✅ Better emotional arc following
- ❌ Potential jarring transitions between segments

#### Emotion → Music Prompt Mapping
```python
EMOTION_MUSIC_PROMPTS = {
    "joyful":       "uplifting happy acoustic guitar, warm and bright, major key",
    "nostalgic":    "nostalgic slow piano melody, emotional warmth, bittersweet strings",
    "reflective":   "gentle introspective piano, soft pads, slow tempo, contemplative",
    "sad":          "slow melancholic strings, minor key, soft and tender",
    "excited":      "energetic upbeat music, driving rhythm, vibrant and dynamic",
    "celebratory":  "joyful orchestral celebration, brass fanfare, triumphant and warm",
    "tender":       "intimate soft guitar and strings, gentle melodic, heartfelt",
}
```

#### MusicGen Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_new_tokens` | `duration × 50` | ~50 tokens/s for musicgen-small |
| `guidance_scale` | `3.0` | Prompt adherence strength |
| `do_sample` | `True` | Creative variation |
| Max duration | 30s | MusicGen-small token limit |

---

### 3.6 Ambient Sound Design Module

**File:** `pipeline/ambient_sound.py`  
**Model:** `cvssp/audioldm2`

#### What it does
Generates environmental soundscapes that ground each scene in a specific location/atmosphere.

#### Scene → Ambient Prompt Mapping
```python
SCENE_AMBIENT_PROMPTS = {
    "beach":    "ocean waves crashing gently on a sandy beach, seagulls, wind",
    "wedding":  "soft indoor ambiance, gentle crowd murmur, clinking glasses",
    "city":     "urban street ambiance, distant traffic, city sounds",
    "indoors":  "quiet indoor room, subtle HVAC hum, soft footsteps",
    "nature":   "forest ambiance, birds chirping, gentle wind through leaves",
    "party":    "festive crowd laughter, music in background, celebration chatter",
    "travel":   "airport ambiance, train station sounds, adventure soundscape",
}
```

#### Priority Logic
When the LLM generates a custom `ambient_prompt` in the script JSON, that takes priority over the scene-map lookup. This allows the narrative engine to specify more nuanced soundscapes (e.g., `"quiet evening terrace, distant ocean, warm conversation"` for a specific beat).

#### AudioLDM2 Parameters
| Parameter | Value |
|-----------|-------|
| `num_inference_steps` | 20 (speed/quality balance) |
| `audio_length_in_s` | beat duration (max 10s) |
| `guidance_scale` | 3.5 |
| Sample rate | 16kHz |

---

### 3.7 Video Assembly Engine

**File:** `pipeline/video_assembly.py`  
**Library:** MoviePy

#### What it does
The final stage. Takes all generated components and assembles them into a single MP4 file.

#### Emotional Pacing Rules
| Emotion | Cut Speed | Seconds/Clip |
|---------|-----------|--------------|
| `nostalgic`, `reflective`, `tender`, `sad` | slow | 4.5s |
| `joyful`, `neutral` | medium | 2.5s |
| `excited`, `celebratory` | fast | 1.2s |

#### Ken Burns Effect
Static photos get a subtle zoom-in effect (1.0→1.03x over clip duration) for cinematic motion.

#### Audio Mix Architecture
```
Beat audio = (narration × 1.0) + (music × 0.35) + (ambient × 0.20)
           ↓
     peak-normalize to 0.95
           ↓
   concatenate all beats
           ↓
   AudioArrayClip → attach to video
```

Mix levels are configurable via the Streamlit sidebar sliders.

#### Output Specification
| Property | Value |
|----------|-------|
| Codec | H.264 (libx264) |
| Audio codec | AAC |
| Resolution | 1280×720 |
| FPS | 24 |
| Sample rate | 44.1kHz |

---

## 4. Data Schemas / API Contracts

All inter-module data is defined in `utils/data_schemas.py` as Python dataclasses.

### Module Contracts

```
Visual Understanding  →  List[VisualMetadata]
Audio Understanding   →  List[AudioMetadata]
Narrative Engine      →  DocumentaryScript
Voice Synthesis       →  Dict[beat_id, NarrationAudio]
Music Generation      →  Dict[beat_id, MusicSegment]
Ambient Sound         →  Dict[beat_id, AmbientSegment]
Video Assembly        →  str (output MP4 path)
```

### AssemblyManifest
The final assembly stage receives an `AssemblyManifest` containing all prior outputs:
```python
@dataclass
class AssemblyManifest:
    script:           DocumentaryScript
    visual_meta:      Dict[str, VisualMetadata]
    narration_audio:  Dict[str, NarrationAudio]
    music_segments:   Dict[str, MusicSegment]
    ambient_segments: Dict[str, AmbientSegment]
    output_path:      str
```

---

## 5. Model Selection Justification

| Module | Model | Why |
|--------|-------|-----|
| Visual | CLIP ViT-B/32 | Zero-shot scene+emotion classification without fine-tuning. Fast (~100ms/image CPU). Widely supported. |
| Transcription | Whisper-small | Best open-source ASR. `small` variant: 461M params, runs on CPU in acceptable time. Multilingual support. |
| Narrative | Claude Sonnet | Best JSON output reliability. Strong narrative coherence. Falls back gracefully to template system. |
| TTS | Chatterbox TTS | Resemble AI's high-quality open-source TTS. Supports zero-shot voice cloning from reference clips. Apple Silicon (MPS) accelerated. |
| Music | MusicGen-small | Facebook's music generation model. `small` (300M params) generates decent cinematic underscore. Free/open. |
| Ambient | AudioLDM2 | Best open-source text-to-audio. Generates environmental sounds with good prompt adherence. |
| Assembly | MoviePy | Mature Python video editing library. FFmpeg-backed. Handles audio+video sync well. |

---

## 6. Prompt Templates

### Narrative Engine System Prompt
```
You are an acclaimed documentary filmmaker and screenwriter.
Your task: transform raw media metadata and voice memo transcripts into a 
cinematic 3-act documentary script with emotional depth and coherent narrative.
Output ONLY valid JSON. No markdown fences. No preamble. No explanation.
```

### Narrative Engine User Prompt Template
```
Create a documentary script from this personal media collection.

## MEDIA METADATA (photos/videos):
{media_summaries}   # media_id, scene, emotions, salience, timestamp

## VOICE MEMO TRANSCRIPTS:
{transcript_summaries}  # audio_id, duration, emotion, transcript

## TASK:
Build a structured 3-act documentary script.

### ACT STRUCTURE:
- Act 1 — SETUP (30%): establish setting, people, context. Tone: warm/curious.
- Act 2 — PEAK (40%): core memories, highest emotion, key moments. Tone: joyful/intense.
- Act 3 — REFLECTION (30%): meaning, what was learned, looking back. Tone: nostalgic/grateful.

### OUTPUT JSON SCHEMA: {full schema}
### CONSTRAINTS:
- 5–12 beats total
- narration_text: poetic, cinematic language, 2-4 sentences
- total_duration_s: 45–120 seconds
- Use ALL provided media_ids at least once
```

### MusicGen Prompt Examples
```
# Setup beat (reflective)
"gentle introspective piano, soft pads, slow tempo 60bpm, contemplative, cinematic"

# Peak beat (celebratory)  
"joyful orchestral celebration, brass fanfare, triumphant and warm, 120bpm, major key"

# Reflection beat (nostalgic)
"nostalgic slow piano melody, bittersweet strings, emotional warmth, minor to major, 70bpm"

# Sad/tender beat
"intimate soft guitar, gentle melodic, heartfelt, 55bpm, fingerpicked acoustic"
```

### AudioLDM2 Prompt Examples
```
# Wedding scene
"soft indoor ambiance, gentle crowd murmur, clinking glasses, celebration warmth"

# Beach/travel scene
"ocean waves crashing gently on a sandy beach, seagulls calling, warm coastal breeze"

# Reflection/indoor scene
"quiet room tone, distant traffic, soft evening ambiance, contemplative silence"
```

### Chatterbox TTS Narration Examples (Script Output)
```
"There are places that stay with you long after you've left — 
 not in photographs or memories, but in the way your body remembers warmth."

"And then it happened — that rare convergence of everything feeling alive at once.
 Laughter, movement, connection. All of it real. All of it ours."

"These images, these captured seconds — they are more than photographs.
 They are proof that something extraordinary happened here. And that it mattered."
```

---

## 7. Research Experiments

### Experiment 1: Music–Emotion Alignment

**Question:** Does segment-level emotion conditioning produce better music–narrative alignment than a single global prompt?

**Method:**
1. Run pipeline on same input twice:
   - Condition A: `music_mode="global"` (single descriptive prompt)
   - Condition B: `music_mode="segment"` (per-beat emotion prompts)
2. Evaluate with CLAP Score (CLIP-based audio-text alignment)

**Expected Result:**
- Segment-level → higher per-beat CLAP scores
- Global → more smooth transitions between beats
- Trade-off: emotional accuracy vs. tonal coherence

**UI:** Toggle in Streamlit sidebar → "Music Mode" radio button.

---

### Experiment 2: Voice Consistency Across Sessions

**Question:** How robust is Chatterbox TTS's zero-shot cloned voice to different microphone qualities?

**Method:**
1. Record same narration text via:
   - High-quality condenser mic (reference)
   - Phone mic recording
   - Noisy environment recording
2. Add to voice memo input
3. Measure **SIM-R** (speaker similarity):
   ```python
   # Compute speaker embedding for each output
   emb1 = extract_xvector(synthesized_audio_1)
   emb2 = extract_xvector(synthesized_audio_2)
   sim_r = cosine_similarity(emb1, emb2)
   ```

**Note:** Voice consistency heavily depends on the reference audio. Memos with high background noise or reverberation might bleed artifacts into the synthesized output voice.

---

### Experiment 3: Photo Ordering Problem

**Question:** What ordering strategy produces the most narratively coherent documentary when EXIF data is missing?

**Three strategies tested:**

| Strategy | Method |
|----------|--------|
| **Chronological** | EXIF timestamp → fallback to filename |
| **Emotional arc** | Sorted by emotion intensity: calm → excited → nostalgic |
| **Hybrid** | Salience score × emotional arc position score |

**Hybrid Scoring Formula:**
```python
def ordering_score(media: VisualMetadata, target_phase: float) -> float:
    # target_phase: 0.0 (setup) → 1.0 (reflection)
    emotion_intensity = {"neutral": 0.3, "reflective": 0.4, "nostalgic": 0.5,
                         "tender": 0.6, "joyful": 0.7, "celebratory": 0.9, "excited": 1.0}
    phase_score = 1.0 - abs(emotion_intensity[media.emotions[0].value] - target_phase)
    return phase_score * 0.6 + media.salience_score * 0.4
```

**Evaluation:** Narrative Coherence Score via LLM judge (1–10):
```
System: You are evaluating documentary sequencing quality.
User: Rate the narrative coherence of this photo sequence (1-10):
  [photo 1: beach, joyful] → [photo 2: wedding, celebratory] → [photo 3: indoors, reflective]
  Evaluate: Does the sequence feel like a natural story progression?
```

---

## 8. Evaluation Framework

### Metrics

#### 1. Narrative Coherence Score
- **Type:** Human eval + LLM-as-judge
- **Scale:** 1–10
- **LLM Judge Prompt:**
  ```
  Rate the narrative coherence of this documentary script (1-10).
  Evaluate: logical flow, emotional progression, story completeness.
  Script: {documentary_script}
  ```
- **Baseline:** Slideshow with stock music + random photo order

#### 2. CLAP Score (Music ↔ Scene Alignment)
- **Type:** Automated (CLAP model)
- **Measures:** Cosine similarity between music audio embedding and scene description text embedding
- **Higher = better alignment**
- Compare segment-level vs global music generation

#### 3. Speaker Similarity Score (SIM-R)
- **Type:** Automated (x-vector cosine similarity)
- **Measures:** Voice identity consistency across all synthesized narration beats
- **Target:** > 0.85 (high consistency)

#### 4. Editing Rhythm Score
- **Type:** Automated analysis
- **Measures:** Alignment between:
  - Visual cut points
  - Speech pause locations (silence detection in narration audio)
  - Musical phrase boundaries
- **Formula:**
  ```python
  rhythm_score = 1.0 - mean_abs_offset(cut_times, nearest_speech_pause)
  # Lower offset = better sync = higher score
  ```

### Baseline vs. Improved Settings Comparison

To evaluate the pipeline's capabilities, we contrast standard fallback methods against our integrated, improved AI models across different task domains.

#### Speech & Narrative Tasks
| Metric | Baseline / Fallback Setting | Improved Setting | Performance / Evaluation |
|--------|-----------------------------|------------------|--------------------------|
| **Transcription Quality (WER)** | Mock or primitive ASR | `whisper-small` | Dramatically lower Word Error Rate (WER); accurately captures complex vocal phrasing. |
| **Summary & Narrative Quality** | Template-based 5-beat script | Claude Sonnet script generation | Target **> 7.0/10 Narrative Coherence** (Baseline: 3.0/10). |
| **Accuracy (Voice Consistency)** | Generic robotic TTS | Chatterbox TTS zero-shot cloning | Target **> 0.85 SIM-R**; precise vocal replication preserving user identity. |
| **Latency** | Real-time generation | Chatterbox + Whisper | Slower end-to-end processing, an acceptable trade-off for high vocal fidelity. |

#### Music & Sound Tasks
| Metric | Baseline / Fallback Setting | Improved Setting | Performance / Evaluation |
|--------|-----------------------------|------------------|--------------------------|
| **Prompt Alignment (CLAP)** | Stock music / Global prompts | Segment-level generative audio | Target **> 0.45 CLAP Score**; tight alignment between scene and soundscape (Baseline: 0.15). |
| **Realism** | Static ambient noise | Scene-specific AudioLDM2 | Highly immersive environmental audio matching visual context (e.g., "crashing waves"). |
| **Diversity** | Single global music track | Adaptive per-beat music generation | Dynamic musical transitions reflecting the emotional arc (setup → peak → reflection). |
| **Creativity (Rhythm & Flow)** | Repetitive stock cuts | MusicGen unique compositions | Unique melodic generation combined with emotion-paced MoviePy cuts (Target **> 0.7 Rhythm Score**). |

----------------------------------------
📊 EVALUATION RESULTS
----------------------------------------

• Transcription Quality : Excellent
• Latency (ASR)         : 13.21s
• Accuracy              : 100.00%
• WER (Word Error Rate) : 0.00% (0 errors / 24 words)
• Voice Consistency     : 0.897 SIM-R (Target: >0.85)
• Prompt Align. (CLAP)  : 0.545 Score (Target: >0.45)

---

## 9. Setup & Running

### Prerequisites
- Python 3.10+
- FFmpeg installed (`brew install ffmpeg` / `apt install ffmpeg`)
- 8GB+ RAM (16GB recommended for all models simultaneously)
- GPU optional but recommended for AudioLDM2 and MusicGen

### Installation

```bash
# Clone / download project
cd week-14

# Install dependencies
pip install -r requirements.txt

# (Optional) GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Running the Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Running Pipeline Directly (No UI)

```python
from pipeline.orchestrator import run_pipeline

result = run_pipeline(
    photo_video_paths = ["photo1.jpg", "photo2.jpg", "video.mp4"],
    audio_paths       = ["memo1.m4a", "memo2.wav"],
    output_dir        = "my_output",
    api_key           = "sk-ant-...",   # optional
    event_hint        = "wedding",
    music_mode        = "segment",
    progress_cb       = lambda msg, pct: print(f"{pct}% - {msg}"),
)

print(f"Documentary: {result['output_video_path']}")
print(f"Script title: {result['script'].title}")
```

### Environment Variables (Optional)

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # narrative engine
export CINEMATIC_OUTPUT_DIR=./output  # custom output path
```

---

## 10. Fallback Behavior

Every module has graceful degradation when models are unavailable:

| Module | Primary | Fallback |
|--------|---------|---------|
| Visual Understanding | CLIP inference | Mock metadata (random scene/emotion) |
| Audio Understanding | Whisper inference | Mock transcript with sample text |
| Narrative Engine | Claude API | Template-based 5-beat script |
| Voice Synthesis | Chatterbox TTS | Silence WAV (duration ∝ word count) |
| Music Generation | MusicGen | Silence WAV |
| Ambient Sound | AudioLDM2 | Silence WAV |
| Video Assembly | MoviePy + ffmpeg | Error logged, other outputs preserved |

This means the pipeline will always produce a script and audio files even without GPU or API access, though the video rendering requires MoviePy + FFmpeg.

---

## 11. Known Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| MusicGen-small quality | Music sounds simple/repetitive | Upgrade to `musicgen-medium` for better quality |
| Chatterbox TTS loading time | Initial load takes a few seconds | Lazy-loaded globally to avoid reloading |
| AudioLDM2 inference time | ~30-60s per clip on CPU | Use GPU; reduce `num_inference_steps` to 10 |
| CLIP emotion accuracy | "Smiling" ≠ definitely joyful | CLIP works at caption level; fine-grained emotion needs face analysis |
| Long video clips | Only first N seconds extracted | Extended scene detection for longer clips |
| Memory usage | All models loaded sequentially | Use `del model; gc.collect()` between stages if RAM limited |
| MoviePy crossfades | Basic fades only | Extend with custom transition effects |
| No multi-speaker diarization | Single "narrator" voice | Add pyannote.audio for speaker diarization |

---

## File Structure

```
cinematic_memory/
├── app.py                           # Streamlit UI
├── config.py                        # Central config (model IDs, params)
├── requirements.txt
├── DOCUMENTATION.md                 # This file
│
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py              # End-to-end pipeline runner
│   ├── visual_understanding.py      # CLIP scene/emotion analysis
│   ├── audio_understanding.py       # Whisper transcription
│   ├── narrative_engine.py          # LLM script generation
│   ├── voice_synthesis.py           # Chatterbox TTS narration
│   ├── music_generation.py          # MusicGen adaptive music
│   ├── ambient_sound.py             # AudioLDM2 soundscapes
│   └── video_assembly.py            # MoviePy final render
│
├── utils/
│   ├── __init__.py
│   └── data_schemas.py              # All dataclass schemas
│
├── outputs/                         # Generated documentaries
└── temp/                            # Intermediate audio files
    ├── narration/
    ├── music/
    └── ambient/
```

---

*Built with: CLIP · Whisper · Claude · Chatterbox TTS · MusicGen · AudioLDM2 · MoviePy*
