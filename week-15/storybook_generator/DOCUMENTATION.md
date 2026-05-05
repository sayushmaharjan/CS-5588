# AI Children's Story Creator — Documentation

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Reuse Map](#2-component-reuse-map)
3. [Data Flow](#3-data-flow)
4. [New Modules](#4-new-modules)
5. [Reused Modules](#5-reused-modules)
6. [Configuration](#6-configuration)
7. [Usage](#7-usage)
8. [Output Structure](#8-output-structure)
9. [Extending the System](#9-extending-the-system)
10. [Evaluation Suite](#10-evaluation-suite)

---

## 1. Architecture Overview
```
Story Idea → Story (LLM) → Illustrations (SD) → Narration (Chatterbox) → Music (MusicGen) → PDF + Video
```

### Conceptual Mapping

| Music Video Concept | Storybook Concept     |
|--------------------|-----------------------|
| Lyrics             | Story idea/theme      |
| Scene              | Page                  |
| Beat sync          | Page duration (narration length) |
| MusicGen full song | MusicGen ambient loop |
| Lip sync           | Narration alignment   |
| Video compositor   | Storybook renderer    |
| Subtitle SRT       | Page captions SRT     |

---

## 2. Component Reuse Map

### Reused Unchanged (import directly from original `pipeline/` and `utils/`)

| Original File | Role in Storybook | How Reused |
|---|---|---|
| `pipeline/music_generator.py` | Chatterbox TTS + MusicGen | `NarrationGenerator` calls `generate_vocals_chatterbox()`; `SoundEngine` calls `generate_instrumental()` |
| `pipeline/subtitle_generator.py` | SRT generation | `generate_storybook_srt()` adapted (same format_time logic) |
| `utils/audio_utils.py` | Audio mixing | `mix_tracks()`, `normalize_audio()`, `pad_or_trim()`, `load_audio()` |
| `utils/image_utils.py` | Image processing | Available for use anywhere |
| `config.py` → `_load_dotenv()` | .env loading | Copied into `StorybookConfig` |

### Replaced / Re-skinned

| Original | Replacement | Changes |
|---|---|---|
| `LyricsProcessor` | `StoryGenerator` | Same `_call_llm()` + `_detect_provider()` verbatim; new story prompts + JSON schema |
| `BackgroundGenerator` | `StorybookIllustrator` | Same SD loading pattern; fixed children's-book style prefix; seed locking |
| `MusicVideoOrchestrator` | `StorybookOrchestrator` | Same step/callback/fallback structure; page-based steps |
| `MusicVideoCompositor` | `VideoRenderer` | ~90% same MoviePy code; Ken Burns instead of beat cuts; no lip sync layer |
| `MusicVideoScript` + `MusicVideoScene` | `Storybook` + `StoryPage` | Simpler fields; no BPM/outfit/camera |
| `app.py` | `storybook_generator/app.py` | Story inputs (name, age, theme) instead of lyrics/genre/mood |
| `cli.py` | `storybook_generator/cli.py` | `--name`, `--age`, `--theme`, `--pages` args |


---

## 3. Data Flow

```
CLI / Gradio UI
     │
     ▼
StorybookOrchestrator.generate_storybook(child_name, age, theme, num_pages, language)
     │
     ├─ [1] StoryGenerator.generate()
     │       ↳ LLM API → JSON → Storybook(title, pages=[StoryPage(...)])
     │
     ├─ [2] StorybookIllustrator.generate_pages()
     │       ↳ SD text-to-image (local) → page_01.png … page_N.png
     │       ↳ Fallback: gradient image if SD unavailable
     │
     ├─ [3] NarrationGenerator.generate_audio()
     │       ↳ Chatterbox TTS per page → narration_page_01.wav …
     │       ↳ Measures duration → sets StoryPage.duration_s
     │
     ├─ [4] SoundEngine.generate_ambient_music()
     │       ↳ MusicGen → ambient_music.wav (loopable, mood-matched)
     │
     ├─ [5] SoundEngine.mix_all_pages()
     │       ↳ narration + ambient clip + SFX → mixed_page_01.wav …
     │
     ├─ [6] PDFRenderer.render()
     │       ↳ Pillow multi-page PDF: cover + story pages
     │
     ├─ [7] VideoRenderer.render()
     │       ↳ MoviePy / FFmpeg fallback
     │       ↳ Ken Burns zoom per page + dynamic chunked subtitles + page audio
     │       ↳ Uses robust PIL rendering for text (100% ImageMagick-free)
     │
     └─ [8] generate_storybook_srt()
             ↳ SRT file: dynamically chunked text timed to narration duration
```

---

## 4. Modules

### `pipeline/story_generator.py` — StoryGenerator

**Key method:**
```python
generator = StoryGenerator(api_key="hf_...", model="meta-llama/Llama-3.1-8B-Instruct")
storybook = generator.generate(child_name="Liam", age=5, theme="dinosaurs", num_pages=4, language="Spanish")
```

**LLM prompt strategy:**

```
System: You are an award-winning children's book author...
User:   Child's name: Liam | Age: 5 | Theme: dinosaurs | Pages: 4 | Language: Spanish
        → Return JSON with title, pages[]{text, illustration_prompt, mood, sound_effects}
```

**Provider routing**:
- `hf_...` → HuggingFace Inference API
- `sk-ant-...` → Anthropic Claude
- `gsk_...` + Groq URL → Groq
- local directory → Transformers pipeline
- anything else → OpenAI-compatible

---

### `pipeline/illustrator.py` — StorybookIllustrator

Generates one SD image per story page.

```python
illustrator = StorybookIllustrator(device="mps")
paths = illustrator.generate_pages(pages=storybook.pages, output_dir="outputs/images")
```

**Style prefix** (prepended to every prompt):
```
children's book illustration, soft pastel colors, Pixar style,
warm lighting, bedtime story style, kid-friendly, watercolor texture,
{page.illustration_prompt}
```

**Negative prompt:** `ugly, scary, dark, violent, adult content, realistic photo, ...`

**Seed locking:** Each page uses `story_seed + page_number` — re-running produces the same images for the same story.

**Fallback:** Gradient image with mood color palette + page text overlay.

---

### `pipeline/narration_generator.py` — NarrationGenerator

Thin wrapper around `MusicVideoAudioGenerator.generate_vocals_chatterbox()`.

```python
narrator = NarrationGenerator(device="mps", voice_reference_path="/path/to/voice.wav")
paths, durations = narrator.generate_audio(pages=storybook.pages, output_dir="outputs/audio/narration")
```

**Narrator prefix:** The narrator prefix (`"Read this slowly, warmly…"`) was originally prepended to each page's text as a TTS style hint. It has since been **removed from the input text** and replaced by Chatterbox's `exaggeration` / `cfg_weight` parameters, which control speaking style without risking the prefix being spoken aloud.

**Voice cloning:** Pass `voice_reference_path` (WAV/MP3, 10–30s) for Chatterbox voice cloning.

---

### `pipeline/sound_engine.py` — SoundEngine

Generates ambient music + mixes per-page audio.

```python
engine = SoundEngine(device="mps", sfx_dir="storybook_generator/assets/sfx")

# Generate ambient track
engine.generate_ambient_music(
    music_prompt="gentle ambient piano, lullaby, slow",
    total_duration_s=120,
    output_path="outputs/audio/ambient_music.wav",
)

# Mix per-page: narration + music + SFX
mixed_paths = engine.mix_all_pages(
    pages=storybook.pages,
    narration_paths=[...],
    ambient_music_path="outputs/audio/ambient_music.wav",
    output_dir="outputs/audio/mixed",
)
```

**Mood → Music map:**

| Mood | MusicGen Prompt |
|------|----------------|
| calm | soft piano melody, gentle, slow, peaceful |
| happy | cheerful xylophone, upbeat gentle, playful |
| magical | harp and bells, twinkling, fairy tale |
| adventure | light orchestral, hopeful, strings |
| sleepy | very slow ambient, lullaby, dreamy |

**Volume mix:** narration 100% · ambient 35% · SFX 25%

---

### `pipeline/pdf_renderer.py` — PDFRenderer

Pure-Pillow PDF (no external PDF library).

```python
renderer = PDFRenderer()
pdf_path = renderer.render(storybook=storybook, output_path="outputs/pdf/liam_story.pdf")
```

**Layout per page:**
```
┌─────────────────────────────────┐
│      Illustration (top 60%)     │  ← SD image or gradient fallback
├─────────────────────────────────┤
│    Story text (bottom 40%)      │  ← Wrapped text, large font
│    🌙 Page 2 🌙                 │  ← Mood icon + page number
└─────────────────────────────────┘
```

Saves as a real multi-page PDF via `Image.save(..., format="PDF", save_all=True, append_images=[...])`.

---

### `pipeline/video_renderer.py` — VideoRenderer

Renders narrated MP4 video.

```python
renderer = VideoRenderer(output_dir="outputs/video")
renderer.render(storybook=storybook, mixed_audio_paths=[...], output_path="outputs/video/story.mp4")
```

**Structure:**
1. Title card (4s) — dark background + title + child name
2. Per-page clips with Ken Burns zoom (slow 1.0→1.08 zoom) + dynamic text subtitle chunks (6 words at a time)
3. End card (3s) — "The End 🌙"
4. Crossfade transitions (0.5s) between pages

**Ken Burns implementation:**
```python
def make_frame(t):
    zoom = 1.0 + 0.08 * (t / duration)
    # resize → center crop → return frame
```

**Fallback:** FFmpeg-based rendering (identical to `MusicVideoCompositor._assemble_with_ffmpeg()`).

---

### `pipeline/subtitle_generator.py`

Adapted from original `pipeline/subtitle_generator.py`.

Each SRT entry = one short chunk (approx 6 words), proportionally timed across the page's narration duration:
```
1
00:00:00,000 --> 00:00:03,500
Once upon a time, Liam found

2
00:00:03,500 --> 00:00:07,000
a tiny dinosaur egg in the

3
00:00:07,000 --> 00:00:10,500
forest near his home.
```

---

### `utils/story_schemas.py`

Replaces `utils/data_schemas.py`.

```python
@dataclass
class StoryPage:
    page_number: int
    text: str
    illustration_prompt: str
    mood: str                   # calm | happy | magical | adventure | sleepy
    sound_effects: List[str]    # ["forest", "magic"]
    image_path: str = ""
    narration_path: str = ""
    mixed_audio_path: str = ""
    duration_s: float = 0.0

@dataclass
class Storybook:
    title: str
    child_name: str
    age: int
    theme: str
    pages: List[StoryPage]
    music_prompt: str
    total_duration_s: float = 0.0
    color_palette: str = "soft pastels, warm, dreamy"
    art_style: str = "Pixar-style children's book illustration"
```

---

## 5. Reused Modules

### `pipeline/music_generator.py` → `MusicVideoAudioGenerator`

Used by `NarrationGenerator` and `SoundEngine`. Not modified.

Key methods used:
- `generate_vocals_chatterbox(lyrics, vocal_style, output_path, voice_reference_path)` — TTS
- `generate_instrumental(prompt, duration_s, output_path)` — MusicGen ambient

### `utils/audio_utils.py`

Used by `SoundEngine` for:
- `load_audio(path, sr)` — load WAV/MP3
- `mix_tracks(t1, t2, vol1, vol2)` — combine tracks
- `pad_or_trim(audio, target_length)` — match lengths
- `normalize_audio(audio)` — prevent clipping

### `utils/image_utils.py`

Available for use throughout the new pipeline.

---

## 6. Configuration

### `storybook_generator/config.py` — `StorybookConfig`

All original fields preserved + story-specific additions:

```python
@dataclass
class StorybookConfig:
    # API keys (unchanged from MusicVideoConfig)
    anthropic_api_key, openai_api_key, hf_token, groq_api_key

    # LLM routing (unchanged)
    llm_base_url, llm_model

    # Hardware (unchanged)
    device, output_dir

    # Video (unchanged, used by VideoRenderer)
    video_width=1280, video_height=720, fps=24

    # Audio volumes (adjusted: quieter music vs narration)
    narration_volume=1.0
    music_volume=0.35     # was 0.6 in music video (narration is primary)
    sfx_volume=0.25

    # Models (unchanged)
    musicgen_model, sd_model

    # Story-specific (new)
    default_pages=4, max_pages=8, min_pages=2
    sd_style_prefix="children's book illustration, soft pastel colors, Pixar style..."
    narration_speed_hint="Speak slowly, warmly, and clearly like a bedtime storyteller."
```

### Environment Variables

Same `.env` file as the original:

```env
HF_TOKEN=hf_...
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
DEVICE=mps
OUTPUT_DIR=outputs
```

---

## 7. Usage

### CLI

```bash
# From week-15/ directory

# Basic (auto-loads API key from .env)
python storybook_generator/cli.py \
    --name Liam --age 5 \
    --theme "dinosaurs and friendship" \
    --language English

# More pages, explicit Groq
python storybook_generator/cli.py \
    --name Sofia --age 7 \
    --theme "a princess who loves science" \
    --pages 6 \
    --model llama-3.1-8b-instant \
    --api-key $GROQ_API_KEY \
    --base-url https://api.groq.com/openai/v1

# Voice-cloned narration (Chatterbox)
python storybook_generator/cli.py \
    --name Ben --age 4 \
    --theme "bunny on the moon" \
    --voice-ref /path/to/parent_voice.wav
```

### Gradio UI

```bash
python storybook_generator/app.py
# → opens at http://localhost:7860
```

### Python API

```python
from storybook_generator.pipeline.orchestrator import StorybookOrchestrator

orch = StorybookOrchestrator(
    api_key="gsk_...",
    device="mps",
    output_dir="outputs",
    llm_base_url="https://api.groq.com/openai/v1",
    llm_model="llama-3.1-8b-instant",
)

result = orch.generate_storybook(
    child_name="Liam",
    age=5,
    theme="dinosaurs and friendship",
    num_pages=4,
    language="English",
)

print(result["pdf_path"])    # outputs/pdf/liam_s_adventure.pdf
print(result["video_path"])  # outputs/video/liam_s_adventure.mp4
```

---

## 8. Output Structure

```
outputs/
├── images/
│   ├── page_01.png    ← SD illustration (or gradient fallback)
│   ├── page_02.png
│   └── …
│
├── audio/
│   ├── narration/
│   │   ├── narration_page_01.wav   ← Chatterbox TTS
│   │   ├── narration_page_02.wav
│   │   └── …
│   ├── mixed/
│   │   ├── mixed_page_01.wav       ← narration + music + SFX
│   │   └── …
│   └── ambient_music.wav           ← MusicGen ambient loop
│
├── pdf/
│   └── liam_s_adventure.pdf        ← Pillow multi-page PDF
│
└── video/
    ├── liam_s_adventure.mp4        ← Final narrated video
    └── liam_s_adventure.srt        ← SRT subtitles
```

---

## 9. Extending the System

### Add a new SFX

1. Drop a `.wav` into `storybook_generator/assets/sfx/`
2. Add an entry to `SFX_MAP` in `sound_engine.py`
3. The LLM will start using the keyword in `sound_effects` lists automatically

### Add a new mood

1. Add to `MOOD_TO_MUSIC_PROMPT` in `sound_engine.py`
2. Add to `MOOD_COLORS` in `illustrator.py` and `video_renderer.py`
3. Update the story prompt in `story_generator.py` to list the new mood as an option

### Change illustration style

Edit `StorybookIllustrator.STYLE_PREFIX` in `illustrator.py`. Examples:
- `"watercolor painting, children's book, Beatrix Potter style, ..."`
- `"flat vector art, bold colors, Dr. Seuss style, ..."`
- `"oil painting, storybook, golden age illustration style, ..."`

### Use a different TTS engine

1. In `narration_generator.py`, replace the `generate_vocals_chatterbox()` call
2. The rest of the pipeline (mixing, video, PDF) is unchanged

### Add IP-Adapter for character consistency

In `illustrator.py`, load an IP-Adapter pipeline instead of plain SD:
```python
from diffusers import StableDiffusionPipeline
# → StableDiffusionImg2ImgPipeline + ip_adapter_image=reference_face
```
All other pipeline steps remain the same.

---

## 10. Evaluation Suite

`storybook_generator/evaluate.py` — an automated benchmark that runs the full story-generation and narration pipeline and reports four quality dimensions.

### Running the evaluator

```bash
# From the week-15/ directory (with conda env active)
python storybook_generator/evaluate.py \
    --name Lily --age 5 \
    --theme "a little dragon who learns to share" \
    --pages 3 \
    --output-dir outputs
```

CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | `Lily` | Child's name |
| `--age` | `5` | Child's age |
| `--theme` | `a little dragon who learns to share` | Story theme |
| `--pages` | `2` | Number of pages to generate |
| `--output-dir` | `outputs` | Where to save WAVs and the JSON report |

### What is measured

#### 1. LLM Latency
Wall-clock time (via `time.perf_counter()`) from the moment `StoryGenerator.generate()` is called until the fully-parsed `Storybook` object is returned.

#### 2. TTS Latency (per page)
Per-page wall-clock time for `NarrationGenerator` (Chatterbox TTS) to produce a WAV file, measured individually then averaged.

#### 3. Word Error Rate (WER)
Each generated narration WAV is transcribed with **OpenAI Whisper** (`tiny` model, English, fp16=False). The transcript is compared to the original page text using **jiwer**:

```python
wer_val = jiwer.wer(
    re.sub(r"[^\w\s]", "", reference.lower()),
    re.sub(r"[^\w\s]", "", hypothesis.lower()),
)
```

Punctuation is stripped from both sides before comparison. A WER of 0.0 = perfect; 1.0 = entirely wrong.

#### 4. TTS Accuracy
`accuracy = max(0.0, 1 − WER)` — expressed as a percentage. This is the complement of WER and represents the fraction of words the TTS engine reproduced correctly.

#### 5. Story Quality Score (0–10)

Five binary checks, 2 points each:

| Check | Criterion |
|-------|-----------|
| **Child name in story** | Child's name appears anywhere in the concatenated page texts |
| **Theme reflected** | At least one theme keyword (>3 chars) appears in the story text |
| **Correct page count** | `len(storybook.pages) == num_pages` |
| **Sentence length OK** | Every page has 2–5 sentences (split on `.!?`) |
| **Bedtime ending** | Last page's mood is `calm`, `sleepy`, or `happy` |

### Output files

```
outputs/
├── eval_narration/
│   ├── eval_narration_page_01.wav    ← Chatterbox TTS output (for Whisper)
│   └── …
└── evaluation_report.json            ← Full machine-readable results
```

### `evaluation_report.json` schema

```json
{
  "config": {
    "child_name": "Lily",
    "age": 5,
    "theme": "a little dragon who learns to share",
    "num_pages": 3
  },
  "latency": {
    "llm_story_generation_s": 2.31,
    "tts_per_page_s": [18.4, 21.1, 19.8],
    "tts_avg_s": 19.77
  },
  "wer_per_page": [
    {
      "page": 1,
      "reference": "Lily the little dragon loved to collect shiny rocks...",
      "hypothesis": "Lily the little dragon loved to collect shiny rocks...",
      "wer": 0.08,
      "accuracy": 92.0,
      "insertions": 0,
      "deletions": 0,
      "substitutions": 1,
      "alignment": "REF: lily the little dragon loved to collect shiny rocks\nHYP: lily the little dragon loved to collect tiny rocks\n                                           S   \n"
    }
  ],
  "accuracy_per_page": [92.0, 85.5, 88.2],
  "summary_quality": {
    "checks": {
      "child_name_in_story": true,
      "theme_reflected": true,
      "correct_page_count": true,
      "sentence_length_ok": true,
      "bedtime_ending": true
    },
    "score": 10,
    "max_score": 10
  },
  "aggregate": {
    "avg_wer": 0.112,
    "avg_accuracy_pct": 88.8,
    "avg_tts_latency_s": 19.77,
    "llm_latency_s": 2.31,
    "story_quality_score": 10,
    "story_quality_max": 10
  }
}
```

### Console summary (printed at end of run)

```
============================================================
  EVALUATION RESULTS SUMMARY
============================================================
  Metric                                   Value
  ---------------------------------------------
  LLM Story Generation Latency              1.3s
  Avg TTS (Narration) Latency/page         66.4s
  Average WER                              1.0%
  Average TTS Accuracy                     99.0%
  Story Quality Score                       10/10
============================================================
```

If WER > 0, the script will also print the specific errors during the run:
```
  Page 1: WER=8.0%  Accuracy=92.0%
    Errors: 1 subs, 0 ins, 0 dels
    Alignment:
      REF: lily the little dragon loved to collect shiny rocks
      HYP: lily the little dragon loved to collect tiny rocks
                                                   S   
```

### Dependencies

The evaluation script requires two additional packages (not needed for normal pipeline runs):

```bash
pip install openai-whisper jiwer
```

### Interpreting results

| Metric | Target | Notes |
|--------|--------|-------|
| LLM Latency | < 5s | Groq typically returns in 1–3s; HF Inference may be slower |
| TTS Latency/page | < 30s | Depends on text length and hardware (MPS/CUDA faster than CPU) |
| Average WER | < 15% | Whisper `tiny` introduces its own transcription error; use `base` for stricter evaluation |
| Average Accuracy | > 85% | Acceptable for production narration |
| Story Quality | ≥ 8/10 | Scores below 6 suggest the LLM prompt or model needs adjustment |
