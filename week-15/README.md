# Story Weaver: AI Children's Story Creator

A production-ready, end-to-end AI pipeline that transforms a **child's name, age, and story theme** into a fully narrated storybook — with LLM-generated story text, Stable Diffusion illustrations, Chatterbox TTS narration, AI ambient music, a multi-page PDF, and a narrated MP4 video.

---

## System Overview

```
INPUT:                                 OUTPUT:
- Child's name                         - Multi-page PDF storybook
- Age                                  - Narrated MP4 video
- Story theme                          - Per-page SRT subtitles
- Number of pages (2–8)               - Chatterbox TTS narration WAVs
- (Optional) voice reference WAV       - SD illustrations per page
                                       - Ambient background music
                                       - evaluation_report.json
```

---

## Architecture

```
StoryGenerator (LLM) ──────────────────────────────────────────────────────┐
     │  JSON: title, pages[text, illustration_prompt, mood, sound_effects]  │
     ▼                                                                      │
StorybookIllustrator (Stable Diffusion) ──────────────────────────────────┤
     │  page_01.png … page_N.png (Pixar-style children's book)             │
     ▼                                                                      │
NarrationGenerator (Chatterbox TTS) ───────────────────────────────────────┤
     │  narration_page_01.wav … (per-page, timed durations)                │
     ▼                                                                      │
SoundEngine (MusicGen + SFX mixer) ────────────────────────────────────────┤
     │  ambient_music.wav + mixed_page_01.wav …                            │
     ▼                                                                      │
PDFRenderer (Pillow) + VideoRenderer (MoviePy/FFmpeg) ─────────────────────┘
     │
     ▼
 story.pdf  +  story.mp4  +  story.srt
```

---

## File Structure

```
week-15/
├── README.md
├── requirements.txt
├── .env                            # API keys (HF_TOKEN, GROQ_API_KEY, etc.)
│
└── storybook_generator/
    ├── app.py                      # Gradio web UI (4-step wizard)
    ├── cli.py                      # Command-line interface
    ├── config.py                   # StorybookConfig dataclass + .env loader
    ├── evaluate.py                 # Automated evaluation suite (WER, latency, quality)
    ├── DOCUMENTATION.md            # Full technical documentation
    │
    ├── pipeline/
    │   ├── story_generator.py      # LLM → Storybook JSON (replaces LyricsProcessor)
    │   ├── illustrator.py          # SD text-to-image per page (replaces BackgroundGenerator)
    │   ├── narration_generator.py  # Chatterbox TTS per page (wraps music_generator.py)
    │   ├── sound_engine.py         # MusicGen ambient + SFX mixing
    │   ├── pdf_renderer.py         # Pillow multi-page PDF renderer
    │   ├── video_renderer.py       # MoviePy Ken Burns video + text overlays
    │   ├── subtitle_generator.py   # SRT generation (adapted from original)
    │   ├── music_generator.py      # Chatterbox TTS + MusicGen (reused unchanged)
    │   └── orchestrator.py         # End-to-end StorybookOrchestrator
    │
    ├── utils/
    │   └── story_schemas.py        # Storybook + StoryPage dataclasses
    │
    └── assets/
        └── sfx/                    # Sound effect WAVs (forest, magic, ocean, …)

outputs/
├── images/          # SD illustrations (page_01.png …)
├── audio/
│   ├── narration/   # Chatterbox TTS WAVs per page
│   ├── mixed/       # narration + ambient + SFX per page
│   └── ambient_music.wav
├── pdf/             # Multi-page PDF storybook
├── video/           # Narrated MP4 + SRT subtitles
└── evaluation_report.json
```

---

## Installation

### 1. Set Up Environment

```bash
cd week-15
conda activate my_env   # or your environment
pip install -r requirements.txt
```

### 2. Set API Keys

Create a `.env` file in the `week-15/` root:

```env
HF_TOKEN=hf_...
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...   # optional
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
DEVICE=mps                     # or cuda / cpu
OUTPUT_DIR=outputs
```

Supported LLM providers: **Groq**, **Anthropic Claude**, **HuggingFace Inference API**, **OpenAI-compatible** endpoints, and local Transformers models.

---

## Usage

### Web UI (Gradio)

```bash
python storybook_generator/app.py
# → http://localhost:7860
```

### Command Line

```bash
# Basic
python storybook_generator/cli.py \
    --name Lily --age 5 \
    --theme "a little dragon who learns to share"

# With voice cloning
python storybook_generator/cli.py \
    --name Ben --age 4 \
    --theme "bunny on the moon" \
    --pages 6 \
    --voice-ref /path/to/parent_voice.wav

# Explicit API
python storybook_generator/cli.py \
    --name Sofia --age 7 \
    --theme "a princess who loves science" \
    --pages 6 \
    --model llama-3.1-8b-instant \
    --api-key $GROQ_API_KEY \
    --base-url https://api.groq.com/openai/v1
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
    child_name="Lily",
    age=5,
    theme="a little dragon who learns to share",
    num_pages=4,
)

print(result["pdf_path"])    # outputs/pdf/lily_s_adventure.pdf
print(result["video_path"])  # outputs/video/lily_s_adventure.mp4
```

---

## Pipeline Steps

1. **Story Generation** — LLM produces structured JSON: title, per-page text, illustration prompts, moods, and sound effect tags.
2. **Illustration** — Stable Diffusion generates one Pixar-style children's book image per page. Fallback: mood-colored gradient.
3. **Narration** — Chatterbox TTS synthesizes per-page narration WAVs. Optional voice cloning via a reference WAV.
4. **Ambient Music** — MusicGen generates a mood-matched ambient loop (loopable, slow, lullaby-style).
5. **Audio Mixing** — Per-page final audio: narration (100%) + ambient (35%) + SFX (25%).
6. **PDF Rendering** — Pillow assembles a real multi-page PDF: illustration (top 60%) + text (bottom 40%) per page.
7. **Video Rendering** — MoviePy/FFmpeg fallback compiles a narrated MP4: title card → Ken Burns zoom per page with dynamic chunked subtitles → end card. No ImageMagick dependency!
8. **Subtitles** — Standalone SRT file generated with precise chunked timestamps matching the video's dynamic text.

---

## Evaluation Suite

Run the automated evaluation suite to benchmark pipeline quality:

```bash
python storybook_generator/evaluate.py \
    --name Lily --age 5 \
    --theme "a little dragon who learns to share" \
    --pages 3
```

The suite measures **4 dimensions** of quality:

| Metric | Method | Description |
|--------|--------|-------------|
| **WER** (Word Error Rate) | Whisper `tiny` → `jiwer` | How accurately Chatterbox TTS speaks each page's text |
| **TTS Accuracy** | `1 − WER` | Percentage of words correctly synthesized |
| **LLM Latency** | `time.perf_counter()` | Story generation wall-clock time (seconds) |
| **TTS Latency/page** | `time.perf_counter()` | Per-page narration synthesis time (seconds) |
| **Story Quality Score** | 5 binary checks × 2 pts | Structural and semantic quality of the LLM output (0–10) |

### Story Quality Checks (0–10 score)

| Check | Points | Criterion |
|-------|--------|-----------|
| Child name in story | 2 | Child's name appears in story text |
| Theme reflected | 2 | At least one theme keyword appears |
| Correct page count | 2 | LLM returned exactly the requested number of pages |
| Sentence length OK | 2 | Each page has 2–5 sentences |
| Bedtime ending | 2 | Last page mood is `calm`, `sleepy`, or `happy` |

### Sample Output

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

Results are also saved to `outputs/evaluation_report.json`.

---

## Key Features

- **LLM story generation** via Groq, Anthropic Claude, HuggingFace, or any OpenAI-compatible API
- **Stable Diffusion illustrations** with children's-book style prefix and seed locking for reproducibility
- **Chatterbox TTS narration** with optional voice cloning from a 10–30s reference WAV
- **AI ambient music** via MusicGen, mood-matched to the story's emotional arc
- **Per-page SFX mixing** (forest, magic, ocean, etc.) driven by LLM-tagged keywords
- **Multi-page PDF** rendered entirely with Pillow — no external PDF library needed
- **Dynamic Narrated Video** with Ken Burns zoom, fast PIL-based chunked subtitles, title card, and end card — **100% ImageMagick-free!**
- **Automated evaluation** — Exact WER (Insertions/Deletions/Substitutions), accuracy, latency, and story quality in one command
- **Graceful fallbacks** throughout — pipeline completes even if SD, MusicGen, or TTS fail
- **Seed-locked illustrations** — re-running the same story produces identical images

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token (for HF Inference API or local models) |
| `GROQ_API_KEY` | Groq API key (`gsk_...`) |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `LLM_BASE_URL` | LLM endpoint URL (default: Groq) |
| `LLM_MODEL` | Model name (default: `llama-3.1-8b-instant`) |
| `DEVICE` | `mps`, `cuda`, or `cpu` |
| `OUTPUT_DIR` | Output directory (default: `outputs`) |

---

## Troubleshooting

### "Chatterbox not installed" / TTS fails
The evaluator and narration generator will write silence files and continue. Install Chatterbox:
```bash
pip install chatterbox-tts
```

### "CUDA / MPS out of memory"
Set `DEVICE=cpu` in `.env`. SD and MusicGen will be slower but will run.

### "SD model not loading" / Gradient fallbacks
Stable Diffusion requires ~4 GB VRAM. On CPU, it will be slow. The pipeline produces gradient fallbacks automatically.

### Whisper not found (WER evaluation)
```bash
pip install openai-whisper
```

### jiwer not found (WER evaluation)
```bash
pip install jiwer
```

---

## License

MIT License. This project is for educational and research purposes.
