# Narrate.AI 🎬

> *Transform unstructured personal media (photos, videos, voice memos) into a cinematic documentary-style video with a coherent narrative, emotional arc, AI-generated narration, adaptive music, and ambient soundscapes.*

**Narrate.AI** is a modular AI pipeline that acts as your personal documentary filmmaker. You simply upload unordered photos, videos, and rough voice memos, and the system automatically handles scene analysis, transcription, story arc construction, voice cloning, music scoring, ambient design, and final video assembly.

---

## 🛠 Libraries and Tools Used

Narrate.AI leverages a suite of state-of-the-art open-source and API-based models to handle multimodal processing:

### Core Pipeline Models
*   **Visual Understanding**: [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) — Extracts scene types, emotion tags, objects, and visual salience scores from images and video frames.
*   **Audio Understanding (ASR)**: [Whisper-small](https://huggingface.co/openai/whisper-small) — Transcribes raw voice memos and rough audio into timestamped text.
*   **Narrative Engine**: [Groq API] — Constructs a cohesive 3-act documentary script, complete with emotional pacing and media ordering.
*   **Voice Synthesis (TTS)**: [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI) — Synthesizes natural, emotional narration with support for **zero-shot voice cloning** (requires a reference `.wav` file).
*   **Adaptive Music Generation**: [MusicGen-small](https://huggingface.co/facebook/musicgen-small) — Generates emotion-conditioned background music tailored to each specific scene/beat.
*   **Ambient Soundscapes**: [AudioLDM2](https://huggingface.co/cvssp/audioldm2) — Generates scene-specific environmental audio (e.g., "crashing waves", "indoor crowd murmurs").

### Engineering & UI Frameworks
*   **Video Assembly**: [MoviePy](https://zulko.github.io/moviepy/) (powered by FFmpeg) — Renders the final MP4 with Ken Burns effects, crossfades, and a normalized 3-layer audio mix.
*   **Web Interface**: [Streamlit](https://streamlit.io/) — Powers the sleek, 4-step interactive wizard UI.
*   **Deep Learning Backend**: PyTorch and Hugging Face `transformers` / `diffusers`.

---

## 🚀 Installation & Setup

### Prerequisites
*   **Python 3.10+** (A virtual environment or Conda environment is highly recommended)
*   **FFmpeg** installed on your system:
    *   Mac: `brew install ffmpeg`
    *   Linux: `sudo apt install ffmpeg`
*   At least 8GB RAM (16GB+ recommended to run all ML models locally)
*   *(Optional but Recommended)* A GPU or Apple Silicon (M1/M2/M3) for accelerated inference.

### Step-by-Step Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone https://github.com/sayushmaharjan/CS-5588.git
   cd week-14
   ```

2. **Install the required Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: PyTorch will be installed automatically, but you may want to install the specific GPU-accelerated version for your system from the [PyTorch website](https://pytorch.org/).)*

3. **Set up your API Keys (Optional but required for the LLM Narrative Engine):**
   ```bash
   export GROQ_API_KEY="gsk_..." 
   ```
   *If no API key is provided, the system gracefully falls back to a built-in 5-beat template script.*

---

## 🎮 Running the Application

### Option 1: Streamlit Web UI (Recommended)
Launch the interactive web application to use the 4-step documentary wizard:

```bash
streamlit run app.py
```
*   The application will open in your browser at `http://localhost:8501`.
*   You can upload your media, tweak music/ambient volumes, supply a voice clone reference, and preview the generated script before final rendering.

### Option 2: Command Line Pipeline
You can bypass the UI and run the orchestrator directly from a python script:

```python
from pipeline.orchestrator import run_pipeline

result = run_pipeline(
    photo_video_paths = ["beach.jpg", "wedding.mp4"],
    audio_paths       = ["voice_memo.m4a"],
    voice_reference_path = "my_voice.wav", # Optional for cloning
    output_dir        = "my_documentary_output",
    api_key           = "sk-ant-...",
    music_mode        = "segment"
)
print(f"Finished rendering: {result['output_video_path']}")
```

---

## 📂 Sample Outputs

When the pipeline completes a run, it generates a structured set of files. Here is what you can expect in your output directory:

```text
outputs/
├── documentary.mp4             # 🎬 The final rendered documentary video (H.264/AAC)
├── documentary_script.json     # 📜 The Claude-generated 3-act narrative script metadata
└── temp/
    ├── narration/
    │   ├── beat_001_narration.wav  # Voice-cloned TTS for Scene 1
    │   └── beat_002_narration.wav  # Voice-cloned TTS for Scene 2
    ├── music/
    │   ├── beat_001_music.wav      # Adaptive MusicGen score for Scene 1
    │   └── beat_002_music.wav      # Adaptive MusicGen score for Scene 2
    └── ambient/
        ├── beat_001_ambient.wav    # AudioLDM2 environmental soundscape
        └── beat_002_ambient.wav    # AudioLDM2 environmental soundscape
```

### Evaluation Output (Baseline vs. Improved)
The project also includes an evaluation script (`evaluate_speech.py`) to measure the performance delta between the baseline models and the improved AI models. 

**Sample Run (`python evaluate_speech.py`):**
```text
----------------------------------------
📊 EVALUATION RESULTS
----------------------------------------
• Transcription Quality : Excellent
• Latency (ASR)         : 13.21s
• Accuracy              : 100.00%
• WER (Word Error Rate) : 0.00% (0 errors / 24 words)
• Voice Consistency     : 0.897 SIM-R (Target: >0.85)
• Prompt Align. (CLAP)  : 0.545 Score (Target: >0.45)
----------------------------------------
```

---
*Built with ❤️ using CLIP, Whisper, Claude, Chatterbox TTS, MusicGen, and AudioLDM2.*
