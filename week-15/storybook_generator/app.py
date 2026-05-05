"""
app.py — AI Children's Story Creator (Gradio UI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replaces the original app.py music video UI with storybook inputs.

Inputs:    child name, age, story theme, num pages, voice reference (opt), LLM settings
Outputs:   video, PDF download, audio preview, status log
"""

import os
import sys
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storybook_generator.config import StorybookConfig, _load_dotenv
_load_dotenv()

from storybook_generator.pipeline.orchestrator import StorybookOrchestrator


# ---------------------------------------------------------------------------
# Generation function
# ---------------------------------------------------------------------------

def generate_storybook(
    child_name: str,
    child_age: int,
    story_theme: str,
    num_pages: int,
    voice_ref,             # gr.Audio upload → tmp path or None
    api_key: str,
    base_url: str,
    model: str,
    progress=gr.Progress(),
):
    """Main generation callback for the Gradio UI."""

    # Validation
    if not child_name.strip():
        return None, None, None, "❌ Please enter the child's name."
    if not story_theme.strip():
        return None, None, None, "❌ Please enter a story theme or idea."
    if not api_key.strip():
        return None, None, None, "❌ Please enter your API key (or set it in .env)."

    num_pages = max(2, min(int(num_pages), 8))

    # Resolve voice reference path
    voice_ref_path = voice_ref if (voice_ref and os.path.exists(voice_ref)) else None

    # Device detection
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"

    def progress_cb(msg: str, pct: float):
        progress(pct / 100, desc=msg)

    try:
        orch = StorybookOrchestrator(
            api_key=api_key,
            device=device,
            output_dir="outputs",
            llm_base_url=base_url if base_url.strip() else None,
            llm_model=model if model.strip() else None,
            voice_reference_path=voice_ref_path,
        )

        result = orch.generate_storybook(
            child_name=child_name.strip(),
            age=int(child_age),
            theme=story_theme.strip(),
            num_pages=num_pages,
            progress_cb=progress_cb,
        )

        sb = result["storybook"]

        status = (
            f"✅ Storybook complete!\n\n"
            f"📖 Title:   {sb.title}\n"
            f"👶 For:     {sb.child_name}, age {sb.age}\n"
            f"📑 Pages:   {len(sb.pages)}\n"
            f"⏱  Length:  {sb.total_duration_s:.0f} seconds\n\n"
            f"🎨 Style:  {sb.art_style}\n"
            f"🎵 Music:  {sb.music_prompt[:60]}…\n\n"
        )
        for page in sb.pages:
            status += f"  Page {page.page_number} [{page.mood}]: {page.text[:60]}…\n"

        video_out = result["video_path"] if os.path.exists(result.get("video_path", "")) else None
        pdf_out   = result["pdf_path"]   if os.path.exists(result.get("pdf_path", ""))   else None
        audio_out = result["ambient_path"] if os.path.exists(result.get("ambient_path", "")) else None

        return video_out, pdf_out, audio_out, status

    except Exception as e:
        import traceback
        return None, None, None, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

cfg = StorybookConfig.from_env()

EXAMPLE_THEMES = [
    "a tiny dragon who is afraid of fire",
    "dinosaurs learning to share",
    "a girl who talks to the stars",
    "a bunny who travels to the moon",
    "a robot who wants to make friends",
    "a lost puppy finds its way home",
]

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="📖 AI Children's Story Creator",
    css="""
    .gr-button-primary { background: linear-gradient(135deg, #7C3AED, #4F46E5) !important; }
    footer { display: none !important; }
    """
) as app:

    gr.Markdown("""
    # 📖 AI Children's Story Creator
    ### Turn a simple idea into a fully illustrated, narrated storybook — in minutes.
    *Powered by LLM story generation · Stable Diffusion illustrations · Chatterbox TTS narration · MusicGen ambient music*
    """)

    with gr.Row():

        # ── Left column: inputs ──────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 👶 About the Child")

            with gr.Row():
                child_name = gr.Textbox(
                    label="Child's Name",
                    placeholder="e.g. Liam",
                    value="",
                )
                child_age = gr.Slider(
                    label="Age",
                    minimum=2,
                    maximum=10,
                    step=1,
                    value=5,
                )

            gr.Markdown("### 📖 Story")

            story_theme = gr.Textbox(
                label="Story Idea / Theme",
                placeholder="e.g. 'dinosaurs and friendship' or 'a princess who loves science'",
                lines=2,
            )

            gr.Examples(
                examples=[[t] for t in EXAMPLE_THEMES],
                inputs=[story_theme],
                label="✨ Example themes",
            )

            num_pages = gr.Slider(
                label="Number of Pages",
                minimum=2,
                maximum=8,
                step=1,
                value=4,
            )

            gr.Markdown("### 🎙️ Narrator Voice (optional)")

            voice_ref = gr.Audio(
                label="Upload a voice reference WAV/MP3 for voice cloning (leave empty for default)",
                type="filepath",
                sources=["upload"],
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                api_key = gr.Textbox(
                    label="LLM API Key",
                    type="password",
                    placeholder="hf_... | gsk_... | sk-ant-... | sk-...",
                    value=cfg.best_api_key(),
                )
                base_url = gr.Textbox(
                    label="LLM Base URL (optional)",
                    placeholder="https://api.groq.com/openai/v1 | http://localhost:11434/v1",
                    value=cfg.llm_base_url or "",
                )
                model = gr.Textbox(
                    label="LLM Model",
                    placeholder="meta-llama/Llama-3.1-8B-Instruct | llama-3.1-8b-instant",
                    value=cfg.llm_model,
                )

            generate_btn = gr.Button(
                "✨ Create Storybook",
                variant="primary",
                size="lg",
            )

        # ── Right column: outputs ────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Your Storybook")

            status = gr.Textbox(
                label="Generation Log",
                interactive=False,
                lines=12,
            )

            with gr.Tabs():
                with gr.Tab("🎬 Video"):
                    output_video = gr.Video(label="Narrated Storybook Video")

                with gr.Tab("📄 PDF"):
                    output_pdf = gr.File(label="Download PDF Storybook")

                with gr.Tab("🎵 Music Preview"):
                    output_audio = gr.Audio(label="Ambient Background Music")

    generate_btn.click(
        fn=generate_storybook,
        inputs=[
            child_name, child_age, story_theme, num_pages,
            voice_ref, api_key, base_url, model,
        ],
        outputs=[output_video, output_pdf, output_audio, status],
    )

    gr.Markdown("""
    ---
    **Tips:**
    - Use a specific child's name for a more personal story
    - Groq (free tier) with `llama-3.1-8b-instant` is the fastest option
    - For voice cloning, upload a 10–30s clear WAV of the narrator's voice
    - Illustrations require a GPU with Stable Diffusion; CPU falls back to gradient images
    """)

if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0")
