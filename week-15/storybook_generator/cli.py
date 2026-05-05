#!/usr/bin/env python3
"""
cli.py — AI Children's Story Creator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Command-line interface mirroring the original cli.py structure.

Usage:
    python storybook_generator/cli.py \\
        --name Liam --age 5 \\
        --theme "dinosaurs and friendship" \\
        --pages 4

    # With voice cloning
    python storybook_generator/cli.py \\
        --name Sofia --age 6 \\
        --theme "a princess who loves science" \\
        --voice-ref /path/to/mum_voice.wav

Environment variables (same as original, loaded from .env):
    HF_TOKEN=hf_...                    # HuggingFace Inference API
    GROQ_API_KEY=gsk_...               # Groq (free tier, fast)
    ANTHROPIC_API_KEY=sk-ant-...       # Anthropic Claude
    OPENAI_API_KEY=sk-...              # OpenAI or compatible
    LLM_BASE_URL=https://...           # Custom endpoint
    LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
    DEVICE=cuda
"""

import argparse
import os
import sys

# Allow running from project root or from within storybook_generator/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storybook_generator.config import StorybookConfig, _load_dotenv
_load_dotenv()

from storybook_generator.pipeline.orchestrator import StorybookOrchestrator


def main():
    cfg = StorybookConfig.from_env()

    parser = argparse.ArgumentParser(
        description="AI Children's Story Creator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic story (HF_TOKEN auto-loaded from .env)
  python storybook_generator/cli.py --name Liam --age 5 --theme "dinosaurs and friendship"

  # More pages, custom model
  python storybook_generator/cli.py --name Sofia --age 7 \\
      --theme "a girl who talks to stars" --pages 6 \\
      --model llama-3.1-8b-instant --api-key $GROQ_API_KEY \\
      --base-url https://api.groq.com/openai/v1

  # Voice-cloned narration
  python storybook_generator/cli.py --name Ben --age 4 \\
      --theme "bunny on the moon" --voice-ref /path/to/dad_voice.wav
        """
    )

    # Story inputs
    parser.add_argument("--name",    "-n", default="Liam", help="Child's first name")
    parser.add_argument("--age",     "-a", type=int, default=5, help="Child's age (e.g. 5)")
    parser.add_argument("--theme",   "-t", default="a brave knight and a friendly dragon", help="Story idea / theme")
    parser.add_argument("--pages",   "-p", type=int, default=2,  help="Number of pages (2–8)")

    # LLM settings
    parser.add_argument("--api-key",  default=cfg.best_api_key(), help="LLM API key (auto from env)")
    parser.add_argument("--base-url", default=cfg.llm_base_url,  help="LLM base URL")
    parser.add_argument("--model",    default=cfg.llm_model,     help="LLM model name")

    # Hardware
    parser.add_argument("--device",  default=cfg.device,         help="Device: cuda | mps | cpu")
    parser.add_argument("--output",  "-o", default=cfg.output_dir, help="Output directory")

    # Voice cloning
    parser.add_argument(
        "--voice-ref", default=None,
        help="Path to reference WAV/MP3 for Chatterbox voice cloning (optional)"
    )

    args = parser.parse_args()

    # Clamp pages
    num_pages = max(2, min(args.pages, 8))

    # Resolve API key: explicit flag → env auto-detect
    api_key = args.api_key or cfg.best_api_key()
    if not api_key:
        print("⚠️  No API key found. Set HF_TOKEN, GROQ_API_KEY, or ANTHROPIC_API_KEY in .env")
        print("   Continuing — some features may fail.\n")

    # Show config
    print("📖 AI Children's Story Creator")
    print(f"   Child:  {args.name}, age {args.age}")
    print(f"   Theme:  {args.theme}")
    print(f"   Pages:  {num_pages}")
    print(f"   Model:  {args.model}")
    print(f"   Device: {args.device}")
    print(f"   Output: {args.output}")
    if args.voice_ref:
        print(f"   Voice:  {args.voice_ref}")
    print()

    # Progress bar (same as original cli.py)
    def on_progress(msg: str, pct: float):
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r[{bar}] {pct:5.1f}%  {msg:<50}", end="", flush=True)

    # Run
    orch = StorybookOrchestrator(
        api_key=api_key,
        device=args.device,
        output_dir=args.output,
        llm_base_url=args.base_url if args.base_url else None,
        llm_model=args.model,
        voice_reference_path=args.voice_ref,
    )

    result = orch.generate_storybook(
        child_name=args.name,
        age=args.age,
        theme=args.theme,
        num_pages=num_pages,
        progress_cb=on_progress,
    )

    print("\n")
    print("✅ Storybook generated!")
    print(f"   📄 PDF:       {result['pdf_path']}")
    print(f"   🎬 Video:     {result['video_path']}")
    print(f"   📝 Subtitles: {result['subtitle_path']}")
    print(f"   🎵 Music:     {result['ambient_path']}")
    sb = result["storybook"]
    print(f"   📖 Title:     {sb.title}")
    print(f"   📑 Pages:     {len(sb.pages)}")
    print(f"   ⏱  Duration:  {sb.total_duration_s:.0f}s")


if __name__ == "__main__":
    main()
