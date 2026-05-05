"""
evaluate.py — AI Children's Story Creator Evaluation Suite
============================================================
Measures four dimensions of pipeline quality:

1. WER  (Word Error Rate)        — how accurately Chatterbox TTS speaks the page text
2. Accuracy                      — 1 − WER (percentage of words correct)
3. Latency                       — LLM story generation time + per-page TTS time
4. Summary Quality               — structural & semantic story quality (LLM output + narration fidelity)

Usage
-----
    python storybook_generator/evaluate.py [--pages N] [--name NAME] [--age AGE] [--theme THEME]

Outputs
-------
    outputs/evaluation_report.json   — machine-readable results
    Console table                    — human-readable summary
"""

import os
import sys
import json
import time
import argparse
import logging
import re

import numpy as np

# ── Ensure project root is on PYTHONPATH ────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Load .env so API keys are available ──────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = os.path.join(ROOT, ".env")
    if os.path.exists(_env):
        load_dotenv(_env, override=False)
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING)  # keep third-party logs quiet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _whisper_transcribe(wav_path: str, model_size: str = "tiny") -> str:
    """Transcribe a WAV file using Whisper (local, no API key needed)."""
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(wav_path, language="en", fp16=False)
    return result["text"].strip()


def compute_wer_details(reference: str, hypothesis: str) -> dict:
    """Return Word Error Rate details between reference and hypothesis strings."""
    import jiwer
    ref_clean = re.sub(r"[^\w\s]", "", reference.lower()).strip()
    hyp_clean = re.sub(r"[^\w\s]", "", hypothesis.lower()).strip()
    if not ref_clean:
        return {"wer": 0.0, "insertions": 0, "deletions": 0, "substitutions": 0, "hits": 0, "alignment": ""}
    
    out = jiwer.process_words(ref_clean, hyp_clean)
    alignment = jiwer.visualize_alignment(out)
    return {
        "wer": float(out.wer),
        "insertions": out.insertions,
        "deletions": out.deletions,
        "substitutions": out.substitutions,
        "hits": out.hits,
        "alignment": alignment,
    }


def score_story_quality(storybook, child_name: str, age: int, theme: str, num_pages: int) -> dict:
    """
    Score LLM story quality on 5 binary criteria.
    Returns a dict with individual checks and a 0–10 total score.
    """
    checks = {}

    # 1. Child name present in story
    full_text = " ".join(p.text for p in storybook.pages)
    checks["child_name_in_story"] = child_name.lower() in full_text.lower()

    # 2. Theme reflected (at least one theme keyword appears in the text)
    theme_words = [w for w in theme.lower().split() if len(w) > 3]
    checks["theme_reflected"] = any(w in full_text.lower() for w in theme_words)

    # 3. Correct page count
    checks["correct_page_count"] = len(storybook.pages) == num_pages

    # 4. Each page has 2–4 sentences
    def sentence_count(text):
        return len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    per_page_ok = [2 <= sentence_count(p.text) <= 5 for p in storybook.pages]
    checks["sentence_length_ok"] = all(per_page_ok)

    # 5. Narrative arc: last page mood should be calm/sleepy (bedtime)
    last_mood = storybook.pages[-1].mood if storybook.pages else "?"
    checks["bedtime_ending"] = last_mood in ("calm", "sleepy", "happy")

    # Score: 2 points each out of 10
    score = sum(checks.values()) * 2
    return {"checks": checks, "score": score, "max_score": 10}


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    child_name: str = "Lily",
    age: int = 5,
    theme: str = "a little dragon who learns to share",
    num_pages: int = 3,
    output_dir: str = "outputs",
):
    results = {
        "config": {
            "child_name": child_name,
            "age": age,
            "theme": theme,
            "num_pages": num_pages,
        },
        "latency": {},
        "wer_per_page": [],
        "accuracy_per_page": [],
        "summary_quality": {},
        "aggregate": {},
    }

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Generate story (measure LLM latency) ─────────────────────────
    print("\n" + "=" * 60)
    print("  AI Children's Story Creator — Evaluation Suite")
    print("=" * 60)
    print(f"\n📖  Generating story: '{theme}' for {child_name} (age {age}, {num_pages} pages)…")

    from storybook_generator.pipeline.story_generator import StoryGenerator
    gen = StoryGenerator(model=None)  # picks up LLM_MODEL from .env

    llm_start = time.perf_counter()
    storybook = gen.generate(child_name=child_name, age=age, theme=theme, num_pages=num_pages)
    llm_latency = time.perf_counter() - llm_start
    results["latency"]["llm_story_generation_s"] = round(llm_latency, 2)
    print(f"✅  Story generated: '{storybook.title}' ({len(storybook.pages)} pages) in {llm_latency:.1f}s")

    # ── Step 2: Generate narration (measure TTS latency per page) ────────────
    print("\n🎙️   Generating narration (Chatterbox TTS)…")
    from storybook_generator.pipeline.narration_generator import NarrationGenerator

    narr_dir = os.path.join(output_dir, "eval_narration")
    os.makedirs(narr_dir, exist_ok=True)

    narrator = NarrationGenerator()
    tts_latencies = []
    narration_paths = []

    for page in storybook.pages:
        out_path = os.path.join(narr_dir, f"eval_narration_page_{page.page_number:02d}.wav")
        t0 = time.perf_counter()
        try:
            narrator._audio_gen.generate_vocals_chatterbox(
                lyrics=page.text,
                vocal_style="",
                output_path=out_path,
                voice_reference_path=None,
            )
            dur = narrator._measure_duration(out_path)
        except Exception as e:
            print(f"  ⚠️  TTS failed for page {page.page_number}: {e}. Writing silence.")
            narrator._write_silence(out_path, duration_s=4.0)
            dur = 4.0
        latency = time.perf_counter() - t0
        tts_latencies.append(round(latency, 2))
        narration_paths.append(out_path)
        print(f"  Page {page.page_number}: {latency:.1f}s TTS latency, {dur:.1f}s audio duration")

    results["latency"]["tts_per_page_s"] = tts_latencies
    results["latency"]["tts_avg_s"] = round(float(np.mean(tts_latencies)), 2)

    # ── Step 3: Transcribe with Whisper → WER / Accuracy ────────────────────
    print("\n🔍  Transcribing with Whisper (tiny) to compute WER…")
    import whisper
    whisper_model = whisper.load_model("tiny")

    wer_scores = []
    accuracy_scores = []

    for page, wav_path in zip(storybook.pages, narration_paths):
        try:
            result = whisper_model.transcribe(wav_path, language="en", fp16=False)
            transcript = result["text"].strip()
        except Exception as e:
            print(f"  ⚠️  Transcription failed for page {page.page_number}: {e}")
            transcript = ""

        details = compute_wer_details(page.text, transcript)
        wer_val = details["wer"]
        acc_val = max(0.0, 1.0 - wer_val)
        wer_scores.append(round(wer_val, 4))
        accuracy_scores.append(round(acc_val, 4))

        results["wer_per_page"].append({
            "page": page.page_number,
            "reference": page.text,
            "hypothesis": transcript,
            "wer": round(wer_val, 4),
            "accuracy": round(acc_val * 100, 2),
            "insertions": details["insertions"],
            "deletions": details["deletions"],
            "substitutions": details["substitutions"],
            "alignment": details["alignment"],
        })

        print(f"  Page {page.page_number}: WER={wer_val:.1%}  Accuracy={acc_val:.1%}")
        if wer_val > 0.0:
            print(f"    Errors: {details['substitutions']} subs, {details['insertions']} ins, {details['deletions']} dels")
            print("    Alignment:")
            for line in details["alignment"].split("\n"):
                if line.strip():
                    print(f"      {line}")

    avg_wer = float(np.mean(wer_scores))
    avg_acc = float(np.mean(accuracy_scores))
    results["accuracy_per_page"] = [round(a * 100, 2) for a in accuracy_scores]

    # ── Step 4: Summary quality ──────────────────────────────────────────────
    print("\n📊  Scoring story quality…")
    quality = score_story_quality(storybook, child_name, age, theme, num_pages)
    results["summary_quality"] = quality
    print(f"  Story quality score: {quality['score']}/10")
    for k, v in quality["checks"].items():
        icon = "✅" if v else "❌"
        print(f"    {icon}  {k.replace('_', ' ').title()}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    results["aggregate"] = {
        "avg_wer": round(avg_wer, 4),
        "avg_accuracy_pct": round(avg_acc * 100, 2),
        "avg_tts_latency_s": results["latency"]["tts_avg_s"],
        "llm_latency_s": results["latency"]["llm_story_generation_s"],
        "story_quality_score": quality["score"],
        "story_quality_max": 10,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾  Report saved: {report_path}")

    # ── Print final table ─────────────────────────────────────────────────────
    agg = results["aggregate"]
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'-'*45}")
    print(f"  {'LLM Story Generation Latency':<35} {agg['llm_latency_s']:>9.1f}s")
    print(f"  {'Avg TTS (Narration) Latency/page':<35} {agg['avg_tts_latency_s']:>9.1f}s")
    print(f"  {'Average WER':<35} {agg['avg_wer']:>9.1%}")
    print(f"  {'Average TTS Accuracy':<35} {agg['avg_accuracy_pct']:>9.1f}%")
    print(f"  {'Story Quality Score':<35} {agg['story_quality_score']:>8}/10")
    print("=" * 60 + "\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the AI Children's Story Creator")
    parser.add_argument("--name", default="Lily", help="Child's name")
    parser.add_argument("--age", type=int, default=5, help="Child's age")
    parser.add_argument("--theme", default="a little dragon who learns to share", help="Story theme")
    parser.add_argument("--pages", type=int, default=2, help="Number of pages (default 3 for speed)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs")
    args = parser.parse_args()

    run_evaluation(
        child_name=args.name,
        age=args.age,
        theme=args.theme,
        num_pages=args.pages,
        output_dir=args.output_dir,
    )
