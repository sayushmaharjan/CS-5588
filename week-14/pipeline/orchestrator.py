"""
Cinematic Memory — Pipeline Orchestrator
Split into run_pipeline_pre() (analysis → audio) and run_pipeline_finalize() (video assembly).
New model:
  - Generates 3 script versions (different tones) — user picks one in Edit panel
  - Generates 1 single global music track per mood choice — no per-beat music
  - Supports target_duration_s to control video length
"""
from __future__ import annotations
import os, logging, time
from typing import List, Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)


def run_pipeline_pre(
    photo_video_paths:   List[str],
    audio_paths:         List[str],
    output_dir:          str = "outputs",
    event_hint:          str = "",
    user_script:         Optional[str] = None,
    music_mood:          str = "cinematic",
    ambient_scene:       str = "nature",
    target_duration_s:   float = 60.0,
    script_mode:         str = "cinematic",
    progress_cb:         Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """
    Pre-processing stage: visual analysis → transcription → 3 script versions.
    API key and provider are read directly from config module.
    Returns intermediate result dict for the Edit panel.
    """
    import config as cfg

    # Read API key and provider directly from config
    llm_provider = cfg.LLM_PROVIDER
    api_key      = None
    if llm_provider == "groq" and cfg.GROQ_API_KEY.strip():
        api_key = cfg.GROQ_API_KEY.strip()
    elif llm_provider == "anthropic" and cfg.ANTHROPIC_API_KEY.strip():
        api_key = cfg.ANTHROPIC_API_KEY.strip()

    if not api_key:
        logger.warning("No API key found in config — falling back to template mode")
        llm_provider = "template"

    from pipeline.visual_understanding import analyze_batch
    from pipeline.audio_understanding  import transcribe_batch
    from pipeline.narrative_engine     import generate_script_versions
    def progress(msg: str, pct: int):
        logger.info(f"[{pct:3d}%] {msg}")
        if progress_cb:
            progress_cb(msg, pct)

    # ── Directories ───────────────────────────────────────────────────────

    temp_dir      = os.path.join(output_dir, "temp")
    ambient_dir   = os.path.join(temp_dir, "ambient")
    music_dir     = os.path.join(temp_dir, "music")
    narration_dir = os.path.join(temp_dir, "narration")
    for d in [temp_dir, ambient_dir, music_dir, narration_dir]:
        os.makedirs(d, exist_ok=True)

    timings = {}
    start   = time.time()

    # ── Stage 1: Visual Understanding ─────────────────────────────────────
    progress("Analyzing photos and videos…", 5)
    t0 = time.time()
    visual_list = analyze_batch(photo_video_paths) if photo_video_paths else []
    visual_meta = {vm.media_id: vm for vm in visual_list}
    timings["visual_understanding"] = round(time.time() - t0, 2)
    progress(f"Visual analysis done: {len(visual_list)} items", 20)

    # ── Stage 2: Audio Understanding ──────────────────────────────────────
    progress("Transcribing voice memos…", 20)
    t0 = time.time()
    audio_list = transcribe_batch(audio_paths) if audio_paths else []
    timings["audio_understanding"] = round(time.time() - t0, 2)
    progress(f"Transcription done: {len(audio_list)} memos", 30)

    # ── Stage 3: Narrative Engine (3 versions) ─────────────────────────────
    progress("Generating 3 script versions with Groq…" if llm_provider == "groq" else "Generating scripts…", 32)
    t0 = time.time()
    script_versions = generate_script_versions(
        visual_meta       = visual_list,
        audio_meta        = audio_list,
        api_key           = api_key,
        event_hint        = event_hint,
        user_script       = user_script,
        llm_provider      = llm_provider,
        target_duration_s = target_duration_s,
        script_mode       = script_mode,
    )
    # Default selected script is the first one (Reflective)
    active_script = script_versions[0]
    timings["narrative_engine"] = round(time.time() - t0, 2)
    progress(f"Scripts ready: {len(script_versions)} versions", 55)

    # Save all scripts to JSON
    import json
    script_json_paths = []
    for i, sc in enumerate(script_versions):
        p = os.path.join(output_dir, f"script_v{i+1}.json")
        try:
            with open(p, "w") as f:
                f.write(sc.to_json())
            script_json_paths.append(p)
        except Exception:
            script_json_paths.append(None)

    timings["pre_total"] = round(time.time() - start, 2)
    progress("Analysis complete — pick your style in the Edit panel!", 90)

    return {
        "script_versions":    script_versions,
        "script":             active_script,
        "selected_version":   0,
        "script_json_paths":  script_json_paths,
        "visual_meta":        visual_meta,
        "ambient_segments":   {},           # no longer pre-generating per-beat ambient
        "global_music_path":  None,
        "global_ambient_path": None,
        "music_mood":         music_mood,
        "ambient_scene":      ambient_scene,
        "timings":            timings,
        "output_dir":         output_dir,
        "temp_dir":           temp_dir,
        "music_dir":          music_dir,
        "ambient_dir":        ambient_dir,
        "narration_dir":      narration_dir,
        "target_duration_s":  target_duration_s,
        "llm_provider":       llm_provider,
    }


def run_pipeline_finalize(
    pre_result:        Dict[str, Any],
    selected_version:  int = 0,
    music_mood:        str = "cinematic",
    ambient_scene:     str = "nature",
    narration_vol:     float = 1.0,
    music_vol:         float = 0.35,
    ambient_vol:       float = 0.20,
    voice_reference_path: Optional[str] = None,
    progress_cb:       Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """
    Finalize stage: synthesize narration → generate global music → assemble video.
    Uses the user-selected script version and music mood.
    """
    from pipeline.voice_synthesis    import synthesize_all_beats
    from pipeline.music_generation   import generate_single_music_track
    from pipeline.ambient_sound      import generate_single_ambient_track
    from pipeline.video_assembly     import assemble_documentary
    import config

    def progress(msg: str, pct: int):
        logger.info(f"[{pct:3d}%] {msg}")
        if progress_cb:
            progress_cb(msg, pct)

    # Resolve active script
    script_versions = pre_result["script_versions"]
    script = script_versions[min(selected_version, len(script_versions) - 1)]

    ambient_segments = dict(pre_result["ambient_segments"])
    output_dir       = pre_result["output_dir"]
    narration_dir    = pre_result["narration_dir"]
    music_dir        = pre_result["music_dir"]
    ambient_dir      = pre_result["ambient_dir"]
    visual_meta      = pre_result["visual_meta"]
    timings          = dict(pre_result["timings"])
    total_duration   = script.total_duration_s

    # No per-beat ambient overrides needed — we use a single global ambient track
    ambient_segments = {}

    # ── Narration Synthesis ────────────────────────────────────────────────
    progress("Synthesizing narrator voice (Chatterbox TTS)…", 87)
    t0 = time.time()
    narration_audio = synthesize_all_beats(script, narration_dir, voice_reference_path=voice_reference_path)
    timings["voice_synthesis"] = round(time.time() - t0, 2)
    progress("Narration done", 92)

    # ── Single Global Music Track ─────────────────────────────────────────
    progress(f"Generating {music_mood} music…", 93)
    t0 = time.time()
    music_path = generate_single_music_track(
        mood=music_mood, duration_s=max(total_duration, 10.0),
        output_dir=music_dir, output_name=f"global_{music_mood}.wav",
    )
    timings["music_generation"] = round(time.time() - t0, 2)
    progress("Music ready", 94)

    # ── Single Global Ambient Track ───────────────────────────────────────
    progress(f"Downloading {ambient_scene} ambient…", 95)
    t0 = time.time()
    ambient_path = generate_single_ambient_track(
        scene=ambient_scene, duration_s=max(total_duration, 10.0),
        output_dir=ambient_dir, output_name=f"global_{ambient_scene}.wav",
    )
    timings["ambient_sound"] = round(time.time() - t0, 2)
    progress("Ambient ready", 96)

    # Inject mix levels
    config.NARRATION_LEVEL = narration_vol
    config.MUSIC_LEVEL     = music_vol
    config.AMBIENT_LEVEL   = ambient_vol

    # ── Video Assembly ────────────────────────────────────────────────────
    progress("Assembling documentary…", 97)
    t0 = time.time()
    output_video = os.path.join(output_dir, "documentary.mp4")
    try:
        video_path, audio_path = assemble_documentary(
            script              = script,
            visual_meta         = visual_meta,
            narration_audio     = narration_audio,
            music_segments      = {},
            ambient_segments    = {},
            output_path         = output_video,
            global_music_path   = music_path,
            global_ambient_path = ambient_path,
        )
        if video_path and os.path.exists(video_path):
            timings["video_assembly"] = round(time.time() - t0, 2)
            progress("Documentary ready! 🎬", 100)
            output_video = video_path
        else:
            output_video = None
            progress("Video unavailable — audio tracks ready", 100)
    except Exception as e:
        logger.error(f"Video assembly failed: {e}", exc_info=True)
        output_video = None
        timings["video_assembly"] = round(time.time() - t0, 2)
        progress(f"Assembly failed: {e}", 100)

    # Save selected script JSON
    import json
    script_path = os.path.join(output_dir, "documentary_script.json")
    with open(script_path, "w") as f:
        f.write(script.to_json())

    return {
        "script":             script,
        "script_versions":    script_versions,
        "script_json_path":   script_path,
        "output_video_path":  output_video,
        "visual_meta":        visual_meta,
        "narration_audio":    narration_audio,
        "global_music_path":  music_path,
        "global_ambient_path": ambient_path,
        "music_mood":         music_mood,
        "ambient_scene":      ambient_scene,
        "ambient_segments":   {},
        "timings":            timings,
    }


def run_pipeline(
    photo_video_paths:   List[str],
    audio_paths:         List[str],
    output_dir:          str = "outputs",
    api_key:             Optional[str] = None,
    event_hint:          str = "",
    user_script:         Optional[str] = None,
    llm_provider:        str = "groq",
    music_mood:          str = "cinematic",
    target_duration_s:   float = 60.0,
    progress_cb:         Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """Legacy single-pass pipeline (pre + finalize)."""
    pre = run_pipeline_pre(
        photo_video_paths = photo_video_paths,
        audio_paths       = audio_paths,
        output_dir        = output_dir,
        api_key           = api_key,
        event_hint        = event_hint,
        user_script       = user_script,
        llm_provider      = llm_provider,
        music_mood        = music_mood,
        target_duration_s = target_duration_s,
        progress_cb       = progress_cb,
    )
    return run_pipeline_finalize(
        pre_result       = pre,
        selected_version = 0,
        music_mood       = music_mood,
        progress_cb      = progress_cb,
    )
