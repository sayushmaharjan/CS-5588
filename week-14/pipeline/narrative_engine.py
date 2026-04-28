"""
Cinematic Memory — Narrative Engine
LLM-based system to reconstruct chronological/emotional story arc from media metadata.
Uses Anthropic Claude API with structured JSON output.
Falls back to template-based script if no API key.
"""
from __future__ import annotations
import os, json, logging, re
from typing import List, Dict, Optional
from utils.data_schemas import (
    VisualMetadata, AudioMetadata, DocumentaryScript,
    NarrationBeat, ActPhase, EmotionTag, SceneType
)
import config

logger = logging.getLogger(__name__)


# ── Prompt Templates ────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are an acclaimed documentary filmmaker and screenwriter.
# Your task: transform raw media metadata and voice memo transcripts into a 
# cinematic 3-act documentary script with emotional depth and coherent narrative.

# Output ONLY valid JSON. No markdown fences. No preamble. No explanation."""
SYSTEM_PROMPT = """You are an award-winning documentary filmmaker, poet, and reflective storyteller.

Your task: transform raw media metadata and voice memo transcripts into a cinematic, deeply human 3-act documentary script.

Write with emotional depth, introspection, and subtlety. The narration should feel like an intimate inner monologue—honest, imperfect, and personal. Avoid sounding robotic, overly structured, or generic.

STYLE GUIDELINES:
- Use soft, reflective language that feels like spoken thoughts, not formal writing
- Let emotions unfold naturally; show vulnerability, curiosity, and quiet realization
- Include sensory details (sounds, light, atmosphere, small moments)
- Embrace pauses, silence, and ambient presence (e.g., footsteps, wind, distant chatter)
- Avoid clichés and dramatic exaggeration; keep it grounded and real
- Make it feel like a personal journey rather than a summary of events

STRUCTURE:
- Act 1: Arrival / curiosity / subtle emotional setup
- Act 2: Immersion / conflict / reflection / small discoveries
- Act 3: Resolution / emotional shift / quiet insight (not forced)

NARRATION STYLE:
- First-person voice
- Gentle, flowing sentences (like a voiceover over soft music)
- Occasional fragmented thoughts are okay if they feel natural

Output ONLY valid JSON. No markdown fences. No preamble. No explanation.
"""

NARRATIVE_PROMPT_TEMPLATE = """
Create a documentary script from this personal media collection.

## MEDIA METADATA (photos/videos):
{media_summaries}

## VOICE MEMO / USER SCRIPT CONTEXT:
{transcript_summaries}

## TASK:
Build a structured 3-act documentary script.

### ACT STRUCTURE:
- Act 1 — SETUP (30%): establish setting, people, context. Tone: warm/curious.
- Act 2 — PEAK (40%): core memories, highest emotion, key moments. Tone: joyful/intense.
- Act 3 — REFLECTION (30%): meaning, what was learned, looking back. Tone: nostalgic/grateful.

### ORDERING RULES:
1. If EXIF timestamps exist → sort chronologically.
2. If no timestamps → sort by emotional arc: setup → peak → reflection.
3. High salience_score media gets priority placement.
4. Group media by scene_type for visual coherence.

### OUTPUT JSON SCHEMA:
{{
  "title": "string — evocative documentary title",
  "arc_summary": "string — 2-3 sentences describing emotional journey",
  "total_duration_s": number,
  "beats": [
    {{
      "beat_id": "beat_001",
      "act_phase": "setup|peak|reflection",
      "narration_text": "string — the actual narration script for this beat (2-4 sentences, poetic, cinematic)",
      "media_ids": ["list of media_ids to use for this beat"],
      "emotion": "joyful|nostalgic|reflective|sad|excited|neutral|celebratory|tender",
      "duration_hint_s": number,
      "cut_speed": "slow|medium|fast",
      "music_prompt": "string — detailed MusicGen prompt for this beat",
      "ambient_prompt": "string — AudioLDM2 ambient sound prompt"
    }}
  ]
}}

### CONSTRAINTS:
- 5–12 beats total
- narration_text: poetic, second-person or first-person, cinematic language. 
  CRITICAL: The narration_text across all beats MUST read as ONE continuous, flowing story. Do not write disjointed, isolated thoughts per beat. Use transitional phrases so sentences flow into the next beat smoothly.
- each beat: 5–15s duration
- total_duration_s: sum of all beat durations
- Use ALL provided media_ids at least once
- music_prompt and ambient_prompt: specific and evocative (15-30 words each)

### VOICE / SCRIPT INTEGRATION:
If a user script is provided, use it as the primary narration source.
Weave its content naturally into narration_text beats. Don't quote verbatim — shape it cinematically.

CRITICAL: If USER-PROVIDED SCRIPT is given below, you MUST base ALL narration_text on the user's
actual words. Do NOT ignore the user's script. Do NOT invent new narration. Reshape the user's
text into cinematic beats while preserving their meaning and key phrases.
"""


def _build_media_summaries(visual_meta: List[VisualMetadata]) -> str:
    lines = []
    for vm in visual_meta:
        ts = f", timestamp={vm.exif_timestamp}" if vm.exif_timestamp else ""
        lines.append(
            f"- media_id={vm.media_id}, type={vm.media_type.value}, "
            f"scene={vm.scene_type.value}, "
            f"emotions=[{', '.join(e.value for e in vm.emotions)}], "
            f"objects=[{', '.join(vm.objects[:5])}], "
            f"salience={vm.salience_score:.2f}{ts}"
        )
    return "\n".join(lines)


def _build_transcript_summaries(audio_meta: List[AudioMetadata]) -> str:
    if not audio_meta:
        return "No voice memos provided."
    lines = []
    for am in audio_meta:
        lines.append(
            f"- audio_id={am.audio_id}, duration={am.duration_s:.1f}s, "
            f"emotion={am.overall_emotion.value}\n"
            f"  TRANSCRIPT: {am.transcript[:500]}"
        )
    return "\n".join(lines)


def _call_anthropic(prompt: str, api_key: str) -> str:
    """Call Anthropic Claude API for narrative generation."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_groq(prompt: str, api_key: str) -> str:
    """Call Groq API for narrative generation (free tier, very fast)."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "groq package not installed. Run: pip install groq"
        )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        max_tokens=2000, # Reduced to avoid hitting 12000 TPM limit across 3 script variants
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


def _parse_llm_response(raw: str, all_media_ids: List[str]) -> DocumentaryScript:
    """Parse LLM JSON response into DocumentaryScript."""
    # Strip any accidental markdown fences
    raw_clean = re.sub(r"```(?:json)?", "", raw).strip()
    data      = json.loads(raw_clean)

    beats = []
    for i, b in enumerate(data.get("beats", [])):
        beat = NarrationBeat(
            beat_id        = b.get("beat_id", f"beat_{i+1:03d}"),
            act_phase      = ActPhase(b.get("act_phase", "setup")),
            narration_text = b.get("narration_text", ""),
            media_ids      = b.get("media_ids", []),
            emotion        = EmotionTag(b.get("emotion", "neutral")),
            duration_hint_s= float(b.get("duration_hint_s", 8.0)),
            cut_speed      = b.get("cut_speed", "medium"),
            music_prompt   = b.get("music_prompt", config.EMOTION_MUSIC_PROMPTS["neutral"]),
            ambient_prompt = b.get("ambient_prompt", "gentle neutral ambiance"),
        )
        beats.append(beat)

    return DocumentaryScript(
        title            = data.get("title", "Untitled Documentary"),
        total_duration_s = float(data.get("total_duration_s", sum(b.duration_hint_s for b in beats))),
        arc_summary      = data.get("arc_summary", ""),
        beats            = beats,
        raw_llm_output   = raw,
    )


def _split_user_script_into_chunks(user_script: str, num_beats: int) -> List[str]:
    """Split user script text into roughly equal chunks for each beat."""
    sentences = re.split(r'(?<=[.!?])\s+', user_script.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [user_script.strip()] * num_beats

    # Distribute sentences across beats as evenly as possible
    chunks = [[] for _ in range(num_beats)]
    for i, sentence in enumerate(sentences):
        chunks[i % num_beats].append(sentence)

    return [" ".join(c) if c else sentences[min(i, len(sentences)-1)] for i, c in enumerate(chunks)]


def _template_fallback(
    visual_meta: List[VisualMetadata],
    audio_meta:  List[AudioMetadata],
    user_script: Optional[str] = None,
) -> DocumentaryScript:
    """
    Template-based script generation when LLM unavailable.
    If user_script is provided, distributes user's text across all beats.
    Otherwise uses default template narration.
    """
    from collections import defaultdict

    # Sort media: EXIF first, then by salience
    sorted_media = sorted(
        visual_meta,
        key=lambda m: (m.exif_timestamp or "z", -m.salience_score)
    )

    total = len(sorted_media)
    setup_end  = max(1, int(total * 0.30))
    peak_end   = max(2, int(total * 0.70))

    setup_media = sorted_media[:setup_end]
    peak_media  = sorted_media[setup_end:peak_end]
    refl_media  = sorted_media[peak_end:]

    # Determine how many beats we'll generate
    num_beats = 3  # minimum: setup, peak, reflection
    if peak_media:
        num_beats += 1  # rising action
    if len(peak_media) > 1:
        num_beats += 1  # peak beat

    # If user script provided, split across beats
    has_user_script = bool(user_script and user_script.strip())
    if has_user_script:
        script_chunks = _split_user_script_into_chunks(user_script, num_beats)
    else:
        script_chunks = None

    # Default template narrations (used when no user script)
    default_narrations = [
        ("There are places and moments that stay with you long after they've passed. "
         "This is the story of one of them — a collection of days, faces, and feelings "
         "that came together in a way that felt almost impossible to put into words."),
        ("It started simply — the way the best things always do. "
         "Small details began to weave together: the light, the laughter, "
         "the particular quality of being exactly where you're supposed to be."),
        ("And then it happened — that rare convergence of everything feeling alive at once. "
         "Laughter, movement, connection. All of it real. All of it ours."),
        ("As the light began to soften, so did everything else. "
         "The rush gave way to something quieter — a gratitude that settles "
         "in your chest when you know you've been somewhere truly special."),
        ("These images, these sounds, these fleeting seconds captured — "
         "they are more than photographs. They are proof that something extraordinary "
         "happened here. And that it mattered."),
    ]

    beats = []
    chunk_idx = 0

    def _get_narration(idx):
        if script_chunks:
            return script_chunks[min(idx, len(script_chunks) - 1)]
        return default_narrations[min(idx, len(default_narrations) - 1)]

    # Beat 1 — Setup
    beats.append(NarrationBeat(
        beat_id="beat_001", act_phase=ActPhase.SETUP,
        narration_text=_get_narration(chunk_idx),
        media_ids=[m.media_id for m in setup_media],
        emotion=EmotionTag.REFLECTIVE, duration_hint_s=10.0, cut_speed="slow",
        music_prompt=config.EMOTION_MUSIC_PROMPTS["reflective"],
        ambient_prompt=config.SCENE_AMBIENT_PROMPTS.get(
            setup_media[0].scene_type.value if setup_media else "unknown", "gentle ambiance"
        ),
    ))
    chunk_idx += 1

    # Beat 2 — Rising action
    if peak_media:
        first_scene = peak_media[0].scene_type.value
        beats.append(NarrationBeat(
            beat_id="beat_002", act_phase=ActPhase.SETUP,
            narration_text=_get_narration(chunk_idx),
            media_ids=[m.media_id for m in peak_media[:max(1,len(peak_media)//3)]],
            emotion=EmotionTag.JOYFUL, duration_hint_s=8.0, cut_speed="medium",
            music_prompt=config.EMOTION_MUSIC_PROMPTS["joyful"],
            ambient_prompt=config.SCENE_AMBIENT_PROMPTS.get(first_scene, "warm ambiance"),
        ))
        chunk_idx += 1

    # Beat 3 — Peak
    if len(peak_media) > 1:
        beats.append(NarrationBeat(
            beat_id="beat_003", act_phase=ActPhase.PEAK,
            narration_text=_get_narration(chunk_idx),
            media_ids=[m.media_id for m in peak_media[len(peak_media)//3:]],
            emotion=EmotionTag.CELEBRATORY, duration_hint_s=12.0, cut_speed="fast",
            music_prompt=config.EMOTION_MUSIC_PROMPTS["celebratory"],
            ambient_prompt="festive crowd ambiance, energy and warmth",
        ))
        chunk_idx += 1

    # Beat 4 — Wind down
    beats.append(NarrationBeat(
        beat_id="beat_004", act_phase=ActPhase.REFLECTION,
        narration_text=_get_narration(chunk_idx),
        media_ids=[m.media_id for m in refl_media[:max(1, len(refl_media)//2)]],
        emotion=EmotionTag.NOSTALGIC, duration_hint_s=10.0, cut_speed="slow",
        music_prompt=config.EMOTION_MUSIC_PROMPTS["nostalgic"],
        ambient_prompt="soft evening ambiance, gentle wind, quiet nature sounds",
    ))
    chunk_idx += 1

    # Beat 5 — Reflection
    beats.append(NarrationBeat(
        beat_id="beat_005", act_phase=ActPhase.REFLECTION,
        narration_text=_get_narration(chunk_idx),
        media_ids=[m.media_id for m in (refl_media[len(refl_media)//2:] or sorted_media[-2:])],
        emotion=EmotionTag.TENDER, duration_hint_s=12.0, cut_speed="slow",
        music_prompt=config.EMOTION_MUSIC_PROMPTS["tender"],
        ambient_prompt="quiet intimate ambiance, soft piano room tone",
    ))

    total_dur = sum(b.duration_hint_s for b in beats)
    return DocumentaryScript(
        title            = "A Story Worth Telling",
        total_duration_s = total_dur,
        arc_summary      = (
            "A journey from quiet beginnings through joyful celebration, "
            "arriving at tender reflection — the emotional arc of a memory made permanent."
        ),
        beats = beats,
    )


def generate_script(
    visual_meta:       List[VisualMetadata],
    audio_meta:        List[AudioMetadata],
    api_key:           Optional[str] = None,
    event_hint:        str = "",
    user_script:       Optional[str] = None,
    llm_provider:      str = "anthropic",
    target_duration_s: Optional[float] = None,
    tone_hint:         str = "",
    script_mode:       str = "cinematic",  # "exact" or "cinematic"
) -> DocumentaryScript:
    """
    Main entry point for Narrative Engine.
    Returns a single DocumentaryScript.

    script_mode:
      - "exact":     user script is split directly across beats (no LLM rewriting)
      - "cinematic": LLM reshapes user script into cinematic narration
    """
    all_media_ids = [vm.media_id for vm in visual_meta]

    # ── Exact mode: bypass LLM, split user script directly into beats ─────
    if script_mode == "exact" and user_script and user_script.strip():
        logger.info("Exact script mode — using user's script text directly")
        script = _template_fallback(visual_meta, audio_meta, user_script=user_script)
        if target_duration_s:
            script = _scale_script_duration(script, target_duration_s)
        return script

    if not api_key or llm_provider == "template":
        logger.info("No API key or template mode — using template-based narrative fallback")
        script = _template_fallback(visual_meta, audio_meta, user_script=user_script)
        if target_duration_s:
            script = _scale_script_duration(script, target_duration_s)
        return script

    try:
        media_summaries = _build_media_summaries(visual_meta)

        if user_script and user_script.strip():
            transcript_summaries = (
                f"USER-PROVIDED SCRIPT (use this as the primary narration source):\n"
                f"IMPORTANT: You MUST use these words as the basis for narration_text.\n"
                f"Do NOT ignore this script. Do NOT generate entirely new narration.\n\n"
                f"{user_script.strip()[:2000]}"
            )
            if audio_meta:
                transcript_summaries += "\n\nADDITIONAL VOICE MEMO CONTEXT:\n" + \
                    _build_transcript_summaries(audio_meta)
        else:
            transcript_summaries = _build_transcript_summaries(audio_meta)

        if event_hint:
            media_summaries = f"EVENT TYPE: {event_hint}\n\n" + media_summaries

        # Build duration constraint
        dur_constraint = ""
        if target_duration_s:
            beat_count = max(3, min(12, int(target_duration_s / 8)))
            dur_constraint = (
                f"\n### DURATION CONSTRAINT:\n"
                f"- Target total video length: {target_duration_s:.0f} seconds\n"
                f"- Use approximately {beat_count} beats\n"
                f"- Each beat: {target_duration_s/beat_count:.1f}s average\n"
                f"- total_duration_s MUST be close to {target_duration_s:.0f}\n"
            )

        # Tone instruction
        tone_instruction = ""
        if tone_hint:
            tone_instruction = f"\n### EMOTIONAL TONE:\nWrite this version with a predominantly **{tone_hint}** emotional tone throughout.\n"

        prompt = NARRATIVE_PROMPT_TEMPLATE.format(
            media_summaries      = media_summaries,
            transcript_summaries = transcript_summaries,
        ) + dur_constraint + tone_instruction

        logger.info(f"Calling {llm_provider} LLM for narrative generation…")
        if llm_provider == "groq":
            raw = _call_groq(prompt, api_key)
        else:
            raw = _call_anthropic(prompt, api_key)

        script = _parse_llm_response(raw, all_media_ids)
        if target_duration_s:
            script = _scale_script_duration(script, target_duration_s)
        logger.info(f"Script generated: '{script.title}', {len(script.beats)} beats")
        return script

    except Exception as e:
        logger.error(f"LLM narrative generation failed: {e}. Using template fallback.")
        script = _template_fallback(visual_meta, audio_meta, user_script=user_script)
        if target_duration_s:
            script = _scale_script_duration(script, target_duration_s)
        return script


def _scale_script_duration(script: DocumentaryScript, target_s: float) -> DocumentaryScript:
    """Scale all beat durations proportionally to hit the target total duration."""
    current_total = sum(b.duration_hint_s for b in script.beats)
    if current_total <= 0:
        return script
    ratio = target_s / current_total
    for beat in script.beats:
        beat.duration_hint_s = round(beat.duration_hint_s * ratio, 1)
    script.total_duration_s = target_s
    return script


TONE_VARIANTS = [
    {
        "label": "Reflective & Nostalgic",
        "icon": "🥺",
        "tone_hint": "reflective and nostalgic",
        "description": "Quiet, introspective tone. Feels like looking back on something precious.",
    },
    {
        "label": "Joyful & Celebratory",
        "icon": "🎉",
        "tone_hint": "joyful and celebratory",
        "description": "Uplifting, warm, and energetic. Celebrates the people and moments.",
    },
    {
        "label": "Adventurous & Cinematic",
        "icon": "🎬",
        "tone_hint": "adventurous and cinematic",
        "description": "Epic, wide-lens storytelling. Feels grand and purposeful.",
    },
]


def generate_script_versions(
    visual_meta:       List[VisualMetadata],
    audio_meta:        List[AudioMetadata],
    api_key:           Optional[str] = None,
    event_hint:        str = "",
    user_script:       Optional[str] = None,
    llm_provider:      str = "anthropic",
    target_duration_s: Optional[float] = None,
    script_mode:       str = "cinematic",
) -> List[DocumentaryScript]:
    """
    Generate multiple script versions with different emotional tones.
    Returns a list of DocumentaryScript objects, one per tone variant.
    Falls back gracefully: if LLM fails for a variant, uses template with adjusted title.
    """
    scripts = []
    import time
    for i, variant in enumerate(TONE_VARIANTS):
        logger.info(f"Generating script variant: {variant['label']}")
        if i > 0:
            time.sleep(2) # Give Groq a breather
        try:
            script = generate_script(
                visual_meta       = visual_meta,
                audio_meta        = audio_meta,
                api_key           = api_key,
                event_hint        = event_hint,
                user_script       = user_script,
                llm_provider      = llm_provider,
                target_duration_s = target_duration_s,
                tone_hint         = variant["tone_hint"],
                script_mode       = script_mode,
            )
            # Tag each script with its variant metadata
            script._variant_label       = variant["label"]
            script._variant_icon        = variant["icon"]
            script._variant_description = variant["description"]
        except Exception as e:
            logger.error(f"Variant '{variant['label']}' failed: {e}")
            script = _template_fallback(visual_meta, audio_meta, user_script=user_script)
            if target_duration_s:
                script = _scale_script_duration(script, target_duration_s)
            script._variant_label       = variant["label"]
            script._variant_icon        = variant["icon"]
            script._variant_description = variant["description"]
        scripts.append(script)
    return scripts
