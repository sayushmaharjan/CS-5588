"""
app.py — Streamlit UI for the Fashion Outfit Generator.

Exposes all three pipeline modes:
  1. Inpaint        — clothing inpainting via segment.py (generate_inpaint)
  2. ControlNet     — full pose + identity pipeline (generate)
  3. Comparison     — baseline vs structured side-by-side (generate_comparison)

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OCCASIONS,
    STYLES,
    COLOR_PALETTES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_HEIGHT,
    DEVICE,
)
from prompt_engine import generate_prompt_pair
from control import validate_input_image, prepare_reference_image


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def compute_output_dims(image, min_short=512, max_long=768, multiple_of=8):
    """Compute SD-compatible (w, h) preserving aspect ratio."""
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h
    if orig_w <= orig_h:
        new_w = min_short
        new_h = int(round(new_w / aspect))
    else:
        new_h = min_short
        new_w = int(round(new_h * aspect))
    if new_w > max_long:
        new_w = max_long
        new_h = int(round(new_w / aspect))
    if new_h > max_long:
        new_h = max_long
        new_w = int(round(new_h * aspect))
    new_w = max(multiple_of, (new_w // multiple_of) * multiple_of)
    new_h = max(multiple_of, (new_h // multiple_of) * multiple_of)
    return new_w, new_h


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion Outfit Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0c0e14; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

/* Topbar */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2rem;
    background: #13151d;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 1.5rem;
}
.topbar-brand { font-size: 0.95rem; font-weight: 600; letter-spacing: 0.04em; color: #e2e8f0; text-transform: uppercase; }
.topbar-meta  { font-size: 0.72rem; color: #4a5568; letter-spacing: 0.05em; }

/* Section label */
.section-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4a5568; margin: 0 0 0.5rem 0;
}

/* Mode tabs */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #718096 !important;
    padding: 0.5rem 1.2rem !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #4361ee !important;
    border-bottom: 2px solid #4361ee !important;
}
div[data-testid="stTabs"] [data-testid="stTabsBar"] {
    background: #13151d !important;
    border-bottom: 1px solid #1e2130 !important;
    border-radius: 8px 8px 0 0;
}

/* Prompt block */
.prompt-block {
    background: #0c0e14; border: 1px solid #1e2130; border-radius: 8px;
    padding: 0.85rem 1rem; font-size: 0.8rem; line-height: 1.65;
    color: #a0aec0; font-family: 'JetBrains Mono', monospace;
    margin-top: 0.35rem; word-break: break-word;
}

/* Tag badges */
.tag { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px;
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; margin-bottom: 0.3rem; }
.tag-baseline  { background: rgba(255,183,77,.1); color: #f6ad55; border: 1px solid rgba(255,183,77,.25); }
.tag-structured{ background: rgba(72,199,142,.1); color: #48c78e; border: 1px solid rgba(72,199,142,.25); }
.tag-inpaint   { background: rgba(99,102,241,.1); color: #818cf8; border: 1px solid rgba(99,102,241,.25); }

/* Buttons */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #4361ee; border: none; color: #fff;
    font-weight: 600; font-size: 0.88rem; padding: 0.6rem 1.5rem;
    border-radius: 8px; width: 100%; transition: background 0.18s, transform 0.12s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #3451d1; transform: translateY(-1px);
}
div[data-testid="stButton"] > button {
    background: #1e2130; border: 1px solid #2d3140; color: #e2e8f0;
    font-weight: 500; font-size: 0.85rem; padding: 0.55rem 1.1rem;
    border-radius: 8px; width: 100%; transition: background 0.18s;
}
div[data-testid="stButton"] > button:hover { background: #252838; }

/* Metric card */
.metric-card {
    background: #13151d; border: 1px solid #1e2130;
    border-radius: 8px; padding: 1rem 0.8rem; text-align: center;
}
.metric-value { font-size: 1.4rem; font-weight: 700; color: #4361ee; }
.metric-label { font-size: 0.7rem; color: #4a5568; margin-top: 0.2rem; line-height: 1.4; }

/* Widgets */
.stSelectbox label, .stSlider label, .stNumberInput label,
.stTextArea label, .stTextInput label {
    font-size: 0.76rem !important; font-weight: 500 !important;
    color: #718096 !important; letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #13151d !important; border-color: #2d3140 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}
div[data-testid="stFileUploader"] {
    background: #13151d; border: 1.5px dashed #2d3140; border-radius: 10px; padding: 1rem;
}
div[data-testid="stFileUploader"]:hover { border-color: #4361ee; }
div[data-testid="stFileUploader"] label { color: #718096 !important; }

hr.section-hr { border: none; border-top: 1px solid #1e2130; margin: 1.25rem 0; }

div[data-testid="stImage"] img { border-radius: 8px; border: 1px solid #1e2130; }
details > summary { font-size: 0.82rem; font-weight: 500; color: #718096; }
div[data-testid="stAlert"] { border-radius: 8px; font-size: 0.83rem; }
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; border: 1px solid #1e2130; }

.app-footer {
    text-align: center; color: #2d3140; font-size: 0.7rem;
    padding: 2rem 0 1rem; letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Topbar
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <span class="topbar-brand">Fashion Outfit Generator</span>
    <span class="topbar-meta">SD 1.5 + ControlNet + IP-Adapter &nbsp;|&nbsp; {DEVICE.upper()}</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
for key, default in [
    ("pipeline", None),
    ("model_loaded", False),
    ("inpaint_results", None),
    ("generated_results", None),
    ("comparison_results", None),
    ("eval_results", None),
    ("active_mode", None),   # tracks which mode produced the current results
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════
# LAYOUT — left controls | right output
# ══════════════════════════════════════════════
left_col, right_col = st.columns([1, 1.6], gap="large")


# ┌─────────────────────────────────────────────
# │  LEFT COLUMN
# └─────────────────────────────────────────────
with left_col:

    # ── Upload ────────────────────────────────
    st.markdown('<p class="section-label">Reference Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a full-body photo (JPEG / PNG / WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible",
    )

    reference_image = None
    out_w, out_h = DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT

    if uploaded_file is not None:
        reference_image = Image.open(uploaded_file).convert("RGB")
        out_w, out_h = compute_output_dims(reference_image)

        validation = validate_input_image(reference_image)
        if validation["valid"]:
            st.image(reference_image, use_container_width=True)
            st.caption(
                f"Input: {reference_image.size[0]}x{reference_image.size[1]} px  "
                f"|  Target: {out_w}x{out_h} px"
            )
        else:
            st.error(validation["message"])
            reference_image = None

    st.markdown("<hr class='section-hr'>", unsafe_allow_html=True)

    # ── Outfit parameters (shared across all modes) ──
    st.markdown('<p class="section-label">Outfit Parameters</p>', unsafe_allow_html=True)

    occasion = st.selectbox("Occasion", OCCASIONS, index=0)
    style    = st.selectbox("Style",    STYLES,    index=0)

    color_palette_name = st.selectbox("Color Palette", list(COLOR_PALETTES.keys()), index=0)
    if color_palette_name == "Custom":
        color_palette = st.text_input(
            "Custom Color Description",
            placeholder="e.g., warm earth tones with subtle gold accents",
        )
    else:
        color_palette = COLOR_PALETTES[color_palette_name]

    with st.expander("Custom Outfit Description (optional)"):
        outfit_override = st.text_area(
            "Override the auto-generated garment description",
            placeholder="e.g., a slim-fit charcoal suit with a white dress shirt and black oxford shoes",
            height=80,
        )
        outfit_override = outfit_override.strip() or None

    st.markdown("<hr class='section-hr'>", unsafe_allow_html=True)

    # ── Generation Mode ───────────────────────
    st.markdown('<p class="section-label">Generation Mode</p>', unsafe_allow_html=True)

    MODE_INPAINT    = "Inpaint  (Clothing Swap)"
    MODE_CONTROLNET = "ControlNet + IP-Adapter"
    MODE_COMPARE    = "Comparison  (Baseline vs Structured)"

    mode = st.radio(
        "Select mode",
        [MODE_INPAINT, MODE_CONTROLNET, MODE_COMPARE],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # ── Mode-specific parameters ──────────────
    seed_input = None       # defined in each branch
    num_images = 1
    num_steps  = DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale    = DEFAULT_GUIDANCE_SCALE
    controlnet_scale  = DEFAULT_CONTROLNET_SCALE
    ip_adapter_scale  = DEFAULT_IP_ADAPTER_SCALE
    inpaint_strength  = 0.95

    if mode == MODE_INPAINT:
        st.markdown("""
        <small style="color:#4a5568;line-height:1.6;display:block;margin-bottom:0.75rem">
        Segments the clothing region from the reference image using
        a semantic segmentation model, then inpaints only the clothing
        pixels with a new outfit — leaving the face and background intact.
        </small>
        """, unsafe_allow_html=True)

        inpaint_strength = st.slider(
            "Inpaint Strength",
            min_value=0.5, max_value=1.0, value=0.95, step=0.01,
            help="Higher = more creative freedom; lower = stays closer to original."
        )
        seed_input = st.number_input(
            "Seed  (0 = random)", min_value=0, max_value=999_999_999, value=42
        )

    elif mode == MODE_CONTROLNET:
        st.markdown("""
        <small style="color:#4a5568;line-height:1.6;display:block;margin-bottom:0.75rem">
        Extracts the body pose via OpenPose (ControlNet) and preserves
        the person's identity via IP-Adapter, then generates a full
        re-dressed outfit from scratch.
        </small>
        """, unsafe_allow_html=True)

        num_images = st.slider("Number of Variations", min_value=1, max_value=4, value=2)

        with st.expander("Advanced Parameters"):
            num_steps = st.slider(
                "Inference Steps", min_value=10, max_value=50,
                value=DEFAULT_NUM_INFERENCE_STEPS,
                help="More steps = higher quality, slower generation."
            )
            guidance_scale = st.slider(
                "Guidance Scale", min_value=1.0, max_value=15.0,
                value=float(DEFAULT_GUIDANCE_SCALE), step=0.5,
            )
            controlnet_scale = st.slider(
                "ControlNet Scale  (Pose)", min_value=0.0, max_value=1.5,
                value=float(DEFAULT_CONTROLNET_SCALE), step=0.05,
            )
            ip_adapter_scale = st.slider(
                "IP-Adapter Scale  (Identity)", min_value=0.0, max_value=1.0,
                value=float(DEFAULT_IP_ADAPTER_SCALE), step=0.05,
            )
            seed_input = st.number_input(
                "Seed  (0 = random)", min_value=0, max_value=999_999_999, value=42
            )

    else:  # Comparison
        st.markdown("""
        <small style="color:#4a5568;line-height:1.6;display:block;margin-bottom:0.75rem">
        Generates two sets of images side-by-side: a naive baseline prompt
        and a structured prompt. Runs automatic evaluation (CLIP, identity,
        quality, diversity) and produces a downloadable report.
        </small>
        """, unsafe_allow_html=True)

        num_images = st.slider("Variations per Mode", min_value=1, max_value=4, value=2)

        with st.expander("Advanced Parameters"):
            num_steps = st.slider(
                "Inference Steps", min_value=10, max_value=50,
                value=DEFAULT_NUM_INFERENCE_STEPS,
            )
            guidance_scale = st.slider(
                "Guidance Scale", min_value=1.0, max_value=15.0,
                value=float(DEFAULT_GUIDANCE_SCALE), step=0.5,
            )
            controlnet_scale = st.slider(
                "ControlNet Scale", min_value=0.0, max_value=1.5,
                value=float(DEFAULT_CONTROLNET_SCALE), step=0.05,
            )
            ip_adapter_scale = st.slider(
                "IP-Adapter Scale", min_value=0.0, max_value=1.0,
                value=float(DEFAULT_IP_ADAPTER_SCALE), step=0.05,
            )
            seed_input = st.number_input(
                "Seed  (0 = random)", min_value=0, max_value=999_999_999, value=42
            )

    # resolve seed
    actual_seed = int(seed_input) if (seed_input and seed_input != 0) else None

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Action button ─────────────────────────
    btn_disabled = reference_image is None
    btn_run = st.button(
        "Run Generation",
        type="primary",
        disabled=btn_disabled,
        use_container_width=True,
    )
    if btn_disabled:
        st.caption("Upload a reference image to enable generation.")


# ┌─────────────────────────────────────────────
# │  RIGHT COLUMN
# └─────────────────────────────────────────────
with right_col:

    # ── Prompt preview (always live) ──────────
    st.markdown('<p class="section-label">Prompt Preview</p>', unsafe_allow_html=True)

    prompts = generate_prompt_pair(occasion, style, color_palette, outfit_override)

    if mode == MODE_INPAINT:
        # Inpaint only uses the structured prompt
        st.markdown('<span class="tag tag-inpaint">Inpaint Prompt</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="prompt-block">{prompts["structured"]}</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<span class="tag tag-baseline">Baseline</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="prompt-block">{prompts["naive"]}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<span class="tag tag-structured">Structured</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="prompt-block">{prompts["structured"]}</div>', unsafe_allow_html=True)

    if reference_image is not None and mode != MODE_INPAINT:
        with st.expander("Preprocessed Reference Preview"):
            prepared = prepare_reference_image(reference_image, out_w, out_h)
            st.image(prepared, caption=f"Center-cropped to {out_w}x{out_h}", use_container_width=True)

    st.markdown("<hr class='section-hr'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════
    if btn_run and reference_image is not None:

        # Clear stale results from other modes
        st.session_state.inpaint_results    = None
        st.session_state.generated_results  = None
        st.session_state.comparison_results = None
        st.session_state.eval_results       = None
        st.session_state.active_mode        = mode

        from pipeline import get_pipeline

        pipe = get_pipeline()

        # ── INPAINT ───────────────────────────
        if mode == MODE_INPAINT:
            status = st.status("Running clothing inpaint pipeline...", expanded=True)
            try:
                def _progress(msg):
                    status.write(msg)
                pipe.set_progress_callback(_progress)

                status.write("Loading segmentation model and inpaint pipeline...")
                result = pipe.generate_inpaint(
                    reference_image=reference_image,
                    occasion=occasion,
                    style=style,
                    color_palette=color_palette,
                    seed=actual_seed if actual_seed is not None else 42,
                    strength=inpaint_strength,
                )
                st.session_state.inpaint_results = result
                status.update(label="Inpainting complete.", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Inpainting failed.", state="error", expanded=True)
                st.error(str(e))
                st.exception(e)

        # ── CONTROLNET + IP-ADAPTER ───────────
        elif mode == MODE_CONTROLNET:
            status = st.status("Loading models and generating images...", expanded=True)
            try:
                def _progress(msg):
                    status.write(msg)
                pipe.set_progress_callback(_progress)

                if not st.session_state.model_loaded:
                    pipe.load_models()
                    st.session_state.model_loaded = True

                result = pipe.generate(
                    reference_image=reference_image,
                    occasion=occasion,
                    style=style,
                    color_palette=color_palette,
                    num_images=num_images,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    controlnet_scale=controlnet_scale,
                    ip_adapter_scale=ip_adapter_scale,
                    width=out_w,
                    height=out_h,
                    seed=actual_seed,
                    outfit_override=outfit_override,
                )
                st.session_state.generated_results = result
                status.update(label="Generation complete.", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Generation failed.", state="error", expanded=True)
                st.error(str(e))
                st.exception(e)

        # ── COMPARISON ────────────────────────
        else:
            status = st.status("Generating comparison (baseline + structured)...", expanded=True)
            try:
                def _progress(msg):
                    status.write(msg)
                pipe.set_progress_callback(_progress)

                if not st.session_state.model_loaded:
                    pipe.load_models()
                    st.session_state.model_loaded = True

                comparison = pipe.generate_comparison(
                    reference_image=reference_image,
                    occasion=occasion,
                    style=style,
                    color_palette=color_palette,
                    num_images=num_images,
                    seed=actual_seed if actual_seed else 42,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    controlnet_scale=controlnet_scale,
                    ip_adapter_scale=ip_adapter_scale,
                    width=out_w,
                    height=out_h,
                )
                st.session_state.comparison_results = comparison

                status.write("Running evaluation metrics...")
                from evaluation import evaluate_comparison
                eval_result = evaluate_comparison(
                    comparison["naive"],
                    comparison["structured"],
                    reference_image,
                )
                st.session_state.eval_results = eval_result
                status.update(label="Comparison complete.", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Generation failed.", state="error", expanded=True)
                st.error(str(e))
                st.exception(e)

    # ══════════════════════════════════════════
    # DISPLAY RESULTS
    # ══════════════════════════════════════════

    # ── Inpaint results ───────────────────────
    if st.session_state.inpaint_results is not None and st.session_state.active_mode == MODE_INPAINT:
        result = st.session_state.inpaint_results

        st.markdown('<p class="section-label">Inpaint Result</p>', unsafe_allow_html=True)

        # Show output image + mask side by side
        col_out, col_mask = st.columns(2)
        with col_out:
            st.markdown("<small style='color:#4a5568'>Generated Output</small>", unsafe_allow_html=True)
            st.image(result["images"][0], use_container_width=True)

        with col_mask:
            st.markdown("<small style='color:#4a5568'>Segmentation Mask</small>", unsafe_allow_html=True)
            st.image(result["mask"], use_container_width=True)
            st.caption("White = clothing region that was inpainted")

        # Reference image used by the pipeline
        if "reference_image" in result:
            with st.expander("Resized Reference Used by Pipeline"):
                st.image(result["reference_image"], use_container_width=True)

        with st.expander("Prompt Used"):
            st.markdown(f'<div class="prompt-block">{result["prompt"]}</div>', unsafe_allow_html=True)

        # Quick single-image metrics
        st.markdown('<p class="section-label" style="margin-top:1rem">Quick Evaluation</p>', unsafe_allow_html=True)
        from evaluation import compute_clip_score, compute_quality_score, compute_identity_score
        generated_img = result["images"][0]
        clip_s  = compute_clip_score(generated_img, result["prompt"])
        qual_s  = compute_quality_score(generated_img)
        
        pref_img = result.get("reference_image", reference_image)
        id_s    = 0.0
        if pref_img is not None:
            id_s = compute_identity_score(pref_img, generated_img)

        m1, m2, m3 = st.columns(3)
        for col, label, val in [
            (m1, "CLIP Score", clip_s),
            (m2, "Visual Quality", qual_s),
            (m3, "Identity Score", id_s),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val:.3f}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── ControlNet results ────────────────────
    if st.session_state.generated_results is not None and st.session_state.active_mode == MODE_CONTROLNET:
        result = st.session_state.generated_results

        st.markdown('<p class="section-label">Generated Outfits</p>', unsafe_allow_html=True)

        with st.expander("Extracted Pose Map"):
            st.image(result["pose_image"], use_container_width=True)

        imgs = result["images"]
        cols = st.columns(min(len(imgs), 4))
        for i, img in enumerate(imgs):
            with cols[i % len(cols)]:
                st.image(img, caption=f"Variation {i + 1}", use_container_width=True)

        with st.expander("Generation Details"):
            st.markdown(f"**Prompt:** {result['prompt']}")
            st.markdown(f"**Negative Prompt:** {result['negative_prompt']}")
            st.json(result["parameters"])

        st.markdown('<p class="section-label" style="margin-top:1rem">Quick Evaluation</p>', unsafe_allow_html=True)
        from evaluation import compute_clip_score, compute_quality_score, compute_identity_score, compute_consistency, compute_diversity

        clip_avg = sum(compute_clip_score(img, result["prompt"]) for img in imgs) / len(imgs)
        qual_avg = sum(compute_quality_score(img)               for img in imgs) / len(imgs)
        
        pref_img = result.get("reference_image", reference_image)
        id_avg   = 0.0
        if pref_img is not None:
            id_avg   = sum(compute_identity_score(pref_img, img) for img in imgs) / len(imgs)
        consist  = compute_consistency(imgs)
        divers   = compute_diversity(imgs)

        m1, m2, m3, m4, m5 = st.columns(5)
        for col, label, val in [
            (m1, "CLIP Score",    clip_avg),
            (m2, "Visual Quality", qual_avg),
            (m3, "Identity Score", id_avg),
            (m4, "Consistency",   consist),
            (m5, "Diversity",     divers),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val:.3f}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Comparison results ────────────────────
    if st.session_state.comparison_results is not None and st.session_state.active_mode == MODE_COMPARE:
        comparison = st.session_state.comparison_results

        st.markdown('<p class="section-label">Baseline vs Structured</p>', unsafe_allow_html=True)

        with st.expander("Shared Pose Map"):
            st.image(comparison["structured"]["pose_image"], use_container_width=True)

        tab_naive, tab_struct = st.tabs(["Baseline", "Structured"])

        with tab_naive:
            st.markdown(f'<div class="prompt-block">{comparison["naive"]["prompt"]}</div>', unsafe_allow_html=True)
            n_cols = st.columns(min(len(comparison["naive"]["images"]), 4))
            for i, img in enumerate(comparison["naive"]["images"]):
                with n_cols[i % len(n_cols)]:
                    st.image(img, caption=f"Baseline {i+1}", use_container_width=True)

        with tab_struct:
            st.markdown(f'<div class="prompt-block">{comparison["structured"]["prompt"]}</div>', unsafe_allow_html=True)
            s_cols = st.columns(min(len(comparison["structured"]["images"]), 4))
            for i, img in enumerate(comparison["structured"]["images"]):
                with s_cols[i % len(s_cols)]:
                    st.image(img, caption=f"Structured {i+1}", use_container_width=True)

    # ── Evaluation dashboard ──────────────────
    if st.session_state.eval_results is not None and st.session_state.active_mode == MODE_COMPARE:
        import numpy as np
        import pandas as pd
        eval_res = st.session_state.eval_results

        st.markdown('<p class="section-label" style="margin-top:1rem">Evaluation Dashboard</p>', unsafe_allow_html=True)

        def _avg(metrics, attr):
            return float(np.mean([getattr(m, attr) for m in metrics]))

        naive_vals = [
            _avg(eval_res.naive_metrics, "clip_score"),
            _avg(eval_res.naive_metrics, "identity_score"),
            _avg(eval_res.naive_metrics, "quality_score"),
            eval_res.consistency_naive,
            eval_res.diversity_naive,
        ]
        struct_vals = [
            _avg(eval_res.structured_metrics, "clip_score"),
            _avg(eval_res.structured_metrics, "identity_score"),
            _avg(eval_res.structured_metrics, "quality_score"),
            eval_res.consistency_structured,
            eval_res.diversity_structured,
        ]
        metric_names = [
            "CLIP Score", "Identity Preservation",
            "Visual Quality", "Consistency", "Diversity",
        ]

        table = {
            "Metric":     metric_names,
            "Baseline":   [f"{v:.4f}" for v in naive_vals],
            "Structured": [f"{v:.4f}" for v in struct_vals],
            "Delta":      [
                f"{'▲' if (d := s - n) >= 0 else '▼'} {abs(d):.4f}"
                for n, s in zip(naive_vals, struct_vals)
            ],
        }

        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        if eval_res.failure_cases:
            st.markdown("**Failure Cases**")
            for fc in eval_res.failure_cases:
                with st.expander(f"Image {fc['image_index']} — {len(fc['issues'])} issue(s)"):
                    for issue in fc["issues"]:
                        st.warning(issue)
                    st.caption(
                        f"CLIP: {fc['clip_score']:.4f}  "
                        f"|  Quality: {fc['quality_score']:.4f}  "
                        f"|  Identity: {fc['identity_score']:.4f}"
                    )
        else:
            st.success("No failure cases detected.")

        from evaluation import generate_report
        report = generate_report(eval_res)

        with st.expander("Full Evaluation Report"):
            st.markdown(report)

        st.download_button(
            "Download Evaluation Report",
            data=report,
            file_name="evaluation_report.md",
            mime="text/markdown",
        )

    # ── Empty state ───────────────────────────
    no_results = (
        st.session_state.inpaint_results is None
        and st.session_state.generated_results is None
        and st.session_state.comparison_results is None
    )
    if no_results:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center;
            justify-content:center; min-height:240px;
            text-align:center; padding:2.5rem 2rem;
        ">
            <div style="font-size:2.5rem;opacity:0.15;margin-bottom:1rem">◈</div>
            <p style="font-size:0.82rem;line-height:1.8;max-width:360px;color:#374151;">
                Upload a reference image, choose a mode and parameters on the left,
                then click <strong>Run Generation</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Fashion Outfit Generator &nbsp;|&nbsp; Stable Diffusion 1.5 + ControlNet + IP-Adapter &nbsp;|&nbsp; CS-5588
</div>
""", unsafe_allow_html=True)
