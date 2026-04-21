"""
evaluation.py — Evaluation metrics for the Fashion Outfit Generator.

Implements metrics for prompt alignment, identity preservation,
visual quality, consistency, and diversity. Also provides baseline
vs. structured comparison and failure case detection.
"""

import numpy as np
from PIL import Image
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    """Container for a single image's evaluation metrics."""
    clip_score: float = 0.0
    identity_score: float = 0.0
    quality_score: float = 0.0
    prompt: str = ""
    is_naive: bool = False


@dataclass
class ComparisonResult:
    """Container for naive vs. structured comparison results."""
    naive_metrics: List[EvaluationResult] = field(default_factory=list)
    structured_metrics: List[EvaluationResult] = field(default_factory=list)
    consistency_naive: float = 0.0
    consistency_structured: float = 0.0
    diversity_naive: float = 0.0
    diversity_structured: float = 0.0
    failure_cases: List[dict] = field(default_factory=list)


# ──────────────────────────────────────────────
# Lazy-loaded models
# ──────────────────────────────────────────────
_clip_model = None
_clip_processor = None


def _load_clip():
    """Lazy-load CLIP model for text-image similarity."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = _clip_model.to(device)
        _clip_model.eval()
    return _clip_model, _clip_processor


# ──────────────────────────────────────────────
# Prompt Alignment (CLIP Score)
# ──────────────────────────────────────────────
def compute_clip_score(image: Image.Image, prompt: str) -> float:
    """
    Compute CLIP cosine similarity between an image and text prompt.

    Higher scores indicate better alignment between the generated image
    and the input prompt.

    Args:
        image: Generated PIL Image.
        prompt: The text prompt used for generation.

    Returns:
        CLIP similarity score (0.0 to 1.0).
    """
    import torch

    model, processor = _load_clip()
    device = next(model.parameters()).device

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Normalize and compute cosine similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        similarity = (image_embeds @ text_embeds.T).squeeze().item()

    # Clamp to [0, 1] range
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


# ──────────────────────────────────────────────
# Identity Preservation
# ──────────────────────────────────────────────
def compute_identity_score(
    reference_image: Image.Image,
    generated_image: Image.Image,
) -> float:
    """
    Compute identity similarity between reference and generated images.

    Uses a lightweight structural similarity approach based on
    histogram and feature comparison. For stronger face matching,
    consider using DeepFace or InsightFace.

    Args:
        reference_image: Original person photo (PIL Image).
        generated_image: Generated outfit image (PIL Image).

    Returns:
        Identity similarity score (0.0 to 1.0).
    """
    # Resize both to same dimensions for comparison
    size = (256, 256)
    ref = reference_image.resize(size, Image.LANCZOS).convert("RGB")
    gen = generated_image.resize(size, Image.LANCZOS).convert("RGB")

    ref_arr = np.array(ref, dtype=np.float32)
    gen_arr = np.array(gen, dtype=np.float32)

    # Method 1: Histogram similarity (color distribution)
    hist_score = _histogram_similarity(ref_arr, gen_arr)

    # Method 2: Structural similarity (SSIM-inspired)
    ssim_score = _simplified_ssim(ref_arr, gen_arr)

    # Combine scores (weighted average)
    identity_score = 0.4 * hist_score + 0.6 * ssim_score

    return float(np.clip(identity_score, 0.0, 1.0))


def _histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute histogram similarity between two images."""
    score = 0.0
    for c in range(3):  # R, G, B channels
        hist1, _ = np.histogram(img1[:, :, c].flatten(), bins=64, range=(0, 256))
        hist2, _ = np.histogram(img2[:, :, c].flatten(), bins=64, range=(0, 256))

        # Normalize
        hist1 = hist1.astype(np.float32) / (hist1.sum() + 1e-8)
        hist2 = hist2.astype(np.float32) / (hist2.sum() + 1e-8)

        # Bhattacharyya coefficient
        score += np.sum(np.sqrt(hist1 * hist2))

    return score / 3.0


def _simplified_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute a simplified SSIM between two images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.clip(ssim, 0.0, 1.0))


# ──────────────────────────────────────────────
# Visual Quality Assessment
# ──────────────────────────────────────────────
def compute_quality_score(image: Image.Image) -> float:
    """
    Estimate visual quality of a generated image using
    no-reference metrics (sharpness, contrast, noise).

    A lightweight approach that doesn't require external models.

    Args:
        image: Generated PIL Image.

    Returns:
        Quality score (0.0 to 1.0).
    """
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)

    # Metric 1: Sharpness (Laplacian variance)
    gray = np.mean(img_arr, axis=2)
    laplacian = _laplacian_variance(gray)
    sharpness = min(1.0, laplacian / 500.0)  # Normalize

    # Metric 2: Contrast (standard deviation of luminance)
    contrast = min(1.0, gray.std() / 80.0)  # Normalize

    # Metric 3: Colorfulness
    R, G, B = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness_metric = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    colorfulness = min(1.0, colorfulness_metric / 100.0)

    # Metric 4: Dynamic range
    dynamic_range = (gray.max() - gray.min()) / 255.0

    # Weighted combination
    quality = (
        0.35 * sharpness +
        0.25 * contrast +
        0.20 * colorfulness +
        0.20 * dynamic_range
    )

    return float(np.clip(quality, 0.0, 1.0))


def _laplacian_variance(gray: np.ndarray) -> float:
    """Compute the Laplacian variance (sharpness estimate)."""
    # Simple Laplacian kernel convolution
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    h, w = gray.shape
    padded = np.pad(gray, 1, mode='reflect')
    result = np.zeros_like(gray)

    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)

    return result.var()


# ──────────────────────────────────────────────
# Consistency (Pairwise SSIM across multiple generations)
# ──────────────────────────────────────────────
def compute_consistency(images: List[Image.Image]) -> float:
    """
    Compute consistency across multiple generated images.

    Uses pairwise SSIM to measure how stable the outputs are
    when generating multiple images with the same parameters.

    Args:
        images: List of generated PIL Images.

    Returns:
        Average pairwise SSIM (0.0 to 1.0). Higher = more consistent.
    """
    if len(images) < 2:
        return 1.0  # Single image is perfectly consistent with itself

    size = (256, 256)
    arrays = [np.array(img.resize(size, Image.LANCZOS).convert("RGB"), dtype=np.float32) for img in images]

    scores = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            score = _simplified_ssim(arrays[i], arrays[j])
            scores.append(score)

    return float(np.mean(scores))


# ──────────────────────────────────────────────
# Diversity (Pairwise difference across multiple generations)
# ──────────────────────────────────────────────
def compute_diversity(images: List[Image.Image]) -> float:
    """
    Compute diversity across multiple generated images.

    Uses pairwise pixel difference to measure how varied the outputs are.
    Higher scores indicate more diverse outputs.

    Args:
        images: List of generated PIL Images.

    Returns:
        Diversity score (0.0 to 1.0). Higher = more diverse.
    """
    if len(images) < 2:
        return 0.0  # Single image has no diversity

    size = (256, 256)
    arrays = [np.array(img.resize(size, Image.LANCZOS).convert("RGB"), dtype=np.float32) / 255.0 for img in images]

    diffs = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            diff = np.mean(np.abs(arrays[i] - arrays[j]))
            diffs.append(diff)

    # Normalize: typical range 0.0 - 0.5
    diversity = min(1.0, np.mean(diffs) * 2.0)
    return float(diversity)


# ──────────────────────────────────────────────
# Failure Case Detection
# ──────────────────────────────────────────────
def detect_failure_cases(
    images: List[Image.Image],
    prompts: List[str],
    reference_image: Image.Image,
    clip_threshold: float = 0.55,
    quality_threshold: float = 0.3,
    identity_threshold: float = 0.2,
) -> List[dict]:
    """
    Detect failure cases in generated images.

    Identifies images with low prompt alignment, poor quality,
    or lost identity.

    Args:
        images: List of generated PIL Images.
        prompts: Corresponding prompts.
        reference_image: Original person photo.
        clip_threshold: Minimum acceptable CLIP score.
        quality_threshold: Minimum acceptable quality score.
        identity_threshold: Minimum acceptable identity score.

    Returns:
        List of failure case dictionaries.
    """
    failures = []

    for i, (image, prompt) in enumerate(zip(images, prompts)):
        issues = []

        clip_score = compute_clip_score(image, prompt)
        if clip_score < clip_threshold:
            issues.append(f"Low prompt alignment (CLIP: {clip_score:.3f} < {clip_threshold})")

        quality = compute_quality_score(image)
        if quality < quality_threshold:
            issues.append(f"Poor visual quality (Score: {quality:.3f} < {quality_threshold})")

        identity = compute_identity_score(reference_image, image)
        if identity < identity_threshold:
            issues.append(f"Identity loss (Score: {identity:.3f} < {identity_threshold})")

        if issues:
            failures.append({
                "image_index": i,
                "issues": issues,
                "clip_score": clip_score,
                "quality_score": quality,
                "identity_score": identity,
            })

    return failures


# ──────────────────────────────────────────────
# Full Evaluation Pipeline
# ──────────────────────────────────────────────
def evaluate_single(
    image: Image.Image,
    prompt: str,
    reference_image: Image.Image,
    is_naive: bool = False,
) -> EvaluationResult:
    """
    Run full evaluation on a single generated image.

    Args:
        image: Generated PIL Image.
        prompt: The prompt used.
        reference_image: Original person photo.
        is_naive: Whether this used a naive prompt.

    Returns:
        EvaluationResult dataclass.
    """
    return EvaluationResult(
        clip_score=compute_clip_score(image, prompt),
        identity_score=compute_identity_score(reference_image, image),
        quality_score=compute_quality_score(image),
        prompt=prompt,
        is_naive=is_naive,
    )


def evaluate_comparison(
    naive_result: dict,
    structured_result: dict,
    reference_image: Image.Image,
) -> ComparisonResult:
    """
    Run full evaluation comparing naive vs. structured generation results.

    Args:
        naive_result: Output from pipeline.generate(use_naive_prompt=True).
        structured_result: Output from pipeline.generate(use_naive_prompt=False).
        reference_image: Original person photo.

    Returns:
        ComparisonResult with all metrics and failure analysis.
    """
    # Evaluate individual images
    naive_metrics = [
        evaluate_single(img, naive_result["prompt"], reference_image, is_naive=True)
        for img in naive_result["images"]
    ]
    structured_metrics = [
        evaluate_single(img, structured_result["prompt"], reference_image, is_naive=False)
        for img in structured_result["images"]
    ]

    # Consistency & diversity
    consistency_naive = compute_consistency(naive_result["images"])
    consistency_structured = compute_consistency(structured_result["images"])
    diversity_naive = compute_diversity(naive_result["images"])
    diversity_structured = compute_diversity(structured_result["images"])

    # Failure detection
    all_images = naive_result["images"] + structured_result["images"]
    all_prompts = (
        [naive_result["prompt"]] * len(naive_result["images"]) +
        [structured_result["prompt"]] * len(structured_result["images"])
    )
    failures = detect_failure_cases(all_images, all_prompts, reference_image)

    return ComparisonResult(
        naive_metrics=naive_metrics,
        structured_metrics=structured_metrics,
        consistency_naive=consistency_naive,
        consistency_structured=consistency_structured,
        diversity_naive=diversity_naive,
        diversity_structured=diversity_structured,
        failure_cases=failures,
    )


def generate_report(comparison: ComparisonResult) -> str:
    """
    Generate a markdown evaluation report from comparison results.

    Args:
        comparison: ComparisonResult from evaluate_comparison().

    Returns:
        Markdown-formatted report string.
    """
    lines = ["# Evaluation Report: Naive vs. Structured Prompts\n"]

    # Average metrics
    def avg(metrics, attr):
        values = [getattr(m, attr) for m in metrics]
        return np.mean(values) if values else 0.0

    lines.append("## Summary Metrics\n")
    lines.append("| Metric | Naive (Baseline) | Structured | Δ (Improvement) |")
    lines.append("|--------|-----------------|------------|-----------------|")

    for attr, label in [
        ("clip_score", "CLIP Score (Prompt Alignment)"),
        ("identity_score", "Identity Preservation"),
        ("quality_score", "Visual Quality"),
    ]:
        naive_avg = avg(comparison.naive_metrics, attr)
        struct_avg = avg(comparison.structured_metrics, attr)
        delta = struct_avg - naive_avg
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label} | {naive_avg:.4f} | {struct_avg:.4f} | {sign}{delta:.4f} |")

    lines.append(f"| Consistency | {comparison.consistency_naive:.4f} | {comparison.consistency_structured:.4f} | {comparison.consistency_structured - comparison.consistency_naive:+.4f} |")
    lines.append(f"| Diversity | {comparison.diversity_naive:.4f} | {comparison.diversity_structured:.4f} | {comparison.diversity_structured - comparison.diversity_naive:+.4f} |")

    # Failure cases
    lines.append("\n## Failure Cases\n")
    if comparison.failure_cases:
        for fc in comparison.failure_cases:
            lines.append(f"### Image {fc['image_index']}")
            for issue in fc["issues"]:
                lines.append(f"- ⚠️ {issue}")
            lines.append("")
    else:
        lines.append("No failure cases detected. ✅\n")

    # Prompts used
    lines.append("## Prompts Used\n")
    if comparison.naive_metrics:
        lines.append(f"**Naive:** `{comparison.naive_metrics[0].prompt}`\n")
    if comparison.structured_metrics:
        lines.append(f"**Structured:** `{comparison.structured_metrics[0].prompt}`\n")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Quick test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Test with dummy images
    from PIL import Image

    dummy_ref = Image.new("RGB", (512, 768), color=(120, 90, 70))
    dummy_gen = Image.new("RGB", (512, 768), color=(130, 95, 75))

    print("Quality score:", compute_quality_score(dummy_gen))
    print("Identity score:", compute_identity_score(dummy_ref, dummy_gen))
    print("Consistency:", compute_consistency([dummy_ref, dummy_gen]))
    print("Diversity:", compute_diversity([dummy_ref, dummy_gen]))
    print("\nEvaluation module loaded successfully.")
