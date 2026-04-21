"""
prompt_engine.py — Data-driven prompt generation for the Fashion Outfit Generator.

Converts structured inputs (occasion, style, color palette) into detailed
Stable Diffusion prompts. Supports both naive and structured prompt modes
for baseline comparison during evaluation.
"""

import random
from config import (
    OUTFIT_DESCRIPTIONS,
    FALLBACK_OUTFIT_BY_OCCASION,
    FALLBACK_OUTFIT_BY_STYLE,
    NEGATIVE_PROMPT,
)


# ──────────────────────────────────────────────
# Core Prompt Template
# ──────────────────────────────────────────────
STRUCTURED_TEMPLATE = (
    "A full-body photo of a person wearing {outfit_description}, "
    "suitable for a {occasion}, in {style} style, with {color_palette}, "
    "highly detailed, realistic lighting, professional fashion photography, "
    "8k uhd, high resolution, sharp focus"
)

NAIVE_TEMPLATES = {
    "wedding": "a person wearing formal clothes for a wedding",
    "job interview": "a person wearing professional clothes for an interview",
    "gym workout": "a person wearing gym clothes",
    "casual outing": "a person wearing casual clothes",
    "date night": "a person wearing nice clothes for a date",
    "business meeting": "a person wearing business clothes",
    "beach party": "a person wearing beach clothes",
    "graduation ceremony": "a person wearing graduation clothes",
    "cocktail party": "a person wearing cocktail party clothes",
    "music festival": "a person wearing festival clothes",
    "brunch": "a person wearing brunch clothes",
    "office work": "a person wearing office clothes",
}


def get_outfit_description(occasion: str, style: str) -> str:
    """
    Retrieve a specific outfit description for the given occasion + style combo.

    Falls back to occasion-level or style-level descriptions if the exact
    combination is not available in the mapping.

    Args:
        occasion: The occasion (e.g., "wedding", "gym workout")
        style: The style (e.g., "formal", "streetwear")

    Returns:
        A detailed garment description string.
    """
    key = (occasion.lower(), style.lower())

    # Try exact match first
    if key in OUTFIT_DESCRIPTIONS:
        descriptions = OUTFIT_DESCRIPTIONS[key]
        return random.choice(descriptions)

    # Fallback: combine occasion-level and style-level descriptions
    occasion_desc = FALLBACK_OUTFIT_BY_OCCASION.get(
        occasion.lower(),
        "well-coordinated outfit with appropriate accessories"
    )
    style_desc = FALLBACK_OUTFIT_BY_STYLE.get(
        style.lower(),
        "stylish and well-put-together clothing"
    )

    return f"{occasion_desc} in a {style_desc} aesthetic"


def generate_structured_prompt(
    occasion: str,
    style: str,
    color_palette: str,
    outfit_override: str = None,
) -> str:
    """
    Generate a fully structured prompt from input parameters.

    Args:
        occasion: The event type (e.g., "wedding", "gym workout")
        style: The design language (e.g., "formal", "streetwear")
        color_palette: Color description (e.g., "earth tones with browns and olive greens")
        outfit_override: Optional manual outfit description (overrides lookup)

    Returns:
        A detailed prompt string ready for Stable Diffusion.
    """
    if outfit_override:
        outfit_desc = outfit_override
    else:
        outfit_desc = get_outfit_description(occasion, style)

    prompt = STRUCTURED_TEMPLATE.format(
        outfit_description=outfit_desc,
        occasion=occasion,
        style=style,
        color_palette=color_palette,
    )

    return prompt


def generate_naive_prompt(occasion: str) -> str:
    """
    Generate a naive (baseline) prompt for comparison.

    These are intentionally simple and lack the detail of structured prompts,
    serving as a baseline for evaluation.

    Args:
        occasion: The event type.

    Returns:
        A simple prompt string.
    """
    return NAIVE_TEMPLATES.get(
        occasion.lower(),
        f"a person wearing clothes for {occasion}"
    )


def get_negative_prompt() -> str:
    """Return the standard negative prompt for artifact suppression."""
    return NEGATIVE_PROMPT


def generate_prompt_pair(
    occasion: str,
    style: str,
    color_palette: str,
    outfit_override: str = None,
) -> dict:
    """
    Generate both naive and structured prompts for side-by-side comparison.

    Args:
        occasion: The event type.
        style: The design language.
        color_palette: Color description.
        outfit_override: Optional manual outfit description.

    Returns:
        Dictionary with 'naive', 'structured', and 'negative' prompt strings.
    """
    return {
        "naive": generate_naive_prompt(occasion),
        "structured": generate_structured_prompt(
            occasion, style, color_palette, outfit_override
        ),
        "negative": get_negative_prompt(),
    }


def batch_generate_prompts(
    occasion: str,
    style: str,
    color_palette: str,
    count: int = 4,
) -> list:
    """
    Generate multiple unique structured prompts for the same input parameters.

    Each prompt may use a different outfit description from the taxonomy
    to provide diversity while maintaining consistency in occasion/style.

    Args:
        occasion: The event type.
        style: The design language.
        color_palette: Color description.
        count: Number of prompts to generate.

    Returns:
        List of structured prompt strings.
    """
    prompts = []
    key = (occasion.lower(), style.lower())
    descriptions = OUTFIT_DESCRIPTIONS.get(key, [])

    for i in range(count):
        if descriptions and i < len(descriptions):
            # Use different descriptions from the taxonomy
            outfit = descriptions[i % len(descriptions)]
        else:
            outfit = get_outfit_description(occasion, style)

        prompt = generate_structured_prompt(occasion, style, color_palette, outfit)
        prompts.append(prompt)

    return prompts


# ──────────────────────────────────────────────
# Quick test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    pair = generate_prompt_pair(
        occasion="wedding",
        style="formal",
        color_palette="navy blue and gold accents",
    )

    print("=" * 70)
    print("NAIVE PROMPT:")
    print(pair["naive"])
    print()
    print("STRUCTURED PROMPT:")
    print(pair["structured"])
    print()
    print("NEGATIVE PROMPT:")
    print(pair["negative"])
    print("=" * 70)

    print("\nBATCH PROMPTS:")
    batch = batch_generate_prompts("date night", "edgy", "bold red and black", count=3)
    for i, p in enumerate(batch, 1):
        print(f"\n  [{i}] {p}")
