"""
config.py — Central configuration for the Fashion Outfit Generator.

Contains model IDs, default generation parameters, device selection,
occasion/style/color taxonomies, and outfit description mappings.
"""

import torch

# ──────────────────────────────────────────────
# Device Selection
# ──────────────────────────────────────────────
def get_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ──────────────────────────────────────────────
# HuggingFace Model IDs
# ──────────────────────────────────────────────
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-openpose"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "models"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.bin"
IP_ADAPTER_IMAGE_ENCODER = "h94/IP-Adapter"
IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER = "models/image_encoder"

# ──────────────────────────────────────────────
# Generation Defaults
# ──────────────────────────────────────────────
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CONTROLNET_SCALE = 0.8
DEFAULT_IP_ADAPTER_SCALE = 0.6
DEFAULT_NUM_IMAGES = 2
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 768  # Taller for full-body shots
DEFAULT_SEED = None  # None = random

# ──────────────────────────────────────────────
# Negative Prompt
# ──────────────────────────────────────────────
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted face, extra limbs, deformed hands, "
    "watermark, text, cropped, out of frame, bad anatomy, ugly, duplicate, "
    "mutation, mutilated, poorly drawn face, disfigured, extra fingers, "
    "fused fingers, too many fingers, long neck, bad proportions, "
    "signature, username, artist name"
)

# ──────────────────────────────────────────────
# Occasion Taxonomy
# ──────────────────────────────────────────────
OCCASIONS = [
    "wedding",
    "job interview",
    "gym workout",
    "casual outing",
    "date night",
    "business meeting",
    "beach party",
    "graduation ceremony",
    "cocktail party",
    "music festival",
    "brunch",
    "office work",
]

# ──────────────────────────────────────────────
# Style Taxonomy
# ──────────────────────────────────────────────
STYLES = [
    "formal",
    "streetwear",
    "casual",
    "bohemian",
    "minimalist",
    "vintage",
    "athleisure",
    "preppy",
    "edgy",
    "classic",
]

# ──────────────────────────────────────────────
# Color Palette Presets
# ──────────────────────────────────────────────
COLOR_PALETTES = {
    "Earth Tones": "warm earth tones with browns, olive greens, and terracotta",
    "Monochrome Black": "monochrome black and charcoal palette",
    "Pastel": "soft pastel colors with light pink, baby blue, and lavender",
    "Navy & White": "classic navy blue and crisp white combination",
    "Jewel Tones": "rich jewel tones with emerald green, sapphire blue, and ruby red",
    "Neutral Beige": "neutral beige and cream tones with subtle gold accents",
    "Bold Red & Black": "bold red and black statement palette",
    "Ocean Blues": "cool ocean-inspired blues and aqua tones",
    "Forest Green": "deep forest greens with khaki accents",
    "Burgundy & Gold": "luxurious burgundy and gold palette",
    "Custom": "",  # User provides their own
}

# ──────────────────────────────────────────────
# Outfit Description Mapping
# Maps (occasion, style) → specific garment descriptions
# ──────────────────────────────────────────────
OUTFIT_DESCRIPTIONS = {
    # ── Wedding ──────────────────────────────
    ("wedding", "formal"): [
        "an elegant tailored navy suit with a white dress shirt, silk tie, and polished oxford shoes",
        "a sophisticated charcoal three-piece suit with a satin lapel, matching pocket square, and leather dress shoes",
        "a floor-length elegant evening gown with delicate lace detailing and strappy heels",
    ],
    ("wedding", "classic"): [
        "a timeless black tuxedo with a white wing-tip collar shirt, bow tie, and patent leather shoes",
        "an ivory A-line dress with pearl accessories, a structured clutch, and kitten heels",
    ],
    ("wedding", "bohemian"): [
        "a flowing maxi dress with floral embroidery, layered necklaces, and leather sandals",
        "a linen blazer over a loose-fit shirt with relaxed trousers and suede loafers",
    ],

    # ── Job Interview ────────────────────────
    ("job interview", "formal"): [
        "a well-fitted charcoal gray suit with a light blue dress shirt, understated tie, and leather oxfords",
        "a tailored pencil skirt suit with a silk blouse, minimal jewelry, and pointed-toe pumps",
    ],
    ("job interview", "classic"): [
        "a navy blazer with pressed khaki trousers, a white button-down, and brown leather belt with loafers",
        "a structured sheath dress in solid color with a matching blazer and closed-toe heels",
    ],
    ("job interview", "minimalist"): [
        "a slim-cut black suit with a crew-neck white top, no tie, and clean white sneakers",
        "a monochrome ensemble with tailored trousers, a turtleneck, and sleek ankle boots",
    ],

    # ── Gym Workout ──────────────────────────
    ("gym workout", "athleisure"): [
        "fitted performance leggings and a breathable crop top with cushioned running sneakers",
        "moisture-wicking joggers with a compression tank top and cross-training shoes",
    ],
    ("gym workout", "streetwear"): [
        "oversized graphic gym tee with tapered sweatpants and retro high-top sneakers",
        "a cropped hoodie with biker shorts and chunky platform trainers",
    ],
    ("gym workout", "casual"): [
        "comfortable cotton shorts with a relaxed-fit t-shirt and lightweight trainers",
        "a zip-up track jacket over a sports bra with full-length leggings and running shoes",
    ],

    # ── Casual Outing ────────────────────────
    ("casual outing", "casual"): [
        "well-fitted jeans with a striped crew-neck t-shirt, white sneakers, and a canvas tote",
        "a relaxed linen shirt tucked into chino shorts with espadrilles and sunglasses",
    ],
    ("casual outing", "streetwear"): [
        "an oversized hoodie with distressed jeans, chunky sneakers, and a crossbody bag",
        "a graphic band tee with cargo pants, high-top sneakers, and a snapback cap",
    ],
    ("casual outing", "bohemian"): [
        "a flowing printed maxi skirt with a tucked-in camisole, layered bracelets, and gladiator sandals",
        "wide-leg linen pants with an embroidered tunic top, beaded necklace, and woven sandals",
    ],
    ("casual outing", "minimalist"): [
        "a clean-cut white t-shirt with tailored black trousers and minimalist leather sneakers",
        "a neutral-toned linen shirt dress with a thin belt and simple flat sandals",
    ],

    # ── Date Night ───────────────────────────
    ("date night", "formal"): [
        "a slim-fit dark suit jacket over a black turtleneck with tailored trousers and suede chelsea boots",
        "a fitted cocktail dress with subtle shimmer, statement earrings, and strappy heeled sandals",
    ],
    ("date night", "edgy"): [
        "a leather jacket over a silk camisole with skinny jeans, ankle boots, and silver jewelry",
        "a moto jacket with a fitted black dress, combat boots, and a chain necklace",
    ],
    ("date night", "classic"): [
        "a cashmere sweater with well-fitted dark jeans, loafers, and a quality leather watch",
        "a wrap dress in a solid jewel tone with gold accessories and classic pumps",
    ],

    # ── Business Meeting ─────────────────────
    ("business meeting", "formal"): [
        "a double-breasted pinstripe suit with a french-cuff shirt, cufflinks, and cap-toe oxfords",
        "a tailored blazer with a pencil skirt, silk blouse, structured handbag, and pointed-toe heels",
    ],
    ("business meeting", "classic"): [
        "a single-breasted navy suit with a patterned tie, leather belt, and dress shoes",
        "a midi skirt with a tucked-in blouse, a structured blazer, and kitten heels",
    ],
    ("business meeting", "minimalist"): [
        "a collarless suit jacket with slim-fit trousers, a plain white shirt, and suede loafers",
        "monochrome trousers and blazer with a high-neck top and clean-line flats",
    ],

    # ── Beach Party ──────────────────────────
    ("beach party", "casual"): [
        "a printed Hawaiian shirt with linen shorts, flip-flops, and aviator sunglasses",
        "a flowy sundress with a wide-brim straw hat, woven sandals, and a beaded anklet",
    ],
    ("beach party", "bohemian"): [
        "a crochet cover-up over swim trunks with leather sandals and shell jewelry",
        "a tie-dye maxi dress with fringe bag, layered necklaces, and platform espadrilles",
    ],
    ("beach party", "streetwear"): [
        "an oversized mesh tank over board shorts with slide sandals and a bucket hat",
        "a crop top with high-waisted cutoff shorts, platform sneakers, and neon accessories",
    ],

    # ── Graduation Ceremony ──────────────────
    ("graduation ceremony", "formal"): [
        "a classic navy suit with a white dress shirt, silk tie in school colors, and polished loafers",
        "an elegant knee-length dress in a solid color with a matching cardigan and low heels",
    ],
    ("graduation ceremony", "classic"): [
        "a fitted blazer with pressed trousers, a light-colored button-down, and leather shoes",
        "a floral midi dress with a tailored jacket, pearl studs, and nude pumps",
    ],
    ("graduation ceremony", "preppy"): [
        "a seersucker blazer over a polo shirt with chino trousers and boat shoes",
        "a plaid skirt with a fitted blazer, white blouse, knee-high socks, and loafers",
    ],

    # ── Cocktail Party ───────────────────────
    ("cocktail party", "formal"): [
        "a velvet blazer with a silk pocket square, slim-fit trousers, and patent leather shoes",
        "a sleek one-shoulder cocktail dress with metallic clutch and stiletto heels",
    ],
    ("cocktail party", "edgy"): [
        "a sequined bomber jacket over a black silk shirt with leather pants and pointed boots",
        "a fitted metallic mini dress with bold statement jewelry, a leather clutch, and block heels",
    ],
    ("cocktail party", "vintage"): [
        "a retro-cut double-breasted blazer with high-waisted trousers, a skinny tie, and wing-tips",
        "a 1950s-inspired fit-and-flare dress with a pearl choker, gloves, and kitten heels",
    ],

    # ── Music Festival ───────────────────────
    ("music festival", "bohemian"): [
        "a fringed suede vest over a graphic tee with ripped jeans and western boots",
        "a tie-dye crop top with high-waisted denim shorts, platform boots, and a flower crown",
    ],
    ("music festival", "streetwear"): [
        "an oversized vintage band tee with cargo shorts, chunky sneakers, and a fanny pack",
        "a mesh top over a bralette with wide-leg pants, platform sneakers, and layered chains",
    ],
    ("music festival", "edgy"): [
        "a studded leather vest over a fishnet top with black skinny jeans and combat boots",
        "a holographic bodysuit with a utility belt, platform boots, and futuristic sunglasses",
    ],

    # ── Brunch ───────────────────────────────
    ("brunch", "casual"): [
        "a light knit sweater with tailored chinos, clean white sneakers, and a leather watch",
        "a midi wrap skirt with a tucked-in striped top, mules, and a straw clutch",
    ],
    ("brunch", "preppy"): [
        "a pastel polo shirt with pressed shorts, boat shoes, and a woven belt",
        "a gingham dress with a cardigan draped over the shoulders and ballet flats",
    ],
    ("brunch", "minimalist"): [
        "a simple linen shirt with relaxed-fit trousers, espadrilles, and a canvas tote",
        "an oversized neutral-toned cardigan over a tank top with wide-leg pants and slides",
    ],

    # ── Office Work ──────────────────────────
    ("office work", "formal"): [
        "a well-tailored charcoal blazer over a crisp white shirt with dark trousers and oxfords",
        "a structured sheath dress with a thin belt, a tailored cardigan, and low block heels",
    ],
    ("office work", "classic"): [
        "a button-down shirt with a v-neck sweater, chinos, and clean leather loafers",
        "wide-leg trousers with a silk blouse, a statement watch, and pointed-toe flats",
    ],
    ("office work", "minimalist"): [
        "a collarless white shirt with slim black trousers and minimalist leather sneakers",
        "a simple shift dress in a solid neutral color with a sleek belt and clean-line flats",
    ],
}

# ──────────────────────────────────────────────
# Fallback Outfit Descriptions
# Used when the exact (occasion, style) combo is not in the mapping
# ──────────────────────────────────────────────
FALLBACK_OUTFIT_BY_OCCASION = {
    "wedding": "elegant formal attire with polished shoes and refined accessories",
    "job interview": "professional business wear with clean lines and minimal accessories",
    "gym workout": "athletic sportswear with performance sneakers",
    "casual outing": "comfortable everyday clothing with casual shoes",
    "date night": "stylish and well-put-together evening wear",
    "business meeting": "sharp professional attire with quality accessories",
    "beach party": "light and breezy summer wear with sandals",
    "graduation ceremony": "smart semi-formal attire with polished shoes",
    "cocktail party": "chic semi-formal evening wear with statement accessories",
    "music festival": "trendy and expressive festival wear with comfortable footwear",
    "brunch": "relaxed yet put-together weekend wear with comfortable shoes",
    "office work": "professional office attire with clean, polished shoes",
}

FALLBACK_OUTFIT_BY_STYLE = {
    "formal": "well-tailored suit or elegant dress with polished dress shoes",
    "streetwear": "oversized graphic tee with joggers and chunky sneakers",
    "casual": "relaxed-fit jeans with a simple t-shirt and white sneakers",
    "bohemian": "flowing earth-toned layers with sandals and natural jewelry",
    "minimalist": "clean-cut neutral-toned pieces with simple accessories",
    "vintage": "retro-inspired clothing with classic silhouettes and timeless accessories",
    "athleisure": "sporty performance wear with comfortable trainers",
    "preppy": "clean-cut polo or button-down with chinos and loafers",
    "edgy": "leather jacket with dark denim and boots",
    "classic": "timeless tailored pieces with quality fabric and understated accessories",
}
