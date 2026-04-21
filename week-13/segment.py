# segment.py
from PIL import ImageFilter
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import numpy as np
from PIL import Image

# ATR labels: 0=bg, 5=upper-clothes, 6=dress, 7=coat, 9=pants, 10=skirt, 12=scarf
# segment.py
CLOTHING_LABELS = {1, 4, 5, 6, 7, 8, 9, 10, 16, 17}  # added more

_processor = None
_model = None

def _load_model():
    global _processor, _model
    if _model is None:
        _processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    return _processor, _model

def get_clothing_mask(image: Image.Image, dilate: int = 21) -> Image.Image:
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    seg = upsampled.argmax(dim=1).squeeze().numpy()

    # DEBUG — print all detected labels + pixel counts
    unique, counts = np.unique(seg, return_counts=True)
    print("=== SEGMENTATION LABELS DETECTED ===")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} pixels")
    print("=====================================")

    # Print what labels were detected — helps debug
    unique_labels = np.unique(seg)
    print(f"Detected segmentation labels: {unique_labels}")

    mask = np.zeros_like(seg, dtype=np.uint8)
    for label in CLOTHING_LABELS:
        mask[seg == label] = 255

    # Always protect the face/head region (top 18% of image height).
    # This prevents inpainting from distorting the face, especially on
    # small or low-resolution input images where the segmenter may leak
    # into hair/neck pixels.
    h_px, w_px = mask.shape
    face_protect_y = int(h_px * 0.18)
    mask[:face_protect_y, :] = 0

    # If mask too small (<10% of image), fallback: mask everything below neck
    mask_coverage = mask.sum() / (255 * mask.size)
    print(f"Mask coverage: {mask_coverage:.1%}")
    if mask_coverage < 0.10:
        print("Mask too small — using body-below-neck fallback")
        neck_y = int(h_px * 0.20)   # top 20% = head, rest = body
        mask[neck_y:, :] = 255
        mask[:face_protect_y, :] = 0  # re-protect face after fallback

    if dilate > 0:
        if dilate % 2 == 0:
            dilate += 1
        mask_img = Image.fromarray(mask)
        return mask_img.filter(ImageFilter.MaxFilter(dilate))
    return Image.fromarray(mask)