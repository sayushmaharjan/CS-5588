"""
control.py — Pose extraction and identity encoding for the Fashion Outfit Generator.

Uses ControlNet's OpenPose detector to extract body pose from a reference image,
and prepares the reference image for IP-Adapter identity conditioning.
"""

import numpy as np
from PIL import Image

# Lazy imports to avoid loading heavy models at import time
_openpose_detector = None


def _load_openpose_detector():
    """Lazy-load the OpenPose detector from controlnet_aux."""
    global _openpose_detector
    if _openpose_detector is None:
        from controlnet_aux import OpenposeDetector
        _openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    return _openpose_detector


def extract_pose(
    image: Image.Image,
    detect_hand: bool = False,
    detect_face: bool = False,
) -> Image.Image:
    """
    Extract an OpenPose skeleton map from a reference image.

    The resulting skeleton map is used to condition ControlNet so the generated
    image maintains the same body pose as the reference.

    Args:
        image: PIL Image of the reference person.
        detect_hand: Whether to include hand keypoints (slower but more detail).
        detect_face: Whether to include face keypoints.

    Returns:
        PIL Image of the OpenPose skeleton map.
    """
    detector = _load_openpose_detector()

    pose_image = detector(
        image,
        hand_and_face=detect_hand or detect_face,
    )

    return pose_image


def prepare_reference_image(
    image: Image.Image,
    target_width: int = 512,
    target_height: int = 768,
) -> Image.Image:
    """
    Prepare the reference image for IP-Adapter identity conditioning.

    Resizes and center-crops the image to the target dimensions while
    maintaining aspect ratio.

    Args:
        image: Original PIL Image of the reference person.
        target_width: Target width for the output image.
        target_height: Target height for the output image.

    Returns:
        Preprocessed PIL Image ready for IP-Adapter.
    """
    # Calculate resize dimensions preserving aspect ratio
    img_w, img_h = image.size
    aspect = img_w / img_h
    target_aspect = target_width / target_height

    if aspect > target_aspect:
        # Image is wider — fit height, crop width
        new_h = target_height
        new_w = int(new_h * aspect)
    else:
        # Image is taller — fit width, crop height
        new_w = target_width
        new_h = int(new_w / aspect)

    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to exact target dimensions
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    image = image.crop((left, top, left + target_width, top + target_height))

    return image


def prepare_pose_image(
    pose_image: Image.Image,
    target_width: int = 512,
    target_height: int = 768,
) -> Image.Image:
    """
    Resize the pose map to match the generation target dimensions.

    Args:
        pose_image: OpenPose skeleton map.
        target_width: Target width.
        target_height: Target height.

    Returns:
        Resized pose map.
    """
    return pose_image.resize((target_width, target_height), Image.LANCZOS)


def extract_and_prepare(
    reference_image: Image.Image,
    target_width: int = 512,
    target_height: int = 768,
    detect_hand: bool = False,
    detect_face: bool = False,
) -> dict:
    """
    Full control extraction pipeline: pose + reference preparation.

    Args:
        reference_image: Original PIL Image of the person.
        target_width: Target generation width.
        target_height: Target generation height.
        detect_hand: Include hand keypoints.
        detect_face: Include face keypoints.

    Returns:
        Dictionary with:
            - 'pose_image': OpenPose skeleton map (PIL Image)
            - 'reference_image': Preprocessed reference for IP-Adapter (PIL Image)
            - 'original_size': Original image dimensions tuple
    """
    original_size = reference_image.size

    # Prepare the reference image (resize + crop)
    prepared_ref = prepare_reference_image(
        reference_image, target_width, target_height
    )

    # Extract pose from the prepared reference
    pose_map = extract_pose(prepared_ref, detect_hand, detect_face)

    # Ensure pose map matches target dimensions
    pose_map = prepare_pose_image(pose_map, target_width, target_height)

    return {
        "pose_image": pose_map,
        "reference_image": prepared_ref,
        "original_size": original_size,
    }


def validate_input_image(image: Image.Image) -> dict:
    """
    Validate that the input image is suitable for processing.

    Checks for minimum dimensions, correct mode, and basic quality.

    Args:
        image: PIL Image to validate.

    Returns:
        Dictionary with 'valid' (bool) and 'message' (str).
    """
    w, h = image.size

    if w < 128 or h < 128:
        return {
            "valid": False,
            "message": f"Image too small ({w}x{h}). Minimum 128x128 pixels required."
        }

    if w > 4096 or h > 4096:
        return {
            "valid": False,
            "message": f"Image too large ({w}x{h}). Maximum 4096x4096 pixels."
        }

    # Convert to RGB if needed
    if image.mode not in ("RGB", "RGBA"):
        return {
            "valid": False,
            "message": f"Unsupported image mode: {image.mode}. Use RGB or RGBA."
        }

    # Check if image is mostly black/white (likely not a person photo)
    img_array = np.array(image.convert("RGB"))
    mean_pixel = img_array.mean()
    if mean_pixel < 10 or mean_pixel > 245:
        return {
            "valid": False,
            "message": "Image appears to be blank or overexposed. Please use a photo of a person."
        }

    return {
        "valid": True,
        "message": f"Image validated: {w}x{h} pixels, {image.mode} mode."
    }


# ──────────────────────────────────────────────
# Quick test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Create a dummy test image
    test_img = Image.new("RGB", (400, 600), color=(128, 128, 128))
    result = validate_input_image(test_img)
    print(f"Validation: {result}")

    prepared = prepare_reference_image(test_img)
    print(f"Prepared image size: {prepared.size}")
