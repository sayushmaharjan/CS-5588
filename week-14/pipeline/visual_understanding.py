"""
Cinematic Memory — Visual Understanding Module
Uses CLIP to extract scene type, objects, emotions, salience from photos/videos.
"""
from __future__ import annotations
import os, sys, logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — only load when module is invoked
_clip_model   = None
_clip_processor = None
_device       = None


def _load_clip():
    global _clip_model, _clip_processor, _device
    if _clip_model is not None:
        return
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP on {_device}…")
        from config import CLIP_MODEL_ID
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model      = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(_device)
        logger.info("CLIP loaded ✓")
    except Exception as e:
        logger.warning(f"CLIP load failed: {e}. Using mock visual analysis.")


# ── Label Banks ──────────────────────────────────────────────────────────

SCENE_LABELS = [
    "a photo of a beach with ocean waves",
    "a wedding ceremony or reception",
    "a city street or urban environment",
    "an indoor room or interior space",
    "a nature scene with trees and greenery",
    "a party or celebration with people",
    "a travel scene at an airport or tourist site",
    "a portrait of a person close up",
]

SCENE_MAP = ["beach","wedding","city","indoors","nature","party","travel","portrait"]

EMOTION_LABELS = [
    "people smiling and looking joyful",
    "people hugging or showing affection, tender moment",
    "a reflective quiet solitary moment",
    "a sad or emotional moment with tears",
    "an excited energetic celebration",
    "a nostalgic old memory or faded moment",
    "a celebratory festive scene",
    "a neutral everyday moment",
]

EMOTION_MAP = ["joyful","tender","reflective","sad","excited","nostalgic","celebratory","neutral"]

SALIENCE_LABELS = [
    "a cinematic dramatic visually stunning scene",
    "a dull or blurry or ordinary photo",
]

OBJECT_LABELS = [
    "people","faces","food","drinks","flowers","rings","cake",
    "sunset","water","mountains","buildings","cars","animals",
    "music instruments","balloons","candles","fireworks",
]


def _clip_score(image, text_labels: List[str]) -> List[float]:
    """Return softmax similarity scores for image vs text labels."""
    import torch
    inputs = _clip_processor(
        text=text_labels, images=image,
        return_tensors="pt", padding=True, truncation=True
    ).to(_device)
    with torch.no_grad():
        outputs  = _clip_model(**inputs)
        logits   = outputs.logits_per_image[0]
        probs    = logits.softmax(dim=0).cpu().numpy()
    return probs.tolist()


def _open_image(file_path: str):
    """Open image, handle both photos and video first-frame extraction."""
    from PIL import Image
    ext = Path(file_path).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]:
        return Image.open(file_path).convert("RGB")
    elif ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        # Extract first frame via OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
        except ImportError:
            logger.warning("OpenCV not available for video frame extraction")
    return None


def _get_exif_timestamp(file_path: str) -> Optional[str]:
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img = Image.open(file_path)
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                if TAGS.get(tag_id) in ("DateTime", "DateTimeOriginal"):
                    return str(value)
    except Exception:
        pass
    return None


def _mock_analyze(file_path: str, media_id: str):
    """Fallback when CLIP unavailable — returns plausible mock data."""
    from utils.data_schemas import VisualMetadata, MediaType, SceneType, EmotionTag
    ext = Path(file_path).suffix.lower()
    mtype = MediaType.VIDEO if ext in [".mp4",".mov",".avi",".mkv"] else MediaType.PHOTO
    import random, hashlib
    rng = random.Random(hashlib.md5(file_path.encode()).hexdigest())
    scene  = rng.choice(list(SceneType))
    emo    = rng.choice(list(EmotionTag))
    return VisualMetadata(
        media_id=media_id, file_path=file_path, media_type=mtype,
        scene_type=scene, objects=["people","smiles"],
        emotions=[emo], salience_score=round(rng.uniform(0.4, 0.9), 2),
        description=f"Scene: {scene.value}, mood: {emo.value}",
        exif_timestamp=_get_exif_timestamp(file_path) if mtype == MediaType.PHOTO else None,
    )


def analyze_media(file_path: str, media_id: Optional[str] = None) -> "VisualMetadata":
    """
    Main entry point for Visual Understanding Module.
    Args:
        file_path: path to photo or video file
        media_id:  optional ID; auto-generated from filename if None
    Returns:
        VisualMetadata with scene, emotions, objects, salience
    """
    from utils.data_schemas import VisualMetadata, MediaType, SceneType, EmotionTag

    if media_id is None:
        media_id = Path(file_path).stem

    ext = Path(file_path).suffix.lower()
    mtype = MediaType.VIDEO if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"] else MediaType.PHOTO

    _load_clip()
    if _clip_model is None:
        return _mock_analyze(file_path, media_id)

    image = _open_image(file_path)
    if image is None:
        return _mock_analyze(file_path, media_id)

    try:
        # Scene classification
        scene_scores = _clip_score(image, SCENE_LABELS)
        scene_idx    = int(np.argmax(scene_scores))
        scene_type   = SceneType(SCENE_MAP[scene_idx]) if scene_idx < len(SCENE_MAP) else SceneType.UNKNOWN

        # Emotion classification (top-2)
        emo_scores = _clip_score(image, EMOTION_LABELS)
        emo_idxs   = np.argsort(emo_scores)[::-1][:2]
        emotions   = [EmotionTag(EMOTION_MAP[i]) for i in emo_idxs if i < len(EMOTION_MAP)]

        # Salience score
        sal_scores     = _clip_score(image, SALIENCE_LABELS)
        salience_score = float(sal_scores[0])  # prob of being "cinematic"

        # Object detection (threshold 0.2)
        obj_scores = _clip_score(image, [f"a photo with {o}" for o in OBJECT_LABELS])
        objects    = [OBJECT_LABELS[i] for i, s in enumerate(obj_scores) if s > 0.2]

        description = (
            f"Scene: {scene_type.value}. "
            f"Mood: {', '.join(e.value for e in emotions)}. "
            f"Contains: {', '.join(objects[:5]) if objects else 'general content'}."
        )

        return VisualMetadata(
            media_id=media_id, file_path=file_path, media_type=mtype,
            scene_type=scene_type, objects=objects,
            emotions=emotions, salience_score=salience_score,
            description=description,
            exif_timestamp=_get_exif_timestamp(file_path) if mtype == MediaType.PHOTO else None,
        )

    except Exception as e:
        logger.error(f"CLIP analysis failed for {file_path}: {e}")
        return _mock_analyze(file_path, media_id)


def analyze_batch(file_paths: List[str]) -> List["VisualMetadata"]:
    """Analyze a list of media files. Returns list of VisualMetadata in same order."""
    results = []
    for i, fp in enumerate(file_paths):
        mid = f"media_{i:04d}_{Path(fp).stem}"
        logger.info(f"Analyzing [{i+1}/{len(file_paths)}]: {Path(fp).name}")
        results.append(analyze_media(fp, media_id=mid))
    return results
