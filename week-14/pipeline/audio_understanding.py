"""
Cinematic Memory — Audio Understanding Module
Uses Whisper-small to transcribe voice memos + classify emotional tone.
"""
from __future__ import annotations
import os, logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_whisper_model     = None
_whisper_processor = None
_emotion_pipeline  = None
_device            = None


def _load_whisper():
    global _whisper_model, _whisper_processor, _device
    if _whisper_model is not None:
        return
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        from config import WHISPER_MODEL_ID
        logger.info(f"Loading Whisper on {_device}…")
        _whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
        _whisper_model     = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID).to(_device)
        logger.info("Whisper loaded ✓")
    except Exception as e:
        logger.warning(f"Whisper load failed: {e}. Using mock transcription.")


def _load_audio(file_path: str, sample_rate: int = 16000):
    """Load audio file as numpy array at target sample rate."""
    try:
        import librosa
        audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
        return audio
    except ImportError:
        try:
            import soundfile as sf
            audio, sr = sf.read(file_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != sample_rate:
                import scipy.signal as signal
                audio = signal.resample(audio, int(len(audio) * sample_rate / sr))
            return audio
        except Exception as e:
            logger.error(f"Audio load failed: {e}")
            return None


def _transcribe_whisper(audio_array, language: str = "en") -> List[dict]:
    """Run Whisper on audio array, return list of {start, end, text} segments."""
    import torch
    import numpy as np
    from config import SAMPLE_RATE

    # Chunk audio into 30s windows (Whisper limit)
    chunk_size = 30 * SAMPLE_RATE
    segments   = []
    offset     = 0.0

    for start_idx in range(0, len(audio_array), chunk_size):
        chunk = audio_array[start_idx:start_idx + chunk_size]
        inputs = _whisper_processor(
            chunk, sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(_device)

        with torch.no_grad():
            predicted_ids = _whisper_model.generate(inputs, language=language)

        text = _whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        if text:
            duration = len(chunk) / SAMPLE_RATE
            segments.append({
                "start": offset,
                "end":   offset + duration,
                "text":  text,
            })
        offset += len(chunk) / SAMPLE_RATE

    return segments


def _classify_emotion_simple(text: str) -> "EmotionTag":
    """
    Rule-based emotion from transcript keywords.
    Used when no ML emotion model available.
    """
    from utils.data_schemas import EmotionTag
    text_lower = text.lower()

    keyword_map = {
        EmotionTag.JOYFUL:       ["happy","joy","laugh","smile","fun","wonderful","amazing","love"],
        EmotionTag.NOSTALGIC:    ["remember","miss","used to","back then","childhood","years ago","old days"],
        EmotionTag.REFLECTIVE:   ["think","wonder","realize","understand","lesson","looking back","appreciate"],
        EmotionTag.SAD:          ["sad","cry","tears","loss","miss","gone","difficult","hard","mourn"],
        EmotionTag.EXCITED:      ["excited","can't wait","incredible","wow","unbelievable","best","amazing"],
        EmotionTag.CELEBRATORY:  ["celebrate","cheers","congrats","wedding","anniversary","birthday","together"],
        EmotionTag.TENDER:       ["love you","family","together","hold","embrace","close","dear","heart"],
    }

    scores = {emo: 0 for emo in EmotionTag}
    for emo, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text_lower:
                scores[emo] += 1

    best = max(scores, key=lambda e: scores[e])
    return best if scores[best] > 0 else EmotionTag.NEUTRAL


def _mock_transcribe(file_path: str, audio_id: str) -> "AudioMetadata":
    from utils.data_schemas import AudioMetadata, TranscriptSegment, EmotionTag
    mock_text = (
        "It was one of those perfect days where everything felt exactly right. "
        "We laughed, we explored, and somewhere in between, we created memories "
        "that I know will last a lifetime. I keep thinking back to this moment."
    )
    return AudioMetadata(
        audio_id=audio_id, file_path=file_path, duration_s=15.0,
        transcript=mock_text, language="en",
        overall_emotion=EmotionTag.NOSTALGIC,
        segments=[
            TranscriptSegment(0, 5,  mock_text[:80],  emotion=EmotionTag.REFLECTIVE),
            TranscriptSegment(5, 10, mock_text[80:160], emotion=EmotionTag.JOYFUL),
            TranscriptSegment(10, 15, mock_text[160:],  emotion=EmotionTag.NOSTALGIC),
        ]
    )


def transcribe_audio(file_path: str, audio_id: Optional[str] = None) -> "AudioMetadata":
    """
    Main entry point for Audio Understanding Module.
    Args:
        file_path: path to voice memo / audio file
        audio_id:  optional ID; auto-generated if None
    Returns:
        AudioMetadata with timestamped transcript and emotion labels
    """
    from utils.data_schemas import AudioMetadata, TranscriptSegment, EmotionTag
    import numpy as np

    if audio_id is None:
        audio_id = Path(file_path).stem

    _load_whisper()

    if _whisper_model is None:
        return _mock_transcribe(file_path, audio_id)

    audio = _load_audio(file_path)
    if audio is None:
        return _mock_transcribe(file_path, audio_id)

    try:
        duration_s = len(audio) / 16000
        raw_segs   = _transcribe_whisper(audio)
        full_text  = " ".join(s["text"] for s in raw_segs)

        segments = []
        for seg in raw_segs:
            emo = _classify_emotion_simple(seg["text"])
            segments.append(TranscriptSegment(
                start=seg["start"], end=seg["end"],
                text=seg["text"], emotion=emo
            ))

        overall_emotion = _classify_emotion_simple(full_text)

        return AudioMetadata(
            audio_id=audio_id, file_path=file_path,
            duration_s=duration_s, transcript=full_text,
            segments=segments, overall_emotion=overall_emotion,
        )

    except Exception as e:
        logger.error(f"Transcription failed for {file_path}: {e}")
        return _mock_transcribe(file_path, audio_id)


def transcribe_batch(file_paths: List[str]) -> List["AudioMetadata"]:
    """Transcribe list of audio files."""
    results = []
    for i, fp in enumerate(file_paths):
        aid = f"audio_{i:04d}_{Path(fp).stem}"
        logger.info(f"Transcribing [{i+1}/{len(file_paths)}]: {Path(fp).name}")
        results.append(transcribe_audio(fp, audio_id=aid))
    return results
