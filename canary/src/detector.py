"""
detector.py — Deepfake audio detection pipeline.

Two-stage detection:
  Stage 1: Spectral heuristics (fast, no model download required)
  Stage 2: Whisper encoder embeddings → anomaly scoring

Heuristic rules (calibrated from ASVspoof research):
  - Spectral flatness > 0.08 → suspicious (TTS over-smooth)
  - f0 regularity > 0.75 → suspicious (TTS pitch too steady)
  - HNR > 20 dB → suspicious (TTS too clean)
  - Phase randomness < 0.8 → suspicious (vocoder artifact)
  - Mel smoothness < 4.0 dB → suspicious (over-smoothed mel)
  - Silence regularity > 0.7 → suspicious (robotic breath patterns)

Model-based:
  Uses openai/whisper-small encoder to extract embeddings.
  Computes Mahalanobis distance from a "natural speech" reference.
  Reference built from first N seconds of human speech samples.
"""

import numpy as np
import librosa

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    TORCH_AVAILABLE = False
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from .features import AudioFeatures, extract_features, feature_vector


# ── Thresholds (heuristic calibration) ────────────────────────────────────────
THRESHOLDS = {
    "spectral_flatness_mean": (0.07, "high"),    # > threshold = suspicious
    "f0_regularity":          (0.72, "high"),
    "hnr":                    (18.0, "high"),
    "phase_randomness":       (0.85, "low"),     # < threshold = suspicious
    "mel_smoothness":         (3.5,  "low"),
    "silence_regularity":     (0.68, "high"),
    "spectral_flux":          (0.005, "low"),    # TTS flux too low
    "dynamic_range":          (15.0, "low"),     # TTS has low dynamic range
}


@dataclass
class DetectionResult:
    synthetic_probability: float        # 0 = human, 1 = synthetic
    confidence: float                   # model certainty
    heuristic_score: float              # raw heuristic vote
    embedding_score: float              # whisper-based score
    triggered_rules: list               # which heuristic rules fired
    explanation: str                    # human-readable reason
    features: Optional[AudioFeatures]  = None
    processing_time_ms: float          = 0.0
    chunk_index: int                   = 0


class DeepfakeDetector:
    """
    Real-time deepfake audio detector.
    
    Usage:
        detector = DeepfakeDetector(use_model=True)
        result = detector.detect(audio_chunk, sr=16000)
    """

    def __init__(self, use_model: bool = True, device: Optional[str] = None):
        self.use_model = use_model
        if TORCH_AVAILABLE:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
            self.use_model = False  # force heuristic-only if torch missing
        self._whisper_model = None
        self._whisper_processor = None
        self._reference_embeddings = []    # calibration embeddings
        self._reference_mean = None
        self._reference_cov_inv = None
        self._chunk_count = 0

        if use_model:
            self._load_model()

    def _load_model(self):
        """Load Whisper encoder (lazy, cached)."""
        try:
            from transformers import WhisperModel, WhisperFeatureExtractor
            print("[Canary] Loading Whisper encoder...")
            self._whisper_processor = WhisperFeatureExtractor.from_pretrained(
                "openai/whisper-small"
            )
            self._whisper_model = WhisperModel.from_pretrained(
                "openai/whisper-small"
            )
            self._whisper_model.eval()
            self._whisper_model.to(self.device)
            print("[Canary] Whisper encoder ready.")
        except Exception as e:
            print(f"[Canary] Whisper load failed: {e}. Heuristic-only mode.")
            self._whisper_model = None

    def _get_whisper_embedding(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extract mean-pooled encoder embedding from Whisper."""
        if not TORCH_AVAILABLE or self._whisper_model is None:
            return None
        try:
            # Whisper expects 16kHz
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Max 30 seconds (Whisper limit)
            max_samples = 30 * sr
            audio = audio[:max_samples]

            inputs = self._whisper_processor(
                audio, sampling_rate=sr, return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)

            import contextlib
            ctx = torch.no_grad() if TORCH_AVAILABLE and torch is not None else contextlib.nullcontext()
            with ctx:
                encoder_out = self._whisper_model.encoder(input_features)
                # Mean-pool over time dimension
                embedding = encoder_out.last_hidden_state.mean(dim=1)
                return embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"[Canary] Embedding error: {e}")
            return None

    def calibrate(self, human_audio_samples: list, sr: int = 16000):
        """
        Build reference distribution from known human speech.
        Call before inference for better embedding-based scoring.
        
        Args:
            human_audio_samples: list of np.ndarray audio chunks
            sr: sample rate
        """
        embeddings = []
        for audio in human_audio_samples:
            emb = self._get_whisper_embedding(audio, sr)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) >= 2:
            emb_matrix = np.stack(embeddings)
            self._reference_mean = np.mean(emb_matrix, axis=0)
            cov = np.cov(emb_matrix.T)
            # Regularize for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-6
            try:
                self._reference_cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                self._reference_cov_inv = np.eye(cov.shape[0])
            print(f"[Canary] Calibrated on {len(embeddings)} samples.")

    def _embedding_score(self, audio: np.ndarray, sr: int) -> float:
        """
        Mahalanobis distance from natural speech reference.
        High distance → likely synthetic.
        Returns score in [0, 1].
        """
        emb = self._get_whisper_embedding(audio, sr)
        if emb is None:
            return 0.5  # neutral if model unavailable

        if self._reference_mean is not None:
            diff = emb - self._reference_mean
            try:
                dist = np.sqrt(diff @ self._reference_cov_inv @ diff)
                # Sigmoid normalization: dist ~0 = human, dist > 10 = synthetic
                score = 1.0 / (1.0 + np.exp(-0.3 * (dist - 8)))
                return float(np.clip(score, 0, 1))
            except Exception:
                pass

        # Fallback: L2 norm magnitude anomaly (no calibration)
        # Whisper embeddings of synthetic audio tend to cluster differently
        norm = np.linalg.norm(emb)
        # Natural speech norms typically in [40, 80]
        if norm < 30 or norm > 100:
            return 0.65
        return 0.4

    def _heuristic_score(self, feats: AudioFeatures) -> Tuple[float, list]:
        """
        Rule-based scoring from spectral features.
        Returns (score 0-1, triggered rules).
        """
        votes = []
        triggered = []

        def check(name, value, threshold, direction, weight):
            """direction: 'high' = suspicious if > threshold, 'low' = suspicious if <"""
            if direction == "high":
                suspicious = value > threshold
                # Severity = normalised absolute distance from threshold (always >= 0)
                if suspicious:
                    severity = (value - threshold) / (abs(threshold) + 1e-8)
                else:
                    severity = (threshold - value) / (abs(threshold) + 1e-8)
            else:  # direction == "low"
                suspicious = value < threshold
                if suspicious:
                    severity = (threshold - value) / (abs(threshold) + 1e-8)
                else:
                    severity = (value - threshold) / (abs(threshold) + 1e-8)

            severity = abs(severity)  # guard: always positive
            if suspicious:
                score = min(0.5 + severity * 0.3, 1.0)
                triggered.append((name, value, threshold, direction, score))
            else:
                # Safe: further from threshold → lower score (more human-like)
                score = max(0.5 - severity * 0.3, 0.0)
            votes.append(score * weight)

        # Apply rules
        check("spectral_flatness",  feats.spectral_flatness_mean, 0.07,  "high", 2.0)
        check("pitch_regularity",   feats.f0_regularity,          0.72,  "high", 2.5)
        check("hnr",                feats.hnr,                    18.0,  "high", 1.5)
        check("phase_randomness",   feats.phase_randomness,       0.85,  "low",  1.5)
        check("mel_smoothness",     feats.mel_smoothness,         3.5,   "low",  2.0)
        check("silence_regularity", feats.silence_regularity,     0.68,  "high", 1.0)
        check("spectral_flux",      feats.spectral_flux,          0.005, "low",  1.5)
        check("dynamic_range",      feats.dynamic_range,          15.0,  "low",  1.0)

        total_weight = 2.0 + 2.5 + 1.5 + 1.5 + 2.0 + 1.0 + 1.5 + 1.0
        score = sum(votes) / total_weight
        return float(np.clip(score, 0, 1)), triggered

    def detect(
        self,
        audio: np.ndarray,
        sr: int,
        chunk_index: int = 0,
    ) -> DetectionResult:
        """
        Full detection pipeline for one audio chunk.
        
        Args:
            audio: mono float32 waveform
            sr: sample rate
            chunk_index: position in stream
            
        Returns:
            DetectionResult with probability, confidence, explanation
        """
        import time
        t0 = time.time()

        # Ensure mono, float32
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        audio = audio.astype(np.float32)

        # Normalize
        peak = np.max(np.abs(audio)) + 1e-8
        audio = audio / peak

        # Skip very short clips
        if len(audio) < sr * 0.5:
            return DetectionResult(
                synthetic_probability=0.5,
                confidence=0.1,
                heuristic_score=0.5,
                embedding_score=0.5,
                triggered_rules=[],
                explanation="Clip too short for reliable analysis.",
                chunk_index=chunk_index,
            )

        # ── Stage 1: Feature extraction ────────────────────────────────────
        feats = extract_features(audio, sr, store_arrays=True)

        # ── Stage 2: Heuristic scoring ─────────────────────────────────────
        h_score, triggered_rules = self._heuristic_score(feats)

        # ── Stage 3: Embedding scoring (if model loaded) ───────────────────
        e_score = self._embedding_score(audio, sr) if self.use_model else 0.5

        # ── Combine ────────────────────────────────────────────────────────
        if self.use_model and self._whisper_model is not None:
            # Weighted: heuristic 40%, embedding 60%
            combined = 0.40 * h_score + 0.60 * e_score
        else:
            combined = h_score

        # Confidence = how far from decision boundary (0.5)
        confidence = float(abs(combined - 0.5) * 2.0)  # [0, 1]

        # ── Explanation ────────────────────────────────────────────────────
        explanation = self._build_explanation(feats, triggered_rules, combined)

        elapsed_ms = (time.time() - t0) * 1000
        self._chunk_count += 1

        return DetectionResult(
            synthetic_probability=float(combined),
            confidence=confidence,
            heuristic_score=h_score,
            embedding_score=e_score,
            triggered_rules=triggered_rules,
            explanation=explanation,
            features=feats,
            processing_time_ms=elapsed_ms,
            chunk_index=chunk_index,
        )

    def detect_stream(self, audio: np.ndarray, sr: int, window_sec: float = 4.0,
                      hop_sec: float = 2.0):
        """
        Process long audio in sliding windows.
        
        Yields DetectionResult for each window.
        """
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)
        total = len(audio)

        idx = 0
        chunk_n = 0
        while idx + window_samples <= total:
            chunk = audio[idx: idx + window_samples]
            yield self.detect(chunk, sr, chunk_index=chunk_n)
            idx += hop_samples
            chunk_n += 1

        # Tail (if remaining > 1s)
        if total - idx > sr:
            chunk = audio[idx:]
            yield self.detect(chunk, sr, chunk_index=chunk_n)

    def _build_explanation(
        self, feats: AudioFeatures, triggered: list, score: float
    ) -> str:
        """Generate human-readable forensic explanation."""
        if score < 0.35:
            base = "Audio characteristics consistent with natural human speech."
            if not triggered:
                return base
            return base + " Minor spectral anomalies detected but below threshold."

        if score < 0.5:
            return ("Borderline audio. Some spectral properties slightly atypical "
                    "but insufficient to classify as synthetic.")

        # Build detailed explanation from triggered rules
        parts = []
        rule_msgs = {
            "spectral_flatness":  "over-smoothed frequency bands (spectral flatness={:.4f}) "
                                  "typical of neural TTS vocoders",
            "pitch_regularity":   "unnaturally stable pitch (regularity={:.2f}) — "
                                  "human speech has higher f0 variance",
            "hnr":                "abnormally clean harmonic structure (HNR={:.1f} dB) — "
                                  "real speech has more noise",
            "phase_randomness":   "structured phase patterns (randomness={:.3f}) "
                                  "consistent with neural vocoder output",
            "mel_smoothness":     "insufficient mel-spectrogram temporal variation "
                                  "(std={:.2f} dB) — TTS lacks micro-variation",
            "silence_regularity": "overly regular silence/pause intervals "
                                  "(regularity={:.2f}) — robotic breath patterns",
            "spectral_flux":      "low spectral flux ({:.5f}) — "
                                  "transitions too smooth for natural speech",
            "dynamic_range":      "compressed dynamic range ({:.1f} dB) — "
                                  "TTS output lacks natural loudness variation",
        }

        for name, value, threshold, direction, rule_score in triggered:
            if name in rule_msgs:
                try:
                    msg = rule_msgs[name].format(value)
                    parts.append(msg)
                except Exception:
                    parts.append(f"{name} anomaly detected")

        if not parts:
            return f"High synthetic probability ({score:.1%}) — multiple weak signals combined."

        severity = "HIGH" if score > 0.75 else "MODERATE"
        summary = f"[{severity}] Detected " + "; ".join(parts[:3]) + "."
        if len(parts) > 3:
            summary += f" (+{len(parts)-3} additional anomalies)"
        return summary
