# 🐦 Canary — Real-Time Deepfake Audio Forensics

> A production-ready prototype for detecting AI-generated (synthetic) speech in real-time VoIP calls with explainability, TTS fingerprinting, and adversarial robustness testing.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Detection Logic](#detection-logic)
4. [Models Used](#models-used)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Module Reference](#module-reference)
8. [Evaluation Metrics](#evaluation-metrics)
9. [TTS Fingerprinting](#tts-fingerprinting)
10. [Adversarial Robustness](#adversarial-robustness)
11. [Limitations & Failure Cases](#limitations--failure-cases)
12. [Deployment Strategy](#deployment-strategy)
13. [Roadmap](#roadmap)

---

## Overview

Canary is a **real-time deepfake audio detection system** designed to flag synthetic (AI-generated) speech during live VoIP calls (Zoom, WebRTC, phone gateways). It provides:

- **Real-time detection** with <500ms latency target
- **Interpretable explanations** of why audio is flagged
- **TTS fingerprinting** to identify likely source model
- **Adversarial robustness** against codec compression, noise, and packet loss
- **Standard evaluation** with EER, t-DCF, AUC metrics

**Threat model**: An attacker uses a TTS system (SpeechT5, XTTS-v2, OpenVoice, etc.) to impersonate a human on a call. Canary detects the synthetic origin in real-time.

---

## Architecture

```
                    ┌────────────────────────────────────────┐
                    │         AUDIO INGESTION LAYER          │
                    │  WebRTC / VoIP / SIP / File Upload     │
                    └──────────────────┬─────────────────────┘
                                       │ raw PCM (8–48kHz)
                    ┌──────────────────▼─────────────────────┐
                    │         PREPROCESSING PIPELINE         │
                    │  · Mono conversion                     │
                    │  · Normalization                       │
                    │  · Optional degradation simulation     │
                    │    (codec / noise / reverb)            │
                    └──────────────────┬─────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │              SLIDING WINDOW ENGINE              │
              │  window=2–10s, hop=0.5–3s                       │
              └───────────┬─────────────────────┬───────────────┘
                          │                     │
         ┌────────────────▼──┐         ┌────────▼────────────────┐
         │  STAGE 1          │         │  STAGE 2                │
         │  Spectral         │         │  Whisper Encoder        │
         │  Heuristics       │         │  (openai/whisper-small) │
         │                   │         │                         │
         │  · Flatness       │         │  · Extract encoder      │
         │  · Pitch reg.     │         │    embeddings           │
         │  · HNR            │         │  · Mahalanobis dist.    │
         │  · Phase rand.    │         │    from human ref.      │
         │  · Mel smooth.    │         │  · Anomaly score        │
         │  · Silence reg.   │         │                         │
         │  · Spec. flux     │         └────────┬────────────────┘
         │  · Dyn. range     │                  │
         └────────┬──────────┘                  │
                  │  40%                        │  60%
                  └───────────────┬─────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │      FUSION ENGINE         │
                    │  weighted combination      │
                    │  → P(synthetic) ∈ [0,1]   │
                    └─────────────┬──────────────┘
                                  │
              ┌───────────────────┼──────────────────┐
              │                   │                  │
  ┌───────────▼──────┐  ┌─────────▼──────┐  ┌───────▼──────────┐
  │  EXPLAINER       │  │  FINGERPRINT   │  │  ALERT SYSTEM    │
  │                  │  │                │  │                  │
  │  · Spectrogram   │  │  · TTS model   │  │  · UI verdict    │
  │  · Anomaly       │  │    ID          │  │  · API alert     │
  │    overlay       │  │  · Artifact    │  │  · Timestamp     │
  │  · Pitch plot    │  │    description │  │  · Confidence    │
  │  · Feature bars  │  └────────────────┘  └──────────────────┘
  └──────────────────┘
```

### Component Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — 5 tabs: Analyze, Stream, Fingerprint, Robustness, Metrics |
| `src/detector.py` | Core detection pipeline (heuristics + Whisper encoder) |
| `src/features.py` | Spectral/prosodic feature extraction (librosa-based) |
| `src/explainer.py` | Plotly visualization — spectrogram, pitch, anomaly dashboard |
| `src/fingerprint.py` | TTS system identification via spectral profiles |
| `src/robustness.py` | Codec/noise/reverb degradation simulation |
| `src/evaluator.py` | EER, t-DCF, AUC, ROC computation |

---

## Detection Logic

### Stage 1: Spectral Heuristics

Eight rules derived from research on TTS artifact signatures:

| Feature | Threshold | Direction | Why it works |
|---------|-----------|-----------|--------------|
| Spectral Flatness | > 0.07 | High = suspicious | TTS vocoders over-smooth frequency bands; natural speech has more spectral variation |
| Pitch Regularity (f0 CV) | > 0.72 | High = suspicious | Human pitch has natural micro-variation; TTS is unnaturally steady |
| Harmonic-to-Noise Ratio | > 18 dB | High = suspicious | TTS is too "clean"; real speech has noise floor |
| Phase Randomness | < 0.85 | Low = suspicious | Neural vocoders introduce structured phase patterns; natural speech phase is random |
| Mel Smoothness | < 3.5 dB | Low = suspicious | TTS lacks the micro-variation of natural speech in mel bands |
| Silence Regularity | > 0.68 | High = suspicious | TTS breath/pause patterns are too regular |
| Spectral Flux | < 0.005 | Low = suspicious | TTS spectral transitions are too smooth |
| Dynamic Range | < 15 dB | Low = suspicious | TTS output has compressed loudness variation |

**Scoring**: Each rule votes with a weight. Weighted average → heuristic score ∈ [0,1].

### Stage 2: Whisper Encoder Embeddings

Uses `openai/whisper-small` encoder as a feature extractor:

1. Resample audio to 16kHz (Whisper requirement)
2. Compute log-mel spectrogram (Whisper's input)
3. Pass through Whisper encoder → get `last_hidden_state`
4. Mean-pool over time → fixed-length embedding
5. Compute **Mahalanobis distance** from a reference distribution of human speech embeddings
6. Sigmoid-normalize → anomaly score ∈ [0,1]

**Why Whisper?** Whisper was trained on vast human speech data. Its encoder representations capture the "fingerprint" of natural speech. Synthetic audio produces embeddings that deviate from this distribution.

**Calibration**: For best results, call `detector.calibrate(human_samples, sr)` with known human speech chunks before deployment. This builds the reference distribution.

### Fusion

```
P(synthetic) = 0.40 × heuristic_score + 0.60 × embedding_score
```

The embedding score gets higher weight when the Whisper model is loaded. Heuristic-only mode uses `P(synthetic) = heuristic_score`.

---

## Models Used

| Model | Source | Use | License |
|-------|--------|-----|---------|
| `openai/whisper-small` | [HuggingFace](https://huggingface.co/openai/whisper-small) | Encoder embeddings for anomaly scoring | MIT |

All other components use **classical signal processing** (librosa, scipy) — no additional model downloads required for heuristic-only mode.

### Production Model Recommendations

For production deployment, fine-tune or use these free anti-spoofing models:

| Model | Description | Source |
|-------|-------------|--------|
| `asvspoof/aasist` | State-of-art anti-spoofing | [GitHub](https://github.com/clovaai/aasist) |
| RawNet2 | Raw waveform classification | ASVspoof repo |
| `facebook/wav2vec2-base` | Fine-tune on ASVspoof 2019 LA | HuggingFace |
| `microsoft/wavlm-base` | WavLM fine-tuned features | HuggingFace |

**Training data**: [ASVspoof 2019 LA dataset](https://datashare.ed.ac.uk/handle/10283/3336) (free, requires registration)

---

## Installation

### Prerequisites

- Python 3.10+
- Git

### Steps

```bash
# Clone
git clone <repo> canary
cd canary

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

### First Run

On first use with `Use Whisper Encoder` enabled, Canary downloads `openai/whisper-small` (~500MB) and caches it. Subsequent runs are instant.

Disable "Use Whisper Encoder" in the sidebar for heuristic-only mode (no download, ~50ms latency).

---

## Usage

### Streamlit UI Tabs

#### [ ANALYZE ]
- Upload any audio file (WAV/MP3/FLAC/OGG/M4A)
- Configure degradation in sidebar (codec, noise, reverb)
- Click **RUN FORENSIC ANALYSIS**
- View: verdict, probability score, forensic explanation, spectral dashboard

#### [ STREAM SIM ]
- Upload audio for sliding-window streaming simulation
- Adjust window/hop in sidebar
- Click **START STREAM ANALYSIS**
- Watch real-time chunk-by-chunk probability chart update

#### [ FINGERPRINT ]
- Upload suspected synthetic audio
- Click **FINGERPRINT AUDIO**
- Get: likely TTS source (SpeechT5/XTTS-v2/OpenVoice/etc.), confidence, artifact list

#### [ ROBUSTNESS ]
- Upload audio, label it as human or synthetic
- Click **RUN ROBUSTNESS BENCHMARK**
- Tests 7 degradation conditions: Clean → Phone → VoIP → Noise → Reverb → Babble → Worst Case
- Download degraded audio preview

#### [ METRICS ]
- Synthetic benchmark: Beta-distributed score simulation
- EER, t-DCF, AUC, FAR/FRR at EER threshold
- ROC curve visualization
- Score distribution histogram

### Python API

```python
from src.detector import DeepfakeDetector
import librosa

# Load detector
detector = DeepfakeDetector(use_model=True)

# Single chunk
audio, sr = librosa.load("voice.wav", sr=None, mono=True)
result = detector.detect(audio, sr)

print(f"Synthetic probability: {result.synthetic_probability:.3f}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Explanation: {result.explanation}")
print(f"Latency: {result.processing_time_ms:.0f} ms")

# Streaming
for chunk_result in detector.detect_stream(audio, sr, window_sec=4.0, hop_sec=2.0):
    print(f"Chunk {chunk_result.chunk_index}: {chunk_result.synthetic_probability:.3f}")
    if chunk_result.synthetic_probability > 0.65:
        print("  ⚠ ALERT: Synthetic speech detected!")

# Calibrate on human speech (improves embedding score)
human_samples = [audio1, audio2, audio3]  # known human speech
detector.calibrate(human_samples, sr=16000)
```

```python
# Fingerprinting
from src.fingerprint import fingerprint_audio

fp = fingerprint_audio(audio, sr)
print(f"Source: {fp.likely_source} ({fp.confidence:.1%})")
for artifact in fp.artifacts:
    print(f"  · {artifact}")

# Robustness
from src.robustness import apply_degradation, DegradationConfig

config = DegradationConfig(
    codec="ulaw",           # μ-law telephone
    target_sr=8000,         # downsample to 8kHz
    noise_type="babble",    # office noise
    noise_snr_db=15.0,
    reverb_rt60=0.3,
    packet_loss_pct=5.0,
)
degraded_audio = apply_degradation(audio, sr, config)
result = detector.detect(degraded_audio, sr)
```

---

## Module Reference

### `src/features.py`

```python
extract_features(audio: np.ndarray, sr: int, store_arrays: bool = True) -> AudioFeatures
```

Extracts 20+ spectral, prosodic, and temporal features.

`AudioFeatures` fields:
- `spectral_flatness_mean/std` — Wiener entropy of spectrum
- `spectral_rolloff_mean` — frequency below which 85% of energy lies
- `spectral_bandwidth_mean` — spread of spectrum
- `spectral_contrast_mean` — [7] subband contrast values
- `spectral_flux` — frame-to-frame spectral change
- `mfcc_mean/std/delta_mean` — [20] MFCC statistics
- `mel_smoothness` — temporal variance across mel bands
- `f0_mean/std/regularity` — pitch statistics
- `voiced_fraction` — proportion of voiced frames
- `rms_mean/std` — energy statistics
- `dynamic_range` — max-min RMS in dB
- `hnr` — harmonic-to-noise ratio (dB)
- `phase_randomness` — mean absolute inter-frame phase difference
- `zcr_mean/std` — zero-crossing rate
- `silence_ratio/regularity` — silence pattern statistics

### `src/detector.py`

```python
class DeepfakeDetector:
    def __init__(use_model: bool, device: str)
    def calibrate(human_samples: list, sr: int)
    def detect(audio: np.ndarray, sr: int, chunk_index: int) -> DetectionResult
    def detect_stream(audio, sr, window_sec, hop_sec) -> Iterator[DetectionResult]
```

`DetectionResult`:
- `synthetic_probability` — [0,1], higher = more likely synthetic
- `confidence` — how certain the model is (distance from 0.5)
- `heuristic_score` — Stage 1 score only
- `embedding_score` — Stage 2 score only
- `triggered_rules` — list of (name, value, threshold, direction, score)
- `explanation` — human-readable forensic explanation
- `processing_time_ms` — inference latency

### `src/fingerprint.py`

```python
fingerprint_audio(audio: np.ndarray, sr: int) -> FingerprintResult
```

`FingerprintResult`:
- `likely_source` — "SpeechT5" / "XTTS-v2" / "OpenVoice" / "WhisperTTS" / "AudioLDM2" / "Human Speech" / "Unknown TTS"
- `confidence` — [0,1]
- `scores` — dict {model_name: similarity_score}
- `artifacts` — list of detected artifact descriptions
- `is_synthetic` — boolean

### `src/robustness.py`

```python
apply_degradation(audio, sr, config: DegradationConfig) -> np.ndarray

DegradationConfig:
  codec: str              # 'ulaw' | 'opus_sim' | 'aac_sim'
  target_sr: int          # downsample target
  noise_type: str         # 'white' | 'pink' | 'babble'
  noise_snr_db: float
  reverb_rt60: float      # seconds
  packet_loss_pct: float  # 0-100
  bandwidth_limit_hz: float
```

### `src/evaluator.py`

```python
compute_eer(scores, labels) -> (eer, threshold)
compute_tdcf(scores, labels, threshold) -> float
compute_auc(scores, labels) -> float
evaluate(scores, labels, latencies_ms) -> EvaluationResult
roc_curve_data(scores, labels) -> (fpr, tpr)
benchmark_under_degradation(detector, audio, sr, label) -> dict
```

---

## Evaluation Metrics

### EER (Equal Error Rate)

The threshold where **False Accept Rate = False Reject Rate**.

- FAR (False Accept Rate): Human speech classified as synthetic
- FRR (False Reject Rate): Synthetic classified as human
- **Lower EER = better**. 0% = perfect, 50% = random

ASVspoof 2019 LA baselines:
- GMM baseline: ~30% EER
- LFCC-LCNN: ~5.06% EER  
- AASIST: ~0.83% EER (SOTA)

### t-DCF (tandem Detection Cost Function)

Standard ASVspoof metric. Weighted combination of FAR and FRR, with false accepts penalized 10× more (security application).

```
t-DCF = C_miss × P_target × FRR + C_fa × (1-P_target) × FAR

Parameters (ASVspoof 2019):
  C_miss = 1, C_fa = 10, P_target = 0.05
```

Normalized: 0 = perfect, 1 = baseline (always predict one class).

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| EER | < 10% | Baseline acceptable for prototype |
| t-DCF | < 0.5 | Industry threshold |
| AUC | > 0.90 | Production requirement |
| Latency | < 500ms | Consumer hardware (10s chunk) |
| FAR | < 5% | Acceptable false alarm rate |
| FRR | < 15% | Max allowed miss rate |

---

## TTS Fingerprinting

Rule-based classifier matching spectral features to known TTS profiles.

### Known Profiles

**SpeechT5** (`microsoft/speecht5_tts`):
- Characteristic spectral peaks in 3–5kHz (formant emphasis)
- Very high spectral flatness (over-smoothed vocoder)
- Ultra-regular pitch (Duration model over-regularizes)

**XTTS-v2** (`coqui/XTTS-v2`):
- High-frequency artifacts > 6kHz (VITS/HifiGAN vocoder)
- Low phase randomness (structured vocoder phase)
- Aggressive voice conversion artifacts

**OpenVoice**:
- Mid-band smoothing in 1–3kHz (voice encoder bottleneck)
- Characteristic formant over-smoothing
- Speaker embedding artifacts in envelope

**WhisperTTS** / **Bark**:
- Low-frequency emphasis (codec2 bottleneck)
- Highly compressed dynamic range
- Regular silence patterns (tokenized prosody)

**AudioLDM2** (`cvssp/audioldm2`):
- Extremely structured phase (diffusion vocoder)
- Very low spectral flux (diffusion smooth generation)
- Abnormally high HNR (clean diffusion output)

### Production Fingerprinting

For production, train a classifier:

```python
# 1. Generate labeled data
from transformers import pipeline
tts_speecht5 = pipeline("text-to-speech", model="microsoft/speecht5_tts")
tts_xtts = ...  # XTTS-v2

# 2. Extract features
from src.features import extract_features, feature_vector
X = []
y = []
for audio, label in labeled_samples:
    feats = extract_features(audio, sr, store_arrays=False)
    X.append(feature_vector(feats))
    y.append(label)

# 3. Train classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)
```

---

## Adversarial Robustness

### Degradation Conditions Tested

| Condition | Description | Key Effect |
|-----------|-------------|------------|
| Clean | No degradation | Baseline |
| Phone (μ-law) | ITU G.711, 8kHz, bandwidth 300–3400Hz | Removes high-freq artifacts |
| VoIP (Opus) | 24kbps, 3% packet loss | Quantization + temporal artifacts |
| Noisy (SNR 10dB) | White noise at 10dB SNR | Masks spectral flatness |
| Reverb | RT60=0.5s, pink noise | Blurs harmonic structure |
| Babble | 8 synthetic voices, SNR 15dB | Masks pitch regularity |
| Worst Case | All combined | Maximum real-world stress |

### Robustness Notes

- **Heuristic features most robust to**: noise addition (spectral flatness still detectable), reverb (pitch still too regular)
- **Heuristic features least robust to**: μ-law quantization (destroys high-freq content), Opus (quantization changes flatness)
- **Whisper embeddings robust to**: moderate noise, mild reverb
- **Whisper embeddings less robust to**: extreme downsampling, heavy packet loss

### Expected Performance Degradation

Under telephony conditions (8kHz μ-law):
- Spectral flatness: ~30% less discriminative (high-freq info lost)
- Pitch regularity: ~5% degraded (robust feature)
- Whisper embedding: ~15% less discriminative
- Overall: expect EER to increase from ~10% to ~20–25% under phone conditions

---

## Limitations & Failure Cases

### Known Limitations

**1. No training data in this prototype**
The spectral heuristics are calibrated from research literature, not trained on actual ASVspoof data. Production deployment MUST fine-tune on [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336).

**2. Whisper calibration required**
Without calling `detector.calibrate(human_samples)`, the embedding score uses a generic fallback. Per-deployment calibration significantly improves accuracy.

**3. Short clips**
Clips < 1 second are unreliable. Minimum 2 seconds recommended; 4+ seconds optimal.

**4. Adversarial attacks**
A sophisticated attacker who knows the detection features can evade:
- Add noise to increase spectral non-flatness
- Apply slight pitch randomization post-synthesis
- Use adversarial perturbations targeting Whisper embeddings

**5. Novel TTS systems**
TTS systems released after calibration/training may not be detected. Canary has no generalization guarantee for unseen architectures without retraining.

**6. Partial deepfakes**
The current system analyzes full windows. Partial deepfakes (only some words synthesized) may achieve `P(synthetic) < 0.65` if the window mostly contains real speech.

**7. Non-speech audio**
Music, noise-only, or environmental audio may trigger false positives. Apply Voice Activity Detection (VAD) before Canary.

**8. Emotional/accented speech**
Some accents and emotional states (shouting, whispering) have unusual spectral properties that may increase false positive rates.

### Failure Cases

| Scenario | Expected Behavior | Mitigation |
|----------|------------------|------------|
| High-quality TTS + noise injection | FRR increases (missed detection) | Lower threshold, ensemble model |
| Children's voices | Higher false positive rate | Calibrate on diverse human speech |
| Codec-processed real speech | FRR increases (classified as synthetic) | Use post-codec calibration data |
| Very long pauses in real speech | Silence regularity triggers | Tune silence_regularity threshold |
| Singing | High spectral flatness triggers | VAD to exclude music |
| Whispered speech | Low HNR may not trigger | HNR threshold adjustment |

---

## Deployment Strategy

### Option A: Streamlit Standalone (Current)

```bash
streamlit run app.py --server.port 8501
```

Best for: demos, security analysts, manual forensic review.

### Option B: REST API Microservice

```python
# FastAPI wrapper (extend this)
from fastapi import FastAPI, File, UploadFile
from src.detector import DeepfakeDetector
import librosa, io

app = FastAPI()
detector = DeepfakeDetector(use_model=True)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    result = detector.detect(audio, sr)
    return {
        "synthetic_probability": result.synthetic_probability,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "latency_ms": result.processing_time_ms,
    }
```

### Option C: Node.js + Python Microservice

```
Browser/VoIP ─→ Node.js WebSocket ─→ Python ML Service ─→ Alert
                 (audio buffer)      (FastAPI + Canary)    (WebSocket)
```

**Node.js** handles:
- WebRTC audio capture via `getUserMedia`
- Chunking audio into 4s windows
- WebSocket communication to Python service
- UI updates from detection results

**Python service** handles:
- ML inference (Whisper encoder)
- Feature extraction
- Result formatting

### Option D: Browser Extension

```javascript
// Capture audio from call tab
chrome.tabCapture.capture({ audio: true }, (stream) => {
    // Process via AudioWorklet (real-time)
    // Send chunks to Python backend
    // Show overlay badge: 🟢 REAL / 🔴 FAKE
});
```

### Latency Budget (10s chunk)

| Component | Target |
|-----------|--------|
| Audio preprocessing | < 5ms |
| Feature extraction | < 20ms |
| Whisper encoder | < 300ms (GPU), < 1500ms (CPU) |
| Heuristic scoring | < 5ms |
| Fusion + explanation | < 5ms |
| **Total (GPU)** | **< 350ms** ✓ |
| **Total (CPU)** | **< 1600ms** ✗ need optimization |

**CPU optimization strategies**:
- Quantize Whisper to INT8 (`bitsandbytes`)
- Use `whisper-tiny` instead of `whisper-small` (4× faster)
- Cache encoder computation for overlapping windows
- Use ONNX runtime for inference

---

## Roadmap

### v1.1 — Production Model
- [ ] Fine-tune Wav2Vec2 on ASVspoof 2019 LA dataset
- [ ] Integrate AASIST model (SOTA anti-spoofing)
- [ ] Compare RawNet2 vs AASIST vs Wav2Vec2 in benchmark table

### v1.2 — Real-Time Streaming
- [ ] WebRTC audio capture integration
- [ ] FastAPI REST endpoint
- [ ] WebSocket streaming results
- [ ] Zoom/Meet browser extension

### v1.3 — Partial Deepfake Detection
- [ ] Frame-level scoring within windows
- [ ] Segment boundary detection
- [ ] "Which words were fake" overlay

### v1.4 — Adaptive Learning
- [ ] Online calibration from confirmed cases
- [ ] Incremental learning for new TTS systems
- [ ] Confidence-weighted model updates

### v2.0 — Enterprise
- [ ] SIP/RTP gateway integration
- [ ] Multi-speaker diarization + per-speaker detection
- [ ] Fraud detection API with case management
- [ ] Audit log with tamper-evident storage

---

## References

1. **ASVspoof 2019** — Wang et al., "ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2020
2. **AASIST** — Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022
3. **RawNet2** — Tak et al., "End-to-End anti-spoofing with RawNet2," ICASSP 2021
4. **Wav2Vec 2.0** — Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS 2020
5. **Whisper** — Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," ICML 2023
6. **t-DCF** — Kinnunen et al., "t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification," Odyssey 2018

---

*Canary v1.0 — Built for the Deepfake Audio Forensics challenge.*  
*Model: openai/whisper-small (MIT license). All other components: classical signal processing.*
