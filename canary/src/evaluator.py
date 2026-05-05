"""
evaluator.py — Detection performance metrics.

Implements:
  - EER (Equal Error Rate): threshold where FAR = FRR
  - t-DCF (tandem Detection Cost Function): ASVspoof standard metric
  - ROC curve
  - Confusion matrix
  - Robustness benchmarking across degradation conditions
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class EvaluationResult:
    eer: float                         # Equal Error Rate
    eer_threshold: float               # Decision threshold at EER
    tdcf: float                        # tandem Detection Cost Function
    auc: float                         # Area Under ROC Curve
    accuracy: float                    # at EER threshold
    far_at_threshold: float            # False Accept Rate
    frr_at_threshold: float            # False Reject Rate
    avg_latency_ms: float              # inference time
    n_samples: int


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate.
    
    Args:
        scores: synthetic probability scores (higher = more synthetic)
        labels: 1 = synthetic (positive), 0 = human (negative)
        
    Returns:
        (eer, threshold) where FAR = FRR
    """
    thresholds = np.linspace(0, 1, 1000)
    far_arr = []
    frr_arr = []

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        # FAR: human classified as synthetic
        far = np.sum((pred == 1) & (labels == 0)) / (n_neg + 1e-8)
        # FRR: synthetic classified as human
        frr = np.sum((pred == 0) & (labels == 1)) / (n_pos + 1e-8)
        far_arr.append(far)
        frr_arr.append(frr)

    far_arr = np.array(far_arr)
    frr_arr = np.array(frr_arr)

    # Find where FAR ≈ FRR
    diff = np.abs(far_arr - frr_arr)
    idx = np.argmin(diff)
    eer = (far_arr[idx] + frr_arr[idx]) / 2
    threshold = thresholds[idx]

    return float(eer), float(threshold)


def compute_tdcf(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    # ASVspoof 2019 cost parameters
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_target: float = 0.05,
) -> float:
    """
    Compute tandem Detection Cost Function (t-DCF).
    
    Standard metric from ASVspoof challenge.
    Penalizes false accepts more heavily (c_fa=10).
    
    t-DCF = C_miss * P_target * FRR + C_fa * (1-P_target) * FAR
    """
    pred = (scores >= threshold).astype(int)
    n_pos = np.sum(labels == 1) + 1e-8
    n_neg = np.sum(labels == 0) + 1e-8

    far = np.sum((pred == 1) & (labels == 0)) / n_neg
    frr = np.sum((pred == 0) & (labels == 1)) / n_pos

    tdcf = c_miss * p_target * frr + c_fa * (1 - p_target) * far
    # Normalize by min possible
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(tdcf / (c_def + 1e-8))


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under ROC Curve via trapezoidal rule."""
    thresholds = np.linspace(0, 1, 500)
    tpr_arr = []
    fpr_arr = []

    n_pos = np.sum(labels == 1) + 1e-8
    n_neg = np.sum(labels == 0) + 1e-8

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        tpr_arr.append(tp / n_pos)
        fpr_arr.append(fp / n_neg)

    tpr_arr = np.array(tpr_arr[::-1])
    fpr_arr = np.array(fpr_arr[::-1])

    auc = float(np.trapezoid(tpr_arr, fpr_arr) if hasattr(np, "trapezoid") else np.trapz(tpr_arr, fpr_arr))
    return float(np.clip(auc, 0, 1))


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    latencies_ms: Optional[List[float]] = None,
) -> EvaluationResult:
    """Full evaluation pipeline."""
    eer, eer_thr = compute_eer(scores, labels)
    auc = compute_auc(scores, labels)
    tdcf = compute_tdcf(scores, labels, eer_thr)

    # Accuracy at EER threshold
    pred = (scores >= eer_thr).astype(int)
    accuracy = float(np.mean(pred == labels))

    n_pos = np.sum(labels == 1) + 1e-8
    n_neg = np.sum(labels == 0) + 1e-8
    far = float(np.sum((pred == 1) & (labels == 0)) / n_neg)
    frr = float(np.sum((pred == 0) & (labels == 1)) / n_pos)

    avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    return EvaluationResult(
        eer=eer,
        eer_threshold=eer_thr,
        tdcf=tdcf,
        auc=auc,
        accuracy=accuracy,
        far_at_threshold=far,
        frr_at_threshold=frr,
        avg_latency_ms=avg_lat,
        n_samples=len(labels),
    )


def generate_synthetic_benchmark(n_human: int = 100, n_synthetic: int = 100,
                                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic benchmark scores for demo purposes.
    Models realistic score distributions:
      - Human: Beta(2, 5) centered around 0.25
      - Synthetic: Beta(5, 2) centered around 0.75
    """
    rng = np.random.RandomState(seed)
    human_scores = rng.beta(2, 5, n_human)
    synth_scores = rng.beta(5, 2, n_synthetic)
    scores = np.concatenate([human_scores, synth_scores])
    labels = np.concatenate([np.zeros(n_human), np.ones(n_synthetic)])
    return scores, labels


def roc_curve_data(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (FPR, TPR) arrays for ROC plot."""
    thresholds = np.linspace(0, 1, 300)
    fprs, tprs = [], []
    n_pos = np.sum(labels == 1) + 1e-8
    n_neg = np.sum(labels == 0) + 1e-8
    for thr in thresholds[::-1]:
        pred = (scores >= thr).astype(int)
        tprs.append(np.sum((pred == 1) & (labels == 1)) / n_pos)
        fprs.append(np.sum((pred == 1) & (labels == 0)) / n_neg)
    return np.array(fprs), np.array(tprs)


def benchmark_under_degradation(
    detector,
    audio: np.ndarray,
    sr: int,
    label: int,          # 1 = synthetic, 0 = human
) -> Dict[str, Dict]:
    """
    Run detector on audio under multiple degradation conditions.
    Returns dict of {condition_name: {score, latency_ms}}.
    """
    from .robustness import apply_degradation, benchmark_degradation_configs

    results = {}
    configs = benchmark_degradation_configs()

    for config_info in configs:
        name = config_info["name"]
        cfg = config_info["config"]
        try:
            degraded = apply_degradation(audio, sr, cfg)
            t0 = time.time()
            result = detector.detect(degraded, sr)
            latency = (time.time() - t0) * 1000
            results[name] = {
                "score": result.synthetic_probability,
                "latency_ms": latency,
                "correct": (result.synthetic_probability > 0.5) == bool(label),
            }
        except Exception as e:
            results[name] = {"score": 0.5, "latency_ms": 0, "correct": False, "error": str(e)}

    return results
