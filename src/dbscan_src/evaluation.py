"""Shared fraud-detection metrics. Larger scores must mean more anomalous."""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def cohort_id(row_indices: np.ndarray, feature_file_sha256: str) -> str:
    """Identify the exact source file and sampled CSV row positions."""
    digest = hashlib.sha256(feature_file_sha256.encode("ascii"))
    digest.update(np.sort(row_indices).astype("<i8").tobytes())
    return digest.hexdigest()[:16]


def detection_metrics(y_true, y_pred, scores=None, row_indices=None, ks=(100, 500, 1000)):
    """Compute binary metrics plus optional ranking metrics; never fit a model.

    PR-AUC is reported as average precision (AP), not trapezoidal integration.
    For Recall@K, tied scores are ordered by ascending source CSV row position.
    If K exceeds the cohort size, all rows are inspected and effective K is saved.
    Metrics requiring both classes are left missing for a single-class cohort.
    """
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.ndim != 1 or truth.size == 0 or prediction.shape != truth.shape:
        raise ValueError("y_true and y_pred must be nonempty aligned 1-D arrays")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Labels must be binary: normal=0, fraud/anomaly=1")
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(truth)) == 2
    result = {
        "n_samples": int(truth.size),
        "n_fraud": int(truth.sum()),
        "n_anomaly_candidates": int(prediction.sum()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, prediction)) if both_classes else np.nan,
        "pr_auc": np.nan,
        "pr_auc_definition": "average_precision",
        "roc_auc": np.nan,
    }
    for k in ks:
        if k < 1:
            raise ValueError("Recall@K requires positive K")
        result[f"recall_at_{k}"] = np.nan
        result[f"effective_k_{k}"] = min(k, truth.size)
    if scores is None:
        return result
    scores = np.asarray(scores, dtype=float)
    if scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and aligned with y_true")
    ids = np.arange(truth.size) if row_indices is None else np.asarray(row_indices)
    if ids.shape != truth.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Source row indices must be aligned and unique")
    if both_classes:
        result["pr_auc"] = float(average_precision_score(truth, scores))
        result["roc_auc"] = float(roc_auc_score(truth, scores))
    if truth.sum():
        order = np.lexsort((ids, -scores))
        for k in ks:
            result[f"recall_at_{k}"] = float(truth[order[:k]].sum() / truth.sum())
    return result
