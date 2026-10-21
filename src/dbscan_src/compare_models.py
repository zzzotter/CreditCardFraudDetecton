"""Build the common comparison table without mixing evaluation cohorts.

Optional B input: --supervised logistic_regression=path.csv random_forest=path.csv
Each file must contain row_index, cohort_id, data_split, y_true, y_score,
threshold. Thresholds must have been fixed independently of this evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .evaluation import detection_metrics
except ImportError:
    from evaluation import detection_metrics


def supervised_metrics(path: Path, reference: pd.DataFrame, method: str):
    frame = pd.read_csv(path)
    required = {"row_index", "cohort_id", "data_split", "y_true", "y_score", "threshold"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path.name} needs columns {sorted(required)}; cannot assume matching samples")
    if frame["row_index"].duplicated().any() or len(frame) != len(reference):
        raise ValueError("Supervised predictions must contain exactly the same unique sampled rows")
    frame = frame.sort_values("row_index").reset_index(drop=True)
    reference = reference.sort_values("row_index").reset_index(drop=True)
    for column in ["row_index", "cohort_id", "data_split", "y_true"]:
        if not np.array_equal(frame[column].to_numpy(), reference[column].to_numpy()):
            raise ValueError(f"Supervised predictions mismatch {column}; do not rank different cohorts together")
    scores = frame["y_score"].to_numpy(dtype=float)
    thresholds = frame["threshold"].to_numpy(dtype=float)
    if not np.isfinite(thresholds).all() or frame["threshold"].nunique() != 1:
        raise ValueError("Use one finite, predeclared threshold per model")
    return {
        "method": method, "method_type": "supervised", "status": "completed",
        "cohort_id": reference["cohort_id"].iloc[0], "data_split": "train",
        "evaluation_scope": "in_sample_clustering", "score_definition": "supervised_fraud_score",
        "threshold_rule": f"y_score >= {thresholds[0]:g}",
        **detection_metrics(frame["y_true"].to_numpy(), (scores >= thresholds).astype(int), scores, frame["row_index"].to_numpy()),
    }


def build_comparison(root: Path, supervised=None):
    tables = root / "results" / "tables"
    metrics_path = tables / "unsupervised_metrics.csv"
    metrics = pd.read_csv(metrics_path).to_dict("records") if metrics_path.is_file() and metrics_path.stat().st_size > 5 else []
    supplied = {}
    if supervised:
        reference_path = tables / "dbscan_cluster_results.csv"
        if not reference_path.is_file():
            reference_path = tables / "optics_cluster_results.csv"
        reference = pd.read_csv(reference_path)
        for spec in supervised:
            name, filename = spec.split("=", 1)
            if name not in {"logistic_regression", "random_forest"} or name in supplied:
                raise ValueError("Use logistic_regression=... and/or random_forest=..., once each")
            supplied[name] = supervised_metrics(Path(filename), reference, name)
    rows = []
    for method in ["logistic_regression", "random_forest", "DBSCAN", "OPTICS", "DENCLUE"]:
        found = supplied.get(method) or next((m for m in metrics if m["method"] == method), None)
        rows.append(found or {"method": method, "method_type": "supervised" if method in {"logistic_regression", "random_forest"} else "unsupervised",
                              "status": "feasibility_only" if method == "DENCLUE" else "pending",
                              "notes": "待完成正式实验" if method == "DENCLUE" else "待提供同 cohort 的预测结果或完成实验"})
    frame = pd.DataFrame(rows)
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "model_comparison.csv", index=False, encoding="utf-8-sig")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--supervised", nargs="*", default=[])
    options = parser.parse_args()
    try:
        result = build_comparison(options.project_root.resolve(), options.supervised)
        print(result[["method", "status"]].to_string(index=False))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Comparison stopped: {exc}\n")
