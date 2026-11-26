"""Run the SMOTE and Borderline-SMOTE baselines on the project training set."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Match the SMOTE(random_state=42) baseline in the upstream smote_experiment.py.
SAMPLING_STRATEGY = "auto"
K_NEIGHBORS = 5
RANDOM_STATE = 42
BORDERLINE_KIND = "borderline-1"
SAMPLERS = {
    "smote": SMOTE,
    "borderline_smote": partial(BorderlineSMOTE, kind=BORDERLINE_KIND),
}


def resample_training_data(
    x_train: pd.DataFrame, y_train: pd.Series, method: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the training features and Class labels after oversampling."""
    if method not in SAMPLERS:
        raise ValueError(f"Unknown sampling method: {method}")
    _validate_training_data(x_train, y_train)

    counts = y_train.value_counts()
    normal = counts.get(0, 0)
    fraud = counts.get(1, 0)
    if not 0 < fraud <= normal:
        raise ValueError("Expected both classes with Class=1 as the minority class.")

    sampler = SAMPLERS[method](
        sampling_strategy=SAMPLING_STRATEGY,
        k_neighbors=K_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    # Use floating point so interpolated values remain fractional for integer columns.
    return sampler.fit_resample(x_train.astype(float, copy=False), y_train)


def _validate_training_data(x_train: pd.DataFrame, y_train: pd.Series) -> None:
    if any(str(column).strip().lower() == "class" for column in x_train.columns):
        raise ValueError("Class must only appear in the labels.")
    if len(x_train) != len(y_train) or not y_train.isin([0, 1]).all():
        raise ValueError("Class labels must be 0 or 1 and align with the feature rows.")


def _read_training_csv(path: Path) -> pd.DataFrame:
    with path.open("rb") as source:
        if source.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path.name} is a Git LFS pointer. Run git lfs pull first.")
    return pd.read_csv(path)


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Read the original preprocessed feature and label files for training."""
    train_dir = PROJECT_ROOT / "data" / "processed"
    x_train = _read_training_csv(train_dir / "X_train.csv")
    labels = _read_training_csv(train_dir / "y_train.csv")
    if list(labels.columns) != ["Class"]:
        raise ValueError("y_train.csv must contain a single column named Class.")
    y_train = labels["Class"]
    _validate_training_data(x_train, y_train)
    return x_train, y_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results" / "sampling"
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Also export the resampled feature and label CSVs for each method.",
    )
    args = parser.parse_args()

    x_train, y_train = load_training_data()
    before = y_train.value_counts()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for method in SAMPLERS:
        print(f"Running {method}...", flush=True)
        x_resampled, y_resampled = resample_training_data(x_train, y_train, method)
        after = y_resampled.value_counts()
        summary = {
            "method": method,
            "sampling_strategy": SAMPLING_STRATEGY,
            "kind": BORDERLINE_KIND if method == "borderline_smote" else "",
            "k_neighbors": K_NEIGHBORS,
            "random_state": RANDOM_STATE,
            "normal_before": int(before[0]),
            "fraud_before": int(before[1]),
            "total_before": len(y_train),
            "normal_after": int(after[0]),
            "fraud_after": int(after[1]),
            "total_after": len(y_resampled),
            "synthetic_added": int(after[1] - before[1]),
            "fraud_rate_before": before[1] / len(y_train),
            "fraud_rate_after": after[1] / len(y_resampled),
        }
        if args.save_data:
            x_resampled.to_csv(args.output_dir / f"X_train_{method}.csv", index=False)
            y_resampled.to_csv(args.output_dir / f"y_train_{method}.csv", index=False)
        summary_path = args.output_dir / f"{method}_summary.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        print(
            f"Normal: {before[0]} -> {after[0]} | Fraud: {before[1]} -> {after[1]} | "
            f"Total: {summary['total_before']} -> {summary['total_after']} | "
            f"Synthetic added: {summary['synthetic_added']} | {summary_path}"
        )


if __name__ == "__main__":
    main()
