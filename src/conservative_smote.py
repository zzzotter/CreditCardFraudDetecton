"""Generate and filter Conservative SMOTE candidates using real training data."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

if __package__:
    from .sampling_methods import K_NEIGHBORS, PROJECT_ROOT, RANDOM_STATE, load_training_data
else:
    from sampling_methods import K_NEIGHBORS, PROJECT_ROOT, RANDOM_STATE, load_training_data


OUTLIER_NEIGHBORS = 5
OUTLIER_QUANTILE = 0.95
SYNTHESIS_MULTIPLIER = 2
MAX_CANDIDATES = 5000
GEOMETRY_NEIGHBORS = 5
GEOMETRY_QUANTILE = 0.75
PLAUSIBILITY_THRESHOLD = 0.75


def _fit_fraud_neighbors(
    x_minority: pd.DataFrame, requested_neighbors: int
) -> NearestNeighbors:
    if len(x_minority) < 2:
        raise ValueError("At least two real fraud rows are needed to compute neighbor distances.")
    return NearestNeighbors(
        n_neighbors=min(requested_neighbors, len(x_minority) - 1),
        algorithm="kd_tree",
        metric="euclidean",
    ).fit(x_minority)


def score_minority_outliers(
    x_minority: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float | bool]]:
    """Apply the paper's distance rule and mark the effective SMOTE base rows."""
    minority_count = len(x_minority)
    neighbors = _fit_fraud_neighbors(x_minority, OUTLIER_NEIGHBORS)
    # Calling kneighbors without a query matrix excludes each fitted row itself.
    distances = neighbors.kneighbors(return_distance=True)[0]
    average_distances = distances.mean(axis=1)
    threshold = float(np.quantile(average_distances, OUTLIER_QUANTILE))
    is_outlier = average_distances > threshold
    inlier_count = int((~is_outlier).sum())

    # SMOTE needs k_neighbors neighbors in addition to the starting row.
    fallback_used = inlier_count <= K_NEIGHBORS
    use_for_smote = ~is_outlier | fallback_used
    scores = pd.DataFrame({
        "train_row_index": x_minority.index,
        "mean_neighbor_distance": average_distances,
        "is_outlier": is_outlier,
        "use_for_smote": use_for_smote,
    })
    summary = {
        "outlier_neighbors": neighbors.n_neighbors,
        "outlier_quantile": OUTLIER_QUANTILE,
        "distance_threshold": threshold,
        "fraud_before": minority_count,
        "outliers_flagged": int(is_outlier.sum()),
        "fraud_after_cleaning": inlier_count,
        "fraud_in_smote_base": int(use_for_smote.sum()),
        "fallback_used": fallback_used,
    }
    return scores, summary


def generate_synthetic_candidates(
    x_majority: pd.DataFrame,
    x_smote_base: pd.DataFrame,
    original_fraud_count: int,
) -> pd.DataFrame:
    """Return only the new minority rows from bounded SMOTE interpolation."""
    if len(x_smote_base) <= K_NEIGHBORS:
        raise ValueError(f"SMOTE needs at least {K_NEIGHBORS + 1} fraud rows in its base.")

    # Equation (4) uses the original fraud count, before outlier removal.
    candidate_count = min(MAX_CANDIDATES, SYNTHESIS_MULTIPLIER * original_fraud_count)
    x_base = pd.concat([x_majority, x_smote_base], ignore_index=True).astype(float, copy=False)
    y_base = np.repeat([0, 1], [len(x_majority), len(x_smote_base)])
    sampler = SMOTE(
        sampling_strategy={1: len(x_smote_base) + candidate_count},
        k_neighbors=K_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    x_resampled, _ = sampler.fit_resample(x_base, y_base)
    return x_resampled.iloc[len(x_base):].reset_index(drop=True)


def score_geometric_coherence(
    x_minority: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Compare each candidate's mean distance with the real-to-real reference."""
    neighbors = _fit_fraud_neighbors(x_minority, GEOMETRY_NEIGHBORS)
    # Equations (7) and (8) use all real fraud rows, with self excluded from the reference.
    real_means = neighbors.kneighbors(return_distance=True)[0].mean(axis=1)
    threshold = float(np.quantile(real_means, GEOMETRY_QUANTILE))
    distances = np.empty(0)
    if not candidates.empty:
        distances = neighbors.kneighbors(candidates, return_distance=True)[0].mean(axis=1)
    passes_geometry = distances <= threshold
    scores = pd.DataFrame({
        "candidate_index": candidates.index,
        "mean_distance_to_real_fraud": distances,
        "passes_geometry": passes_geometry,
    })
    summary = {
        "geometry_neighbors": neighbors.n_neighbors,
        "geometry_quantile": GEOMETRY_QUANTILE,
        "geometry_distance_threshold": threshold,
        "geometry_reference_fraud_rows": len(x_minority),
        "candidates_after_geometry": int(passes_geometry.sum()),
    }
    return scores, summary


def score_candidate_plausibility(
    x_train: pd.DataFrame, y_train: pd.Series, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Score the geometry survivors with a classifier fitted on real rows only."""
    probabilities = np.empty(0)
    if not candidates.empty:
        auxiliary = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        auxiliary.fit(x_train, y_train)
        probabilities = auxiliary.predict_proba(candidates)[:, 1]
    return pd.DataFrame({
        "candidate_index": candidates.index,
        "fraud_probability": probabilities,
        "passes_plausibility": probabilities >= PLAUSIBILITY_THRESHOLD,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Also generate synthetic candidates.")
    parser.add_argument("--full", action="store_true", help="Generate candidates and apply both filters.")
    args = parser.parse_args()

    x_train, y_train = load_training_data()
    x_minority = x_train.loc[y_train.eq(1)]
    scores, summary = score_minority_outliers(x_minority)
    output_dir = PROJECT_ROOT / "results" / "conservative_smote"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "minority_outlier_scores.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "minority_cleaning_summary.csv", index=False)
    print(
        f"Real fraud rows: {summary['fraud_before']} | "
        f"Flagged outliers: {summary['outliers_flagged']} | "
        f"SMOTE base rows: {summary['fraud_in_smote_base']} | "
        f"Fallback: {summary['fallback_used']}"
    )

    if args.generate or args.full:
        x_majority = x_train.loc[y_train.eq(0)]
        x_smote_base = x_minority.loc[scores["use_for_smote"].to_numpy()]
        candidates = generate_synthetic_candidates(x_majority, x_smote_base, len(x_minority))
        candidates.to_csv(output_dir / "synthetic_candidates.csv", index=False)

        # After filtering, the retained candidates will be added to the full real set.
        fraud_with_candidates = len(x_minority) + len(candidates)
        total_with_candidates = len(x_train) + len(candidates)
        generation_summary = {
            "alpha": SYNTHESIS_MULTIPLIER,
            "candidate_cap": MAX_CANDIDATES,
            "k_neighbors": K_NEIGHBORS,
            "random_state": RANDOM_STATE,
            "normal_before": len(x_majority),
            "fraud_before": len(x_minority),
            "fraud_after_cleaning": summary["fraud_after_cleaning"],
            "fraud_in_smote_base": len(x_smote_base),
            "fallback_used": summary["fallback_used"],
            "candidates_generated": len(candidates),
            "fraud_rate_before": len(x_minority) / len(x_train),
            "fraud_with_all_candidates": fraud_with_candidates,
            "total_with_all_candidates": total_with_candidates,
            "fraud_rate_with_all_candidates": fraud_with_candidates / total_with_candidates,
        }
        pd.DataFrame([generation_summary]).to_csv(
            output_dir / "synthetic_generation_summary.csv", index=False
        )
        print(
            f"Generated candidates: {len(candidates)} | "
            f"Fraud rate with all candidates: {fraud_with_candidates / total_with_candidates:.4%}",
            flush=True,
        )

        if args.full:
            geometry_scores, geometry_summary = score_geometric_coherence(x_minority, candidates)
            geometric_candidates = candidates.loc[geometry_scores["passes_geometry"].to_numpy()]
            plausibility_scores = score_candidate_plausibility(x_train, y_train, geometric_candidates)
            final_samples = geometric_candidates.loc[plausibility_scores["passes_plausibility"].to_numpy()]

            geometry_scores.to_csv(output_dir / "geometric_filter_scores.csv", index=False)
            plausibility_scores.to_csv(output_dir / "plausibility_scores.csv", index=False)
            final_samples.to_csv(output_dir / "final_synthetic_samples.csv", index=False)
            final_fraud = len(x_minority) + len(final_samples)
            final_total = len(x_train) + len(final_samples)
            final_summary = {
                **generation_summary,
                "outlier_neighbors": summary["outlier_neighbors"],
                "outlier_quantile": summary["outlier_quantile"],
                **geometry_summary,
                "auxiliary_classifier": HistGradientBoostingClassifier.__name__,
                "plausibility_threshold": PLAUSIBILITY_THRESHOLD,
                "final_synthetic_samples": len(final_samples),
                "normal_after": len(x_majority),
                "fraud_after": final_fraud,
                "total_after": final_total,
                "final_fraud_rate": final_fraud / final_total,
            }
            pd.DataFrame([final_summary]).to_csv(
                output_dir / "conservative_smote_summary.csv", index=False
            )
            print(
                f"Geometry survivors: {len(geometric_candidates)} | "
                f"Final synthetic rows: {len(final_samples)} | "
                f"Final fraud rate: {final_fraud / final_total:.4%}"
            )

    print(f"Saved reports to {output_dir}")


if __name__ == "__main__":
    main()
