"""Explore DBSCAN eps on the exact saved cohort; preserve the original baseline.

This is a retrospective, in-sample sensitivity analysis, not independent model
selection. Labels are read only after every clustering fit has finished.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

try:
    from .evaluation import cohort_id, detection_metrics
    from .unsupervised import cluster_stats, fit_with_timeout, knn_scores, load_features, load_labels, plotting_module, save_csv, sha256_file
except ImportError:
    from evaluation import cohort_id, detection_metrics
    from unsupervised import cluster_stats, fit_with_timeout, knn_scores, load_features, load_labels, plotting_module, save_csv, sha256_file


START = "<!-- DBSCAN_EPS_SWEEP_START -->"
END = "<!-- DBSCAN_EPS_SWEEP_END -->"


def replace_section(text, section):
    """Idempotently update just this analysis without removing the baseline."""
    if START in text:
        if END not in text:
            raise ValueError("Incomplete eps report markers; refusing to discard existing report text")
        prefix, remainder = text.split(START, 1)
        _, suffix = remainder.split(END, 1)
        return prefix.rstrip() + "\n\n" + section + suffix
    return text.rstrip() + "\n\n" + section + "\n"


def validate_cohort(features_path, manifest, positions):
    source_hash = sha256_file(features_path)
    if source_hash != manifest["source_sha256"]:
        raise ValueError("X_train changed since the baseline; do not compare different cohorts")
    expected = cohort_id(positions, source_hash)
    if expected != manifest["cohort_id"] or len(positions) != manifest["sample_size"]:
        raise ValueError("Saved sample rows do not match the baseline cohort")
    if len(np.unique(positions)) != len(positions):
        raise ValueError("Duplicate sampled row positions")


def plot_sweep(frame, path):
    plt = plotting_module()
    valid = frame[frame["status"] == "ok"].sort_values("eps")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    x = valid["eps"]
    axes[0, 0].plot(x, valid["recall"] * 100, "o-", label="Recall")
    axes[0, 0].plot(x, valid["precision"] * 100, "s-", label="Precision")
    axes[0, 0].set(title="Detection trade-off", ylabel="Percent", ylim=(0, 105))
    axes[0, 1].plot(x, valid["false_positives"], "o-", label="False positives")
    axes[0, 1].plot(x, valid["false_negatives"], "s-", label="Missed fraud")
    axes[0, 1].set_yscale("symlog", linthresh=1)
    axes[0, 1].set(title="False alarms and missed fraud", ylabel="Transaction count (symlog)")
    axes[1, 0].plot(x, valid["f1"], "o-", label="F1")
    axes[1, 0].plot(x, valid["mcc"], "s-", label="MCC")
    axes[1, 0].set(title="Combined binary metrics", ylabel="Score")
    axes[1, 1].plot(x, valid["fit_seconds"], "o-", label="DBSCAN fit")
    axes[1, 1].set(title="Measured fit time", ylabel="Seconds")
    for ax in axes.flat:
        ax.axvline(3, color="#777777", linestyle=":", alpha=.8)
        ax.set_xlabel("eps (min_samples fixed)")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(f"DBSCAN eps sensitivity: {int(valid.iloc[0]['n_samples']):,} training rows; in-sample only")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sweep_section(frame, metadata):
    ok = frame[frame["status"] == "ok"].sort_values("eps")
    lines = [START, "## DBSCAN 扩展 eps 敏感性实验", "",
             f"本轮开始时间：{metadata['started_at']}。固定与原实验完全一致的 {metadata['sample_size']:,} 条训练交易（cohort_id=`{metadata['cohort_id']}`）、特征缩放、欧氏距离和 min_samples={metadata['min_samples']}，仅改变 eps。",
             f"预先列出 {len(metadata['eps_grid'])} 个取值：{', '.join(f'{value:g}' for value in metadata['eps_grid'])}。所有组合均记录，未按结果删掉表现较差的组合。", "",
             "这是看过原基线后开展的扩展探索，不是独立验证；本轮标签仅在各组合聚类完成后用于评价。原始 eps=3 的主结果和统一模型对比表保持原有含义，不把探索中最好的数值替换成已验证的最终成绩。", "",
             "| eps | 簇数 | 异常候选 | 检出欺诈 | 漏报 | 误报正常交易 | Recall | Precision | F1 | MCC | 拟合秒数 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in frame.sort_values("eps").to_dict("records"):
        if row["status"] != "ok":
            lines.append(f"| {row['eps']:g} | {row['status']} | | | | | | | | | |")
            continue
        lines.append(f"| {row['eps']:g} | {int(row['n_clusters'])} | {int(row['n_noise'])} | {int(row['true_positives'])}/{int(row['n_fraud'])} | {int(row['false_negatives'])} | {int(row['false_positives'])} | {row['recall']:.2%} | {row['precision']:.2%} | {row['f1']:.4f} | {row['mcc']:.4f} | {row['fit_seconds']:.2f} |")
    if len(ok):
        first, last = ok.iloc[0], ok.iloc[-1]
        lines += ["", "### 本轮观察", "",
                  f"在本次固定样本中，eps 从 {first['eps']:g} 增大到 {last['eps']:g}，异常候选由 {int(first['n_noise'])} 条变为 {int(last['n_noise'])} 条，误报由 {int(first['false_positives'])} 条变为 {int(last['false_positives'])} 条；同时检出欺诈由 {int(first['true_positives'])}/{int(first['n_fraud'])} 变为 {int(last['true_positives'])}/{int(last['n_fraud'])}。这反映密度规则放宽时，误报与漏报之间的权衡。"]
        # Descriptive maxima are explicitly retrospective, never a deployment recommendation.
        for field, title in [("f1", "F1"), ("mcc", "MCC")]:
            best = ok.loc[ok[field].idxmax()]
            lines += ["", f"本轮网格内观察到的最高 {title}={best[field]:.4f}，对应 eps={best['eps']:g}，检出 {int(best['true_positives'])}/{int(best['n_fraud'])} 条欺诈、误报 {int(best['false_positives'])} 条正常交易。这个最大值是在同一样本上事后比较得到的，只作描述，不证明该参数在新数据上最优。"]
        lines += ["", "参数选择应先约定可接受的复核数量、漏报或噪声比例，再结合无标签的 k 距离和稳定性检查；若使用验证标签调参，应单独声明该步骤为有监督的参数选择，并保留独立数据作最终评价。不能只为了得到“正常”数字而挑参数。"]
    lines += ["", "### 图表与复现文件", "",
              "![DBSCAN eps 敏感性实验](../results/figures/dbscan_eps_sweep.png)", "",
              "- 全部指标：results/tables/dbscan_eps_sweep_metrics.csv。",
              "- 每个参数对应的逐样本簇标签：results/dbscan_eps_sweep_labels.npz（eps、row_index、cluster_labels；不含伪造概率）。",
              "- 运行设置与数据指纹：results/dbscan_eps_sweep_run.json。",
              "- 复现命令：`python src/dbscan_eps_sweep.py`。", "",
              "固定 min_samples 后，辅助 kNN 排序分数不会随 eps 改变，因此本轮 AP、ROC-AUC、Recall@K 相同是预期现象；它们不是 eps 调整带来的改善，判断 eps 应看上表中的二元检测指标。", END]
    return "\n".join(lines)


def refresh_sweep_reports(root):
    """Reattach exploration after a baseline report is regenerated, if compatible."""
    metadata_path = root / "results" / "dbscan_eps_sweep_run.json"
    metrics_path = root / "results" / "tables" / "dbscan_eps_sweep_metrics.csv"
    if not metadata_path.is_file() or not metrics_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    baseline = json.loads((root / "results" / "unsupervised_run.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or metadata["cohort_id"] != baseline["cohort_id"]:
        return False
    frame = pd.read_csv(metrics_path)
    section = sweep_section(frame, metadata)
    for relative in ["report/03_unsupervised_and_comparison.md", "results/dbscan_experiment_notes.md"]:
        path = root / relative
        if path.is_file():
            # Image paths are relative to each Markdown file's directory.
            local_section = section if path.parent.name == "report" else section.replace("../results/figures/", "figures/")
            path.write_text(replace_section(path.read_text(encoding="utf-8"), local_section), encoding="utf-8")
    path = root / "report" / "03_c_progress.md"
    if path.is_file():
        summary = f"{START}\n## 新增 eps 扩展实验\n\n固定原样本与 min_samples={metadata['min_samples']}，比较 {len(frame)} 个 eps。全部参数结果、误报与漏报权衡已加入报告；这是样本内探索，原主结果未替换。详见 report/03_unsupervised_and_comparison.md 和 results/tables/dbscan_eps_sweep_metrics.csv。\n{END}"
        path.write_text(replace_section(path.read_text(encoding="utf-8"), summary), encoding="utf-8")
    return True


def run(args):
    root = args.project_root.resolve()
    output = root / "results"
    baseline = json.loads((output / "unsupervised_run.json").read_text(encoding="utf-8"))
    positions = pd.read_csv(output / "tables" / "unsupervised_sample_indices.csv")["row_index"].to_numpy(dtype=int)
    features_path = root / "data" / "processed" / "X_train.csv"
    validate_cohort(features_path, baseline, positions)
    features = load_features(features_path)
    scaler_path = root / "models" / "unsupervised_scaler.joblib"
    scaler = joblib.load(scaler_path)
    matrix = scaler.transform(features.iloc[positions])
    scores = knn_scores(matrix, args.min_samples)
    metadata = {"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                "status": "running", "cohort_id": baseline["cohort_id"], "sample_size": len(positions),
                "source_sha256": baseline["source_sha256"], "scaler_sha256": sha256_file(scaler_path),
                "eps_grid": args.eps, "min_samples": args.min_samples, "fit_timeout": args.fit_timeout,
                "evaluation_scope": "retrospective_in_sample_sensitivity", "data_split": "train",
                "selection_policy": "show all grid results; do not replace baseline or claim independent optimality",
                "versions": baseline["versions"]}
    meta_path = output / "dbscan_eps_sweep_run.json"
    metrics_path = output / "tables" / "dbscan_eps_sweep_metrics.csv"
    rows, label_arrays, completed_eps = [], [], []
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for eps in args.eps:
        print(f"eps={eps:g}, min_samples={args.min_samples}: fitting ...", flush=True)
        result = fit_with_timeout("DBSCAN", matrix, {"eps": eps, "min_samples": args.min_samples}, args.fit_timeout)
        row = {"eps": eps, "min_samples": args.min_samples, "cohort_id": baseline["cohort_id"],
               "evaluation_scope": metadata["evaluation_scope"], "data_split": "train",
               **{key: result[key] for key in ["status", "error", "fit_seconds", "wall_seconds"] if key in result}}
        if result["status"] == "ok":
            row.update(cluster_stats(result["labels"]))
            completed_eps.append(eps)
            label_arrays.append(result["labels"])
        rows.append(row)
        save_csv(pd.DataFrame(rows), metrics_path)
        if label_arrays:
            np.savez_compressed(output / "dbscan_eps_sweep_labels.npz", eps=np.asarray(completed_eps),
                                row_index=positions, cluster_labels=np.asarray(label_arrays), cohort_id=baseline["cohort_id"])
        print(f"  {row['status']}; noise={row.get('n_noise', 'N/A')}; fit={row.get('fit_seconds', float('nan')):.2f}s", flush=True)
    print("All fits finished; reading labels for evaluation now.", flush=True)
    label_path = root / "data" / "processed" / "y_train.csv"
    if sha256_file(label_path) != baseline["label_source_sha256"]:
        raise ValueError("Labels changed since the baseline; cannot compare old and new results")
    truth = load_labels(label_path, len(features))[positions]
    label_lookup = dict(zip(completed_eps, label_arrays))
    for row in rows:
        if row["status"] == "ok":
            labels = label_lookup[row["eps"]]
            row.update(detection_metrics(truth, (labels == -1).astype(int), scores, positions))
    frame = pd.DataFrame(rows)
    save_csv(frame, metrics_path)
    metadata.update(status="completed", completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                    label_source_sha256=baseline["label_source_sha256"], sample_fraud_count=int(truth.sum()),
                    n_completed=len(completed_eps), n_failed=len(rows) - len(completed_eps))
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if completed_eps:
        plot_sweep(frame, output / "figures" / "dbscan_eps_sweep.png")
    else:
        (output / "figures" / "dbscan_eps_sweep.png").unlink(missing_ok=True)
    refresh_sweep_reports(root)
    print(frame.reindex(columns=["eps", "status", "n_noise", "true_positives", "false_positives", "false_negatives", "recall", "precision", "f1", "mcc"]).to_string(index=False))
    if not completed_eps:
        raise RuntimeError("All eps fits failed; inspect the recorded errors")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--eps", type=float, nargs="+", default=[2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10])
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--fit-timeout", type=float, default=180)
    args = parser.parse_args(argv)
    args.eps = sorted(set(args.eps))
    if not args.eps or any(not np.isfinite(eps) or not 0 < eps <= 10 for eps in args.eps):
        parser.error("Use finite eps in (0,10] for this bounded laptop experiment")
    if not 2 <= args.min_samples <= 10000 or not np.isfinite(args.fit_timeout) or args.fit_timeout <= 0:
        parser.error("Use min-samples 2..10000 and a finite positive fit-timeout")
    return args


if __name__ == "__main__":
    mp.freeze_support()
    try:
        run(parse_args())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"eps exploration stopped: {exc}", file=sys.stderr)
        sys.exit(1)
