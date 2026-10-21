"""Export saved eps-sweep fits into separate CSV/figure/notes bundles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    from .dbscan_eps_sweep import validate_cohort
    from .evaluation import detection_metrics
    from .unsupervised import knn_scores, load_features, load_labels, plot_clusters, save_csv, sha256_file
except ImportError:
    from dbscan_eps_sweep import validate_cohort
    from evaluation import detection_metrics
    from unsupervised import knn_scores, load_features, load_labels, plot_clusters, save_csv, sha256_file


def run(args):
    root = args.project_root.resolve()
    results = root / "results"
    metadata = json.loads((results / "dbscan_eps_sweep_run.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed":
        raise ValueError("Finish the eps sweep before exporting its results")
    archive = np.load(results / "dbscan_eps_sweep_labels.npz", allow_pickle=False)
    positions = archive["row_index"]
    if str(archive["cohort_id"].item()) != metadata["cohort_id"]:
        raise ValueError("Saved cluster labels do not match the sweep cohort")
    feature_path = root / "data" / "processed" / "X_train.csv"
    label_path = root / "data" / "processed" / "y_train.csv"
    scaler_path = root / "models" / "unsupervised_scaler.joblib"
    validate_cohort(feature_path, metadata, positions)
    if sha256_file(label_path) != metadata["label_source_sha256"] or sha256_file(scaler_path) != metadata["scaler_sha256"]:
        raise ValueError("Labels or scaler changed since the saved clustering experiment")
    available = archive["eps"]
    selected = {}
    for eps in args.eps:
        matches = np.flatnonzero(available == eps)
        if len(matches) != 1:
            raise ValueError(f"eps={eps:g} has no unique completed clustering fit in the archive")
        selected[eps] = archive["cluster_labels"][matches[0]]
    features = load_features(feature_path)
    matrix = joblib.load(scaler_path).transform(features.iloc[positions])
    projection = PCA(n_components=2, svd_solver="full").fit_transform(matrix)
    minimum = metadata["min_samples"]
    scores = knn_scores(matrix, minimum)
    truth = load_labels(label_path, len(features))[positions]
    metrics_table = pd.read_csv(results / "tables" / "dbscan_eps_sweep_metrics.csv")
    exported = []
    for eps, labels in selected.items():
        if labels.shape != truth.shape:
            raise ValueError("Cluster labels are not aligned with source sample rows")
        anomaly = (labels == -1).astype(int)
        metrics = detection_metrics(truth, anomaly, scores, positions)
        reference = metrics_table[metrics_table["eps"].eq(eps) & metrics_table["status"].eq("ok")]
        if len(reference) != 1:
            raise ValueError(f"eps={eps:g} has no unique metrics row")
        reference = reference.iloc[0]
        for field in ["true_positives", "false_positives", "false_negatives", "true_negatives", "recall", "precision", "f1", "mcc", "pr_auc"]:
            if not np.isclose(metrics[field], reference[field], rtol=1e-10, atol=1e-12):
                raise ValueError(f"Exported {field} does not match the recorded sweep")
        name = f"eps_{eps:g}"
        folder = results / name
        (folder / "tables").mkdir(parents=True, exist_ok=True)
        (folder / "figures").mkdir(exist_ok=True)
        frame = pd.DataFrame({"row_index": positions, "cohort_id": metadata["cohort_id"], "data_split": "train",
                              "eps": eps, "min_samples": minimum, "y_true": truth, "cluster_label": labels,
                              "is_anomaly": anomaly, "anomaly_score": scores,
                              "score_definition": f"auxiliary_knn_distance_k={minimum - 1}",
                              "evaluation_scope": metadata["evaluation_scope"]})
        save_csv(frame, folder / "tables" / "dbscan_cluster_results.csv")
        plot_clusters(projection, labels, truth, folder / "figures" / "dbscan_clusters.png", f"DBSCAN eps={eps:g}, min_samples={minimum}")
        clusters = len(set(labels) - {-1})
        now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        notes = f"""# DBSCAN eps={eps:g} 实验记录

## 数据与运行设置

- 原实验时间：{metadata['started_at']} 至 {metadata['completed_at']}。
- 文件导出时间：{now}。
- 参数：eps={eps:g}，min_samples={minimum}，欧氏距离，全部 {matrix.shape[1]} 个特征。
- 数据：A 提供的 X_train.csv，固定抽样 {len(positions):,} 条，真实欺诈 {metrics['n_fraud']} 条，cohort_id=`{metadata['cohort_id']}`。
- 本次直接导出 results/dbscan_eps_sweep_labels.npz 中的真实聚类结果，没有重新训练或重新抽样。CSV 的 row_index 是从 0 开始的训练文件行位置，不是原始信用卡数据行号。
- 所有特征使用同一个仅在训练特征上拟合的 StandardScaler；不读取验证集或测试集。
- 原实验拟合耗时：{reference['fit_seconds']:.3f} 秒；包含进程启动的耗时：{reference['wall_seconds']:.3f} 秒。

## 聚类与异常判定

得到 {clusters} 个簇，噪声候选 {metrics['n_anomaly_candidates']} 条，占抽样数据的 {metrics['n_anomaly_candidates']/len(positions):.2%}。
cluster_label=-1 表示噪声，is_anomaly=1 表示异常候选；噪声不必然是欺诈。算法拟合没有使用 Class，标签仅用于聚类后的评价。

## 实际检测结果

真实欺诈 {metrics['n_fraud']} 条；异常候选 {metrics['n_anomaly_candidates']} 条；检出欺诈 {metrics['true_positives']} 条；误报正常交易 {metrics['false_positives']} 条；漏报 {metrics['false_negatives']} 条；正确判为非异常的正常交易 {metrics['true_negatives']} 条。

| 指标 | 数值 |
| --- | ---: |
| Recall | {metrics['recall']:.4f}（{metrics['recall']:.2%}） |
| Precision | {metrics['precision']:.4f}（{metrics['precision']:.2%}） |
| F1 | {metrics['f1']:.4f} |
| MCC | {metrics['mcc']:.4f} |
| 辅助 kNN AP | {metrics['pr_auc']:.4f} |
| 辅助 kNN ROC-AUC | {metrics['roc_auc']:.4f} |
| 辅助 kNN Recall@100 | {metrics['recall_at_100']:.4f} |
| 辅助 kNN Recall@500 | {metrics['recall_at_500']:.4f} |
| 辅助 kNN Recall@1000 | {metrics['recall_at_1000']:.4f} |

anomaly_score 是距第 {minimum-1} 个其他邻居的欧氏距离，越大代表局部越稀疏；它是辅助排序分数，不是欺诈概率。PR-AUC 采用 average precision。固定 min_samples 后，该辅助分数不会随 eps 改变，因此不同 eps 的 AP 和 Recall@K 可以相同。

## 输出文件

- tables/dbscan_cluster_results.csv：逐样本簇标签、噪声标记、真实标签和辅助分数。
- figures/dbscan_clusters.png：二维 PCA 投影与真实标签对照，实际聚类使用全部特征。
- dbscan_experiment_notes.md：本记录。

![DBSCAN eps={eps:g} 聚类结果](figures/dbscan_clusters.png)

## 评价边界

这是看过基线后的训练样本内探索。本参数由用户从已比较的组合中指定导出，并非独立验证集上确认的最优参数；不能据此声称新交易上的泛化效果。原 eps=3 的主结果仍单独保留。

## 复现导出

在项目根目录执行：`python src/export_dbscan_results.py --eps {eps:g}`。
前提是已运行 unsupervised.py 和 dbscan_eps_sweep.py，生成同 cohort 的结果与归档。
"""
        (folder / "dbscan_experiment_notes.md").write_text(notes, encoding="utf-8")
        exported.append((eps, name, metrics))
        print(f"{folder}: candidates={metrics['n_anomaly_candidates']}, TP={metrics['true_positives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}")
    report = root / "report" / "03_unsupervised_and_comparison.md"
    if report.is_file():
        begin, end = "<!-- DBSCAN_EXPORTS_START -->", "<!-- DBSCAN_EXPORTS_END -->"
        lines = [begin, "## 指定参数的完整聚类结果导出", "", "每组单独保存，CSV、聚类图和实验记录均对应同一参数。", ""]
        for eps, name, metrics in exported:
            lines += [f"- eps={eps:g}：候选 {metrics['n_anomaly_candidates']} 条，检出欺诈 {metrics['true_positives']}/{metrics['n_fraud']}，误报 {metrics['false_positives']} 条，漏报 {metrics['false_negatives']} 条。",
                      f"  [结果 CSV](../results/{name}/tables/dbscan_cluster_results.csv)、[聚类图](../results/{name}/figures/dbscan_clusters.png)、[实验记录](../results/{name}/dbscan_experiment_notes.md)。"]
        lines.append(end)
        section = "\n".join(lines)
        text = report.read_text(encoding="utf-8")
        if begin in text and end in text:
            prefix, rest = text.split(begin, 1)
            _, suffix = rest.split(end, 1)
            text = prefix.rstrip() + "\n\n" + section + suffix
        else:
            text = text.rstrip() + "\n\n" + section + "\n"
        report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--eps", type=float, nargs="+", default=[8, 5])
    options = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0 for value in options.eps):
        parser.error("eps must be finite and positive")
    run(options)
