"""Fix eps=8 and explore min_samples on the saved 10,000-row training cohort."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    from .dbscan_eps_sweep import validate_cohort
    from .evaluation import detection_metrics
    from .unsupervised import cluster_stats, fit_with_timeout, load_features, load_labels, plot_clusters, plotting_module, save_csv, sha256_file
except ImportError:
    from dbscan_eps_sweep import validate_cohort
    from evaluation import detection_metrics
    from unsupervised import cluster_stats, fit_with_timeout, load_features, load_labels, plot_clusters, plotting_module, save_csv, sha256_file


START = "<!-- DBSCAN_MIN_SAMPLES_START -->"
END = "<!-- DBSCAN_MIN_SAMPLES_END -->"


def update_report(path, section):
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if START in text:
        if END not in text:
            raise ValueError("Incomplete min_samples report markers")
        prefix, remainder = text.split(START, 1)
        _, suffix = remainder.split(END, 1)
        text = prefix.rstrip() + "\n\n" + section + suffix
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    path.write_text(text, encoding="utf-8")


def plot_experiment(frame, path):
    plt = plotting_module()
    ok = frame[frame.status.eq("ok")].sort_values("min_samples")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    x = ok.min_samples
    for field, label in [("recall", "Recall"), ("precision", "Precision")]:
        axes[0, 0].plot(x, ok[field]*100, "o-", label=label)
    axes[0, 0].set(title="Detection trade-off", ylabel="Percent", ylim=(0, 105))
    for field, label in [("false_positives", "False positives"), ("false_negatives", "Missed fraud")]:
        axes[0, 1].plot(x, ok[field], "o-", label=label)
    axes[0, 1].set_yscale("symlog", linthresh=1)
    axes[0, 1].set(title="False alarms and missed fraud", ylabel="Transaction count (symlog)")
    for field, label in [("f1", "F1"), ("mcc", "MCC")]:
        axes[1, 0].plot(x, ok[field], "o-", label=label)
    axes[1, 0].set(title="Combined binary metrics", ylabel="Score")
    axes[1, 1].plot(x, ok.n_noise, "o-", label="Noise candidates")
    axes[1, 1].set(title="Number of noise candidates", ylabel="Transaction count")
    for ax in axes.flat:
        ax.axvline(10, linestyle=":", color="#777777", alpha=.8)
        ax.set_xlabel("min_samples (eps fixed)")
        ax.set_xticks(x)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(f"DBSCAN min_samples sensitivity: eps={ok.iloc[0].eps:g}; in-sample only")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def experiment_notes(frame, metadata):
    ok = frame[frame.status.eq("ok")].sort_values("min_samples")
    lines = ["# DBSCAN 固定 eps=8 的 min_samples 实验记录", "",
             f"实验时间：{metadata['started_at']} 至 {metadata['completed_at']}。", "",
             "## 实验设计", "",
             f"固定 eps={metadata['eps']:g}、欧氏距离、全部 30 个输入特征、原训练集缩放器和原抽样 {metadata['sample_size']:,} 条交易（cohort_id=`{metadata['cohort_id']}`）。仅改变 min_samples，预先设置为 {', '.join(map(str,metadata['min_samples_grid']))}。",
             "min_samples 包含样本自身。抽样和所有模型拟合均不使用 Class；所有拟合完成后才读取标签评价。不读取验证集、测试集。",
             "eps=8 是上一轮同样本探索中表现较好的取值，本轮属于继续进行的样本内敏感性分析，不是独立验证或无偏的最终参数选择。", "",
             "## 全部参数结果", "",
             "| min_samples | 簇数 | 异常候选 | 检出欺诈 | 误报 | 漏报 | Recall | Precision | F1 | MCC | 拟合秒数 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in frame.sort_values("min_samples").to_dict("records"):
        if row["status"] != "ok":
            lines.append(f"| {row['min_samples']} | {row['status']} | | | | | | | | | |")
            lines.append(f"\n失败或超时原因：{row.get('error','')}。\n")
        else:
            lines.append(f"| {row['min_samples']} | {int(row['n_clusters'])} | {int(row['n_noise'])} | {int(row['true_positives'])}/{int(row['n_fraud'])} | {int(row['false_positives'])} | {int(row['false_negatives'])} | {row['recall']:.2%} | {row['precision']:.2%} | {row['f1']:.4f} | {row['mcc']:.4f} | {row['fit_seconds']:.2f} |")
    if len(ok):
        first, last = ok.iloc[0], ok.iloc[-1]
        lines += ["", "## 观察与结论", "",
                  f"min_samples 从 {int(first.min_samples)} 增大到 {int(last.min_samples)} 时，异常候选从 {int(first.n_noise)} 条变为 {int(last.n_noise)} 条，误报从 {int(first.false_positives)} 条变为 {int(last.false_positives)} 条；检出欺诈从 {int(first.true_positives)} 条变为 {int(last.true_positives)} 条。固定 eps 时，更高的 min_samples 要求更高局部密度，因此更多交易可能被标为噪声，但噪声仍不等于欺诈。"]
        for field, label in [("f1", "F1"), ("mcc", "MCC")]:
            best = ok[ok[field].eq(ok[field].max())]
            values = ', '.join(str(int(value)) for value in best.min_samples)
            lines += ["", f"本轮观察到的最高 {label}={best.iloc[0][field]:.4f}，对应 min_samples={values}。这是同一批少量欺诈样本上的事后比较，只能作为后续独立验证的候选依据。"]
    lines += ["", "## 连续评分与评价口径", "",
              "二元异常规则为 cluster_label=-1，基于该规则计算 Recall、Precision、F1、MCC 和误报数。异常候选不是已确认的欺诈。",
              "辅助 anomaly_score 为距第 (min_samples-1) 个其他邻居的欧氏距离；min_samples 包含自身。不同 min_samples 的 k 不同，所以本轮辅助 AP、ROC-AUC 和 Recall@K 也可能改变。它们衡量不同 k 的 kNN 排序，不是 DBSCAN 的原生欺诈概率；不能把这部分变化完全归因于簇划分改善。PR-AUC 仍采用 average precision。", "",
              "## 文件与复现", "",
              "- results/tables/dbscan_min_samples_sweep_metrics.csv：全部参数及检测和排序指标。",
              "- results/figures/dbscan_min_samples_sweep.png：对比图。",
              "- results/dbscan_min_samples_sweep_labels.npz：min_samples、row_index 与逐样本 cluster_labels。",
              "- results/dbscan_min_samples_sweep_run.json：参数、数据指纹、版本与完成状态。",
              "- results/eps_8_min_samples_sweep/min_samples_N/：每组独立的 tables/dbscan_cluster_results.csv、figures/dbscan_clusters.png 和 dbscan_experiment_notes.md。",
              "- 复现：`python src/dbscan_min_samples_sweep.py`。", "",
              "![固定 eps=8 的 min_samples 对比](figures/dbscan_min_samples_sweep.png)", ""]
    return "\n".join(lines)


def refresh_reports(root):
    results = root / "results"
    meta_path = results / "dbscan_min_samples_sweep_run.json"
    if not meta_path.is_file():
        return False
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    baseline = json.loads((results / "unsupervised_run.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed" or metadata["cohort_id"] != baseline["cohort_id"]:
        return False
    frame = pd.read_csv(results / "tables" / "dbscan_min_samples_sweep_metrics.csv")
    notes = experiment_notes(frame, metadata)
    (results / "dbscan_min_samples_experiment_notes.md").write_text(notes, encoding="utf-8")
    section = START + "\n" + notes.replace("\n## ", "\n### ").replace("# DBSCAN 固定 eps=8 的 min_samples 实验记录", "## 固定 eps=8 的 min_samples 扩展实验", 1)
    section = section.replace("](figures/", "](../results/figures/") + "\n" + END
    update_report(root / "report" / "03_unsupervised_and_comparison.md", section)
    update_report(root / "report" / "03_c_progress.md", START + f"\n## 固定 eps=8 的 min_samples 实验\n\n已记录 {len(frame)} 个 min_samples 的实验、逐样本结果与图表，详见 results/dbscan_min_samples_experiment_notes.md。评价仍为训练样本内探索。\n" + END)
    return True


def run(args):
    root = args.project_root.resolve()
    output = root / "results"
    baseline = json.loads((output / "unsupervised_run.json").read_text(encoding="utf-8"))
    previous = json.loads((output / "dbscan_eps_sweep_run.json").read_text(encoding="utf-8"))
    positions = pd.read_csv(output / "tables" / "unsupervised_sample_indices.csv").row_index.to_numpy(dtype=int)
    feature_path = root / "data" / "processed" / "X_train.csv"
    scaler_path = root / "models" / "unsupervised_scaler.joblib"
    validate_cohort(feature_path, baseline, positions)
    if previous["cohort_id"] != baseline["cohort_id"] or sha256_file(scaler_path) != previous["scaler_sha256"]:
        raise ValueError("Current cohort or scaler does not match the prior eps experiment")
    if max(args.min_samples) > len(positions):
        raise ValueError("min_samples exceeds the sample size")
    features = load_features(feature_path)
    matrix = joblib.load(scaler_path).transform(features.iloc[positions])
    metadata = {"status": "running", "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                "eps": args.eps, "min_samples_grid": args.min_samples, "cohort_id": baseline["cohort_id"],
                "source_sha256": baseline["source_sha256"], "scaler_sha256": previous["scaler_sha256"],
                "sample_size": len(positions), "fit_timeout": args.fit_timeout, "versions": baseline["versions"],
                "evaluation_scope": "retrospective_in_sample_sensitivity", "data_split": "train"}
    meta_path = output / "dbscan_min_samples_sweep_run.json"
    metrics_path = output / "tables" / "dbscan_min_samples_sweep_metrics.csv"
    meta_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    rows, label_map = [], {}
    for minimum in args.min_samples:
        print(f"eps={args.eps:g}, min_samples={minimum}: fitting ...",flush=True)
        fit = fit_with_timeout("DBSCAN",matrix,{"eps":args.eps,"min_samples":minimum},args.fit_timeout)
        row = {"eps":args.eps,"min_samples":minimum,"cohort_id":baseline["cohort_id"],
               "data_split":"train","evaluation_scope":metadata["evaluation_scope"],
               **{key:fit[key] for key in ["status","error","fit_seconds","wall_seconds"] if key in fit}}
        if fit["status"] == "ok":
            label_map[minimum] = fit["labels"]
            row.update(cluster_stats(fit["labels"]))
        rows.append(row)
        save_csv(pd.DataFrame(rows),metrics_path)
        if label_map:
            np.savez_compressed(output/"dbscan_min_samples_sweep_labels.npz",min_samples=np.asarray(list(label_map)),
                                row_index=positions,cluster_labels=np.asarray(list(label_map.values())),eps=args.eps,cohort_id=baseline["cohort_id"])
        print(f"  {row['status']}; noise={row.get('n_noise','N/A')}",flush=True)
    if not label_map:
        raise RuntimeError("No fit succeeded; see recorded failures")
    print("All fits finished; calculating neighbor scores and reading labels for evaluation.",flush=True)
    distances = NearestNeighbors(n_neighbors=max(label_map),algorithm="kd_tree",n_jobs=1).fit(matrix).kneighbors(matrix)[0]
    label_path = root/"data"/"processed"/"y_train.csv"
    if sha256_file(label_path) != baseline["label_source_sha256"]:
        raise ValueError("Labels changed since the baseline")
    truth = load_labels(label_path,len(features))[positions]
    projection = PCA(n_components=2,svd_solver="full").fit_transform(matrix)
    for row in rows:
        minimum = row["min_samples"]
        if minimum not in label_map:
            continue
        labels = label_map[minimum]
        scores = distances[:,minimum-1]
        metrics = detection_metrics(truth,(labels==-1).astype(int),scores,positions)
        row.update(metrics)
        row["score_definition"] = f"auxiliary_knn_distance_k={minimum-1}"
        folder = output/"eps_8_min_samples_sweep"/f"min_samples_{minimum}"
        (folder/"tables").mkdir(parents=True,exist_ok=True)
        (folder/"figures").mkdir(exist_ok=True)
        frame = pd.DataFrame({"row_index":positions,"cohort_id":baseline["cohort_id"],"data_split":"train",
                              "eps":args.eps,"min_samples":minimum,"y_true":truth,"cluster_label":labels,
                              "is_anomaly":(labels==-1).astype(int),"anomaly_score":scores,
                              "score_definition":row["score_definition"],"evaluation_scope":metadata["evaluation_scope"]})
        save_csv(frame,folder/"tables"/"dbscan_cluster_results.csv")
        plot_clusters(projection,labels,truth,folder/"figures"/"dbscan_clusters.png",f"DBSCAN eps={args.eps:g}, min_samples={minimum}")
        notes = f"""# DBSCAN eps={args.eps:g} min_samples={minimum} 实验记录

本次使用固定的 {len(positions):,} 条训练抽样交易，cohort_id=`{baseline['cohort_id']}`。原训练集拟合的缩放器、全部 30 个特征和欧氏距离保持不变，拟合过程不使用 Class。拟合耗时 {row['fit_seconds']:.3f} 秒。

得到 {row['n_clusters']} 个簇，噪声候选 {row['n_noise']} 条。真实欺诈 {metrics['n_fraud']} 条；检出 {metrics['true_positives']} 条；误报正常交易 {metrics['false_positives']} 条；漏报 {metrics['false_negatives']} 条；正确判为非异常的正常交易 {metrics['true_negatives']} 条。

Recall={metrics['recall']:.4f}，Precision={metrics['precision']:.4f}，F1={metrics['f1']:.4f}，MCC={metrics['mcc']:.4f}。
辅助 kNN AP={metrics['pr_auc']:.4f}，Recall@100={metrics['recall_at_100']:.4f}。

cluster_label=-1 是噪声候选，不一定是欺诈。anomaly_score 是距第 {minimum-1} 个其他邻居的距离，属于辅助排序分数，不是欺诈概率。这里只在同一训练样本内事后评价，不能直接解释为泛化能力。

- tables/dbscan_cluster_results.csv：逐样本结果。
- figures/dbscan_clusters.png：PCA 仅用于展示，聚类使用全部特征。
- [完整参数比较](../../dbscan_min_samples_experiment_notes.md)。

![聚类结果](figures/dbscan_clusters.png)
"""
        (folder/"dbscan_experiment_notes.md").write_text(notes,encoding="utf-8")
    frame = pd.DataFrame(rows)
    save_csv(frame,metrics_path)
    metadata.update(status="completed",completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                    label_source_sha256=baseline["label_source_sha256"],sample_fraud_count=int(truth.sum()),
                    n_completed=len(label_map),n_failed=len(rows)-len(label_map))
    meta_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    plot_experiment(frame,output/"figures"/"dbscan_min_samples_sweep.png")
    refresh_reports(root)
    print(frame.reindex(columns=["eps","min_samples","status","n_noise","true_positives","false_positives","false_negatives","recall","precision","f1","mcc"]).to_string(index=False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--eps",type=float,default=8)
    parser.add_argument("--min-samples",type=int,nargs="+",default=[5,10,20,30,50])
    parser.add_argument("--fit-timeout",type=float,default=180)
    args = parser.parse_args(argv)
    args.min_samples = sorted(set(args.min_samples))
    if args.eps != 8:
        parser.error("This isolated experiment fixes eps=8")
    if any(value<2 or value>10000 for value in args.min_samples):
        parser.error("min_samples must be 2..10000")
    if not np.isfinite(args.fit_timeout) or args.fit_timeout<=0:
        parser.error("fit-timeout must be finite and positive")
    return args


if __name__ == "__main__":
    mp.freeze_support()
    run(parse_args())
