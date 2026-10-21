"""Member C: reproducible DBSCAN/OPTICS experiments on A's training data.

Run from any directory: python path/to/src/unsupervised.py
Class labels are loaded only AFTER sampling, scaling, clustering and scoring.
The experiment evaluates the clustering sample itself, NOT unseen transactions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_xi
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from .evaluation import cohort_id, detection_metrics
except ImportError:
    from evaluation import cohort_id, detection_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Missing data file: {path}. Obtain A's processed data first.")
    with path.open("rb") as handle:
        if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path.name} is a Git LFS pointer. Run git lfs pull first.")
    return pd.read_csv(path)


def load_features(path: Path) -> pd.DataFrame:
    frame = read_csv_checked(path)
    forbidden = {"class", "label", "target", "y", "y_true", "unnamed: 0"}
    if any(str(c).strip().lower() in forbidden for c in frame.columns):
        raise ValueError("Feature file contains a label or saved index column; check A's interface.")
    if len(frame) < 3 or frame.shape[1] < 2:
        raise ValueError("At least 3 rows and 2 feature columns are required")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("All features must be numeric")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing or infinite values")
    return frame


def load_labels(path: Path, expected_rows: int) -> np.ndarray:
    frame = read_csv_checked(path)
    if list(frame.columns) != ["Class"] or len(frame) != expected_rows:
        raise ValueError("y_train must have one Class column and align row-for-row with X_train")
    values = frame["Class"].to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Class values must be 0 (normal) or 1 (fraud)")
    return values.astype(int)


def sample_indices(n_rows: int, sample_size: int, seed: int) -> np.ndarray:
    """Uniform sampling without replacement, with no access to class labels."""
    if sample_size < 3:
        raise ValueError("sample_size must be at least 3")
    return np.sort(np.random.default_rng(seed).choice(n_rows, min(sample_size, n_rows), replace=False))


def knn_scores(matrix: np.ndarray, min_samples: int) -> np.ndarray:
    """Distance to the (min_samples-1)-th OTHER point; self counts in min_samples.

    This is an auxiliary local sparsity ranking, not a DBSCAN/OPTICS probability.
    Both algorithms use this same score when min_samples is the same.
    """
    model = NearestNeighbors(n_neighbors=min_samples, algorithm="kd_tree", n_jobs=1)
    distances, _ = model.fit(matrix).kneighbors(matrix)
    return distances[:, -1]


def _cluster_worker(connection, algorithm, matrix, parameters):
    try:
        start = perf_counter()
        cls = DBSCAN if algorithm == "DBSCAN" else OPTICS
        model = cls(algorithm="kd_tree", n_jobs=1, **parameters).fit(matrix)
        result = {"status": "ok", "labels": model.labels_, "fit_seconds": perf_counter() - start}
        if algorithm == "OPTICS":
            result.update(
                reachability=model.reachability_,
                ordering=model.ordering_,
                predecessor=model.predecessor_,
                core_distances=model.core_distances_,
            )
        connection.send(result)
    except Exception as exc:
        connection.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def fit_with_timeout(algorithm, matrix, parameters, timeout):
    """A slow fit is terminated, so OPTICS cannot run indefinitely on a laptop."""
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_cluster_worker, args=(sender, algorithm, matrix, parameters))
    start = perf_counter()
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            return {"status": "timeout", "error": f"Fit exceeded {timeout:g}s", "wall_seconds": perf_counter() - start}
        try:
            result = receiver.recv()
        except EOFError:
            result = {"status": "error", "error": "Worker exited without results"}
        result["wall_seconds"] = perf_counter() - start
        return result
    finally:
        receiver.close()
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join()


def cluster_stats(labels):
    return {
        "n_clusters": int(len(set(labels) - {-1})),
        "n_noise": int(np.sum(labels == -1)),
        "noise_ratio": float(np.mean(labels == -1)),
    }


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def remove_stale_primary(root, method):
    """A failed rerun must not leave a previous experiment looking current."""
    name = method.lower()
    paths = [root / "results" / "tables" / f"{name}_cluster_results.csv",
             root / "results" / "figures" / f"{name}_clusters.png"]
    if method == "OPTICS":
        paths.append(root / "results" / "figures" / "optics_reachability.png")
    for path in paths:
        path.unlink(missing_ok=True)


def plotting_module():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_clusters(projection, labels, y_true, path, method):
    plt = plotting_module()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    normal = labels != -1
    if normal.any():
        axes[0].scatter(*projection[normal].T, c=labels[normal], cmap="tab20", s=9, alpha=0.55, label=f"In clusters ({normal.sum():,})")
    if (~normal).any():
        axes[0].scatter(*projection[~normal].T, color="#e28b39", s=12, alpha=0.65, label=f"Noise (-1): {(~normal).sum():,}")
    axes[0].set_title(f"{method}: {len(set(labels) - {-1})} clusters, {(~normal).sum():,} noise")
    axes[0].legend(loc="best")
    for label, color, title, size in [(0, "#9eacb8", "Normal", 9), (1, "#c9364a", "Fraud (evaluation only)", 34)]:
        mask = y_true == label
        axes[1].scatter(*projection[mask].T, color=color, s=size, alpha=0.7, label=title)
    axes[1].set_title("Ground truth, revealed after clustering")
    axes[1].legend(loc="best")
    for ax in axes:
        ax.set(xlabel="Principal component 1", ylabel="Principal component 2")
    fig.suptitle("PCA projection for display only; clustering uses all features")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reachability(result, path):
    plt = plotting_module()
    values = result["reachability"][result["ordering"]]
    finite = np.isfinite(values)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.plot(np.flatnonzero(finite), values[finite], linewidth=0.7, color="#31708e")
    if (~finite).any():
        top = values[finite].max() * 1.05 if finite.any() else 1.0
        ax.scatter(np.flatnonzero(~finite), np.full((~finite).sum(), top), marker="x", color="#c9364a", label="Infinite reachability (display cap)")
        ax.legend()
    ax.set(title="OPTICS reachability ordering", xlabel="Position in OPTICS ordering", ylabel="Reachability distance")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rankings(y_true, score_map, path):
    from sklearn.metrics import precision_recall_curve

    plt = plotting_module()
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    distinct = []
    for method, scores in score_map.items():
        match = next((entry for entry in distinct if np.array_equal(entry[1], scores)), None)
        if match is not None:
            match[0] += f" / {method} (shared score)"
        else:
            distinct.append([method, scores])
    for method, scores in distinct:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.step(recall, precision, where="post", label=method)
    ax.axhline(y_true.mean(), linestyle="--", color="#777777", label=f"Fraud prevalence {y_true.mean():.3%}")
    ax.set(title="Auxiliary kNN sparsity scores (in-sample)", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_notes(method, experiments, primary, manifest, output):
    lines = [
        f"# {method} 实验记录", "",
        f"运行时间：{manifest['created_at']}。数据来自 A 的 X_train.csv，均匀无放回抽样 {manifest['sample_size']:,} 行，随机种子 {manifest['seed']}。",
        f"样本标识：`{manifest['cohort_id']}`；CSV 行位置保存在 `tables/unsupervised_sample_indices.csv`，从 0 开始，不是原始数据行号。", "",
        "全部 30 个特征参与欧氏距离计算。为避免 V 特征方差不同主导距离，StandardScaler 仅在完整训练特征上拟合；不读取验证集和测试集。A 已缩放 Time/Amount，本阶段另存距离计算用缩放器，不修改 A 的数据。", "",
        "参数在评价之前指定，未使用 Class 选参数、抽样或确定异常阈值。导出的主结果对应命令行指定的参数，不按欺诈检测指标挑选最优组合。", "",
        "## 参数实验", "",
        "| 参数 | 状态 | 拟合秒数 | 簇数 | 噪声点 | 噪声比例 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in experiments:
        desc = row["parameters"]
        lines.append(f"| `{desc}` | {row['status']} | {row.get('fit_seconds', float('nan')):.3f} | {row.get('n_clusters', '')} | {row.get('n_noise', '')} | {row.get('noise_ratio', float('nan')):.2%} |")
        if row["status"] != "ok":
            lines.append(f"\n该组合未完成：{row.get('error', '')}。\n")
    lines += ["", "## 异常判定与评分", "", "二元异常候选点定义为 cluster_label == -1。噪声不等于欺诈，必须与聚类后才读取的 Class 标签核对。",
              "连续 anomaly_score 是第 (min_samples - 1) 个其他邻居的欧氏距离（min_samples 包含自身），数值越大代表局部越稀疏。它是辅助 kNN 排序分数，不是算法自带的欺诈概率，也不是簇编号。PR-AUC 采用 average precision（AP）。",
              "DBSCAN 和 OPTICS 在相同样本、相同 min_samples 下共享这项辅助分数，因此 AP、ROC-AUC、Recall@K 可以完全相同；这些排序指标不能用来证明两种聚类算法效果相同。聚类差异应看噪声规则对应的 Recall、Precision、F1、MCC 和误报数。",
              "Recall@K 同分时按 CSV 行位置升序排列；若 K 大于样本量，使用整个样本并记录 effective_k。"]
    if method == "OPTICS":
        lines += ["", "OPTICS 只拟合一次邻域排序图，然后使用不同 xi 提取簇；各 xi 行共享同一次拟合时间，另记 extraction_seconds。可达距离的 inf 保留在 CSV 中，仅绘图时用标记显示，不把无穷值伪装成概率。"]
    if primary is not None:
        lines += ["", "## 主结果", "", f"真实欺诈 {primary['n_fraud']} 条；异常候选 {primary['n_anomaly_candidates']} 条；检出欺诈 {primary['true_positives']} 条；误报正常交易 {primary['false_positives']} 条；漏报 {primary['false_negatives']} 条。",
                  f"Recall={primary['recall']:.4f}，Precision={primary['precision']:.4f}，F1={primary['f1']:.4f}，MCC={primary['mcc']:.4f}，辅助 kNN AP={primary['pr_auc']:.4f}。"]
    else:
        lines += ["", "主参数实验未完成；结果表不填入虚构指标。"]
    lines += ["", "## 局限与下一步", "", "这是训练抽样上的样本内评价，不能直接解释为对未知交易的泛化能力。随机小样本中的欺诈数量很少，指标存在较大波动；后续需要固定参数后设计独立验证方案，并与 B 在完全相同的样本和评价规则下对比。",
              "完整参数表和混淆矩阵计数见 tables 目录。每次拟合有超时限制；超时结果如实记录，不增加无限运行时间。", ""]
    (output / f"{method.lower()}_experiment_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(primary_metrics, manifest, root):
    lines = ["# 无监督异常检测与方法对比", "",
             f"本次实验开始于 {manifest['created_at']}，对应成员 C 的 DBSCAN、OPTICS 初步实验与评价框架。", "",
             f"使用 A 提供的训练特征 {manifest['n_train_rows']:,} 行，按 seed={manifest['seed']} 随机抽取 {manifest['sample_size']:,} 行；样本中实际欺诈数量为 {manifest['sample_fraud_count']}。抽样不按标签分层，不保证保留固定数量的欺诈。",
             "只读取训练集；标签在聚类、缩放和评分全部完成之后用于评价。使用训练集拟合的全特征标准化，使不同 V 特征不因方差大小主导欧氏距离。PCA 仅用于二维展示，聚类使用全部特征。", "",
             "## 已得到的结果", "",
             "| 方法 | 簇数 | 噪声候选 | 真正检出欺诈 | 误报 | Recall | Precision | F1 | MCC |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in primary_metrics:
        lines.append(f"| {row['method']} | {row['n_clusters']} | {row['n_noise']} | {row['true_positives']} | {row['false_positives']} | {row['recall']:.4f} | {row['precision']:.4f} | {row['f1']:.4f} | {row['mcc']:.4f} |")
    lines += ["", "这些数值来自实际运行的主参数组合；参数在读取评价标签之前指定，不以评价结果筛选。"]
    for row in primary_metrics:
        if row["n_anomaly_candidates"]:
            lines += ["", f"{row['method']} 的异常候选中，真正欺诈占 {row['precision']:.2%}；检出了 {row['true_positives']}/{row['n_fraud']} 条欺诈，同时产生 {row['false_positives']} 次误报。这说明较高的 Recall 仍可能伴随很大的人工复核负担。"]
        else:
            lines += ["", f"{row['method']} 的主参数没有产生噪声候选，漏掉全部 {row['n_fraud']} 条欺诈。零误报在这里不能代表有效检测，也不能据此否定整个算法；还需检查其他预先设置的参数及密度结构。"]
    optics_path = root / "results" / "tables" / "optics_parameter_experiments.csv"
    if optics_path.is_file():
        experiments = pd.read_csv(optics_path)
        ok = experiments[experiments["status"] == "ok"]
        if len(ok):
            lines += ["", "OPTICS 的预设 xi 实验说明参数敏感性：", "",
                      "| xi | 噪声候选 | 检出欺诈 | 误报 |", "| ---: | ---: | ---: | ---: |"]
            for row in ok.to_dict("records"):
                lines.append(f"| {row['xi']:g} | {int(row['n_noise'])} | {int(row['true_positives'])} | {int(row['false_positives'])} |")
            lines += ["", "这里只描述预先指定参数的实际结果，没有根据标签改选主参数。"]
    lines += ["",
              "## 如何理解异常分数", "",
              "两种方法使用相同定义的辅助 kNN 距离分数来计算 PR-AUC（AP）、ROC-AUC、Recall@100/500/1000。min_samples 相同时排序分数相同，排序指标也相同；不能把这当成 DBSCAN 与 OPTICS 聚类结果等效的证据。簇编号不是概率。", "",
              "## 方法对比的边界", "",
              "当前结果是聚类样本内评价，不能与 B 的验证集或测试集指标直接排名。统一表保留 data_split、evaluation_scope、cohort_id、参数及阈值定义；只有同 cohort 的数值可在同组比较。",
              "逻辑回归、随机森林结果尚未接入，统一表标为 pending；DENCLUE 仅进行技术可行性检查，正式指标留空。不得把待完成项当作已完成实验。B 可按 README 的预测文件接口使用 compare_models.py 合并同一批交易的结果。", "",
              "## 交付文件", "",
              "- src/unsupervised.py：两种聚类、参数实验、评分、绘图及复现入口。",
              "- results/tables/：主聚类结果、参数表、抽样行位置、特征缩放统计、统一对比表。",
              "- results/figures/：聚类投影图、k 距离曲线、可达距离图和辅助评分 PR 曲线。",
              "- results/unsupervised_run.json：文件 SHA256、样本标识、环境版本、运行参数与评价范围。",
              "- results/dbscan_experiment_notes.md 与 optics_experiment_notes.md：实验记录。",
              "- results/denclue_experiment_notes.md：DENCLUE 可行性检查。", "",
              "## 下一步", "",
              "与 B 统一独立验证及阈值方案；评估扩大样本后的稳定性与运行成本；DENCLUE 完整实现需进一步验证密度吸引子合并与参数选择。测试集继续保留给最终评价。", "",
              "## 技术依据", "",
              "- [DBSCAN 文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)",
              "- [OPTICS 文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)",
              "- [Average precision 定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)", ""]
    (root / "report").mkdir(exist_ok=True)
    (root / "report" / "03_unsupervised_and_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    progress = ["# 成员 C 第一阶段进度说明", "",
                f"本次实验开始时间：{manifest['created_at']}。", "",
                "## 已完成工作", "",
                "- 建立读取 A 数据的无监督实验入口，并完成标签泄漏和输入数据检查。",
                "- 完成的参数实验、耗时、噪声统计与聚类结果见 results/tables/，失败或超时组合在实验记录中单独标明。",
                "- 输出二元异常检测指标、辅助 kNN 排序指标、聚类图、可达距离图及实验记录。",
                "- 建立五种方法的对比框架和同 cohort 监督预测导入接口。",
                "- 提供 DENCLUE 高斯密度吸引子可行性检查脚本；实际运行记录见 results/denclue_experiment_notes.md。", "",
                "## 文件位置", "",
                "代码在 src/，实验结果在 results/tables/ 与 results/figures/，运行元数据为 results/unsupervised_run.json，报告为 report/03_unsupervised_and_comparison.md。复现命令见 README.md。", "",
                "## 初步结论与问题", ""]
    for row in primary_metrics:
        progress.append(f"- {row['method']}：异常候选 {row['n_noise']}，检出欺诈 {row['true_positives']}/{row['n_fraud']}，误报 {row['false_positives']}；Recall={row['recall']:.4f}，Precision={row['precision']:.4f}。")
    progress += ["- 评价仅覆盖训练抽样，不能说明未知交易上的泛化能力。",
                 "- 欺诈样本只有少量，指标可能随抽样变化；DBSCAN 误报和 OPTICS 参数敏感性需继续研究。",
                 "- B 的监督结果尚未接入；DENCLUE 正式聚类实验待完成，不能填入虚构指标。", "",
                 "## 10 月 22 日至 25 日继续完成的内容", "",
                 "与 B 约定独立验证及同样本对比协议；评估参数和不同抽样的稳定性；研究 DENCLUE 的完整簇合并、噪声判定与性能优化。", ""]
    (root / "report" / "03_c_progress.md").write_text("\n".join(progress), encoding="utf-8")


def run(args):
    root = args.project_root.resolve()
    data_dir = root / "data" / "processed"
    output = root / "results"
    tables, figures = output / "tables", output / "figures"
    for folder in [tables, figures, root / "models"]:
        folder.mkdir(parents=True, exist_ok=True)
    print("Loading A's training features (validation/test files are not read)...", flush=True)
    feature_path = data_dir / "X_train.csv"
    features = load_features(feature_path)
    indices = sample_indices(len(features), args.sample_size, args.seed)
    n_samples = len(indices)
    if max(args.min_samples_grid + [args.min_samples, args.optics_min_samples]) > n_samples:
        raise ValueError("min_samples cannot exceed the sampled row count")
    scaler = StandardScaler().fit(features)
    matrix = scaler.transform(features.iloc[indices])
    joblib.dump(scaler, root / "models" / "unsupervised_scaler.joblib")
    file_hash = sha256_file(feature_path)
    cohort = cohort_id(indices, file_hash)
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source": "data/processed/X_train.csv", "source_sha256": file_hash,
        "n_train_rows": len(features), "feature_names": list(features.columns),
        "sample_size": n_samples, "requested_sample_size": args.sample_size, "seed": args.seed,
        "sampling": "numpy.default_rng.choice, uniform without replacement, sorted CSV row positions",
        "cohort_id": cohort, "data_split": "train", "evaluation_scope": "in_sample_clustering",
        "scaling": "StandardScaler fit on all X_train features; no labels, validation or test",
        "score_definition": "distance to (min_samples-1)th other neighbor; auxiliary kNN sparsity, not probability",
        "pr_auc_definition": "average_precision", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    save_csv(pd.DataFrame({"row_index": indices, "cohort_id": cohort}), tables / "unsupervised_sample_indices.csv")
    save_csv(pd.DataFrame({"feature": features.columns, "input_std": features.std(ddof=0), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}), tables / "unsupervised_feature_scaling.csv")
    scores = {}
    for k in sorted(set(args.min_samples_grid + [args.min_samples, args.optics_min_samples])):
        scores[k] = knn_scores(matrix, k)
    fits = []
    primary_results = {}
    db_grid = sorted(set(product(args.eps_grid + [args.eps], args.min_samples_grid + [args.min_samples])))
    for eps, minimum in db_grid:
        parameters = {"eps": eps, "min_samples": minimum}
        print(f"DBSCAN {parameters} ...", flush=True)
        fit = fit_with_timeout("DBSCAN", matrix, parameters, args.fit_timeout)
        row = {"method": "DBSCAN", "parameters": json.dumps(parameters), "eps": eps, "min_samples": minimum,
               **{k: fit[k] for k in ["status", "error", "fit_seconds", "wall_seconds"] if k in fit}}
        if fit["status"] == "ok":
            row.update(cluster_stats(fit["labels"]))
            row["_labels"] = fit["labels"]
            if eps == args.eps and minimum == args.min_samples:
                primary_results["DBSCAN"] = fit
        fits.append(row)
        print(f"  {row['status']}; noise={row.get('n_noise', 'N/A')}", flush=True)
    optics_parameters = {"min_samples": args.optics_min_samples, "min_cluster_size": args.min_cluster_size, "xi": args.xi}
    print(f"OPTICS {optics_parameters} ...", flush=True)
    optics = fit_with_timeout("OPTICS", matrix, optics_parameters, args.fit_timeout)
    for xi in sorted(set(args.xi_grid + [args.xi])):
        row = {"method": "OPTICS", "parameters": json.dumps({**optics_parameters, "xi": xi}), "xi": xi,
               "min_samples": args.optics_min_samples,
               **{k: optics[k] for k in ["status", "error", "fit_seconds", "wall_seconds"] if k in optics}}
        if optics["status"] == "ok":
            start = perf_counter()
            labels, _ = cluster_optics_xi(reachability=optics["reachability"], predecessor=optics["predecessor"],
                                          ordering=optics["ordering"], min_samples=args.optics_min_samples,
                                          min_cluster_size=args.min_cluster_size, xi=xi)
            row["extraction_seconds"] = perf_counter() - start
            row.update(cluster_stats(labels))
            row["_labels"] = labels
            if xi == args.xi:
                primary_results["OPTICS"] = {**optics, "labels": labels}
        fits.append(row)
    print(f"OPTICS {optics['status']}; loading labels for evaluation only now.", flush=True)
    truth_all = load_labels(data_dir / "y_train.csv", len(features))
    truth = truth_all[indices]
    manifest["label_source_sha256"] = sha256_file(data_dir / "y_train.csv")
    manifest["sample_fraud_count"] = int(truth.sum())
    manifest["label_alignment"] = "A's X/y CSVs paired by row position; original row IDs were not exported by A"
    primary_metrics = []
    for row in fits:
        labels = row.pop("_labels", None)
        row.update(cohort_id=cohort, data_split="train", evaluation_scope="in_sample_clustering")
        if labels is not None:
            row.update(detection_metrics(truth, (labels == -1).astype(int), scores[row["min_samples"]], indices))
    projection = PCA(n_components=2, svd_solver="full").fit_transform(matrix)
    score_map = {}
    for method, minimum in [("DBSCAN", args.min_samples), ("OPTICS", args.optics_min_samples)]:
        method_rows = [row for row in fits if row["method"] == method]
        filename = method.lower()
        save_csv(pd.DataFrame(method_rows), tables / f"{filename}_parameter_experiments.csv")
        metric = None
        if method in primary_results:
            fit = primary_results[method]
            labels = fit["labels"]
            score = scores[minimum]
            metric = {
                "method": method, "method_type": "unsupervised", "status": "completed",
                "data_split": "train", "evaluation_scope": "in_sample_clustering", "cohort_id": cohort,
                "min_samples": minimum, "score_definition": f"auxiliary_knn_distance_k={minimum - 1}",
                "fit_seconds": fit["fit_seconds"], "wall_seconds": fit["wall_seconds"],
                "threshold_rule": "cluster_label == -1", **cluster_stats(labels),
                **detection_metrics(truth, (labels == -1).astype(int), score, indices),
            }
            metric["parameters"] = json.dumps({"eps": args.eps, "min_samples": minimum} if method == "DBSCAN" else optics_parameters)
            primary_metrics.append(metric)
            frame = pd.DataFrame({"row_index": indices, "cohort_id": cohort, "data_split": "train", "y_true": truth,
                                  "cluster_label": labels, "is_anomaly": (labels == -1).astype(int), "anomaly_score": score,
                                  "score_definition": f"auxiliary_knn_distance_k={minimum - 1}"})
            if method == "OPTICS":
                rank = np.empty(n_samples, dtype=int)
                rank[fit["ordering"]] = np.arange(n_samples)
                frame["ordering_position"] = rank
                frame["reachability"] = fit["reachability"]
                frame["core_distance"] = fit["core_distances"]
                plot_reachability(fit, figures / "optics_reachability.png")
            save_csv(frame, tables / f"{filename}_cluster_results.csv")
            plot_clusters(projection, labels, truth, figures / f"{filename}_clusters.png", method)
            score_map[f"{method}, k={minimum - 1}"] = score
        else:
            remove_stale_primary(root, method)
        write_notes(method, method_rows, metric, manifest, output)
    plt = plotting_module()
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(np.sort(scores[args.min_samples]), color="#31708e")
    ax.axhline(args.eps, color="#c9364a", linestyle="--", label=f"Primary eps={args.eps}")
    ax.set(title="Training sample k-distance curve (labels not used)", xlabel="Rank by ascending k-distance", ylabel=f"Distance to other neighbor {args.min_samples - 1}")
    ax.legend()
    fig.savefig(figures / "dbscan_k_distance.png", dpi=160)
    plt.close(fig)
    if score_map and len(np.unique(truth)) == 2:
        plot_rankings(truth, score_map, figures / "unsupervised_pr_curves.png")
    else:
        (figures / "unsupervised_pr_curves.png").unlink(missing_ok=True)
    save_csv(pd.DataFrame(primary_metrics), tables / "unsupervised_metrics.csv")
    (output / "unsupervised_run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(primary_metrics, manifest, root)
    try:
        from .dbscan_eps_sweep import refresh_sweep_reports
    except ImportError:
        from dbscan_eps_sweep import refresh_sweep_reports
    refresh_sweep_reports(root)
    try:
        from .dbscan_min_samples_sweep import refresh_reports
    except ImportError:
        from dbscan_min_samples_sweep import refresh_reports
    refresh_reports(root)
    try:
        from .compare_models import build_comparison
    except ImportError:
        from compare_models import build_comparison
    build_comparison(root)
    print(f"Done: {n_samples:,} sampled rows, {truth.sum()} fraud; outputs in {output}", flush=True)
    if len(primary_results) != 2:
        raise RuntimeError("A primary clustering fit failed or timed out. See experiment notes; no result was invented.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=3.0, help="Predeclared primary DBSCAN eps")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--eps-grid", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    parser.add_argument("--min-samples-grid", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--optics-min-samples", type=int, default=10)
    parser.add_argument("--min-cluster-size", type=float, default=0.05, help="OPTICS minimum cluster fraction")
    parser.add_argument("--xi", type=float, default=0.05, help="Predeclared primary OPTICS xi")
    parser.add_argument("--xi-grid", type=float, nargs="+", default=[0.03, 0.05, 0.1])
    parser.add_argument("--fit-timeout", type=float, default=180, help="Maximum seconds per model fit")
    args = parser.parse_args(argv)
    if not 3 <= args.sample_size <= 20000:
        parser.error("Use 3..20000 sampled rows; benchmark before expanding this laptop experiment")
    if any(x <= 0 for x in args.eps_grid + [args.eps]) or args.fit_timeout <= 0:
        parser.error("eps and fit-timeout must be positive")
    if any(x < 2 for x in args.min_samples_grid + [args.min_samples, args.optics_min_samples]):
        parser.error("min_samples must be at least 2 for a non-self neighbor score")
    if not 0 < args.min_cluster_size <= 1 or any(not 0 < x < 1 for x in args.xi_grid + [args.xi]):
        parser.error("min-cluster-size must be in (0,1], xi must be in (0,1)")
    return args


if __name__ == "__main__":
    mp.freeze_support()
    try:
        run(parse_args())
    except (ValueError, RuntimeError) as exc:
        print(f"Experiment stopped: {exc}", file=sys.stderr)
        sys.exit(1)
