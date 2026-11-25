# -*- coding: utf-8 -*-
"""
src/mutual_information.py
=========================
信用卡欺诈检测 · 第二阶段 · A（数据与特征优化负责人）· Day 2 方法一
互信息（Mutual Information）特征重要性排序。

口径说明
--------
1. 只用真实训练集 X_train / y_train；验证集、测试集完全不读入。
   互信息是有监督度量，一旦用验证/测试标签算分，特征选择就"看过"留出集，
   后续所有模型评估都会失效。
2. 30 个特征全部为连续量（V1..V28 是 PCA 连续值，Time/Amount 已标准化），
   因此显式声明 discrete_features=False，不依赖 'auto' 的推断。
3. 不额外做标准化：互信息估计的是"单个特征与标签"的依赖强度，对单列线性缩放
   不敏感（等价于换距离单位，kNN 半径自适应），再套 scaler 只会让口径与 B、C
   的建模输入不一致。
4. 互信息在高维稀疏正例（344 个）下估计噪声较大，排序只作基准参考；
   "哪种特征选择方法更有用"由后续模型实验回答，不靠 MI 分数本身下结论。

用法
----
    python src/mutual_information.py

输出
----
    results/feature_selection/mutual_information.csv
    results/feature_selection/mutual_information_ranking.png
    results/feature_selection/mutual_information_topk.json
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# 0. 全局配置
# ============================================================
SEED = 42
N_NEIGHBORS = 3                     # Kraskov 估计的近邻数
K_LIST = (10, 15, 20)               # 需要输出的 Top-k
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL_COL = "Class"

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "feature_selection")

CSV_PATH = os.path.join(OUT_DIR, "mutual_information.csv")
PNG_PATH = os.path.join(OUT_DIR, "mutual_information_ranking.png")
JSON_PATH = os.path.join(OUT_DIR, "mutual_information_topk.json")


# ============================================================
# 1. 读取并校验训练集（只读 train）
# ============================================================
def load_train_only() -> tuple[pd.DataFrame, pd.Series]:
    x_path = os.path.join(DATA_DIR, "X_train.csv")
    y_path = os.path.join(DATA_DIR, "y_train.csv")
    for p in (x_path, y_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少文件：{p}")

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)[LABEL_COL].astype(int)

    if list(X.columns) != FEATURE_ORDER:
        raise ValueError(
            f"特征列与冻结口径不一致。\n期望：{FEATURE_ORDER}\n实际：{list(X.columns)}"
        )
    if LABEL_COL in X.columns:
        raise ValueError("特征矩阵中混入了标签列 Class，存在数据泄漏。")
    if len(X) != len(y):
        raise ValueError(f"X 有 {len(X)} 行，y 有 {len(y)} 行，不对齐。")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("训练数据中存在缺失值，请先回到 Day 1 审计排查。")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"标签出现非 0/1 取值：{sorted(y.unique())}")
    return X, y


# ============================================================
# 2. 计算互信息
# ============================================================
def compute_mi(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError as exc:
        raise ImportError("未安装 scikit-learn，请先执行：pip install scikit-learn") from exc

    return mutual_info_classif(
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int64),
        discrete_features=False,
        n_neighbors=N_NEIGHBORS,
        random_state=SEED,
    )


def build_ranking(mi: np.ndarray) -> pd.DataFrame:
    """按分数降序排序；分数相同时按冻结列序（原始下标）细分，保证结果唯一可复现。"""
    table = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "mi_score": np.asarray(mi, dtype=float),
        "_idx": np.arange(len(mi)),
    })
    table = table.sort_values(["mi_score", "_idx"], ascending=[False, True], kind="mergesort")
    table = table.reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["rank", "feature", "mi_score"]]


# ============================================================
# 3. 出图
# ============================================================
def plot_ranking(table: pd.DataFrame, n_samples: int, n_positive: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = table.iloc[::-1]  # 反转后最高分在最上方
    colors = ["#c0392b" if r <= 20 else "#95a5a6" for r in d["rank"]]

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(d["feature"], d["mi_score"], color=colors)
    ax.set_xlabel("Mutual information with Class (nats)")
    ax.set_ylabel("Feature")
    ax.set_title(
        f"Mutual information ranking (train only)\n"
        f"n={n_samples:,}, positives={n_positive}, seed={SEED}, kNN k={N_NEIGHBORS}"
    )
    for i, (_, row) in enumerate(d.iterrows()):
        ax.text(row["mi_score"], i, f" {row['mi_score']:.4f}", va="center", fontsize=7)
    ax.margins(x=0.12)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 4. 主流程
# ============================================================
def main() -> int:
    print("=" * 78)
    print("Day 2 · 方法一：互信息特征选择（只用真实训练集）")
    print("=" * 78)

    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        X, y = load_train_only()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        return 2

    n_samples, n_positive = len(X), int(y.sum())
    print(f"训练集：{n_samples:,} 行 × {X.shape[1]} 特征；正例 {n_positive}（{n_positive/n_samples:.4%}）")
    print(f"参数：discrete_features=False, n_neighbors={N_NEIGHBORS}, random_state={SEED}")
    print("计算中（约 1–3 分钟）...")

    try:
        mi = compute_mi(X, y)
    except Exception as exc:
        print(f"\n[ERROR] 互信息计算失败：{type(exc).__name__}: {exc}")
        return 2

    table = build_ranking(mi)
    table.to_csv(CSV_PATH, index=False, float_format="%.10f")

    top_k = {str(k): table["feature"].head(k).tolist() for k in K_LIST}

    card = {
        "method": "mutual_information",
        "estimator": "sklearn.feature_selection.mutual_info_classif",
        "params": {
            "discrete_features": False,
            "n_neighbors": N_NEIGHBORS,
            "random_state": SEED,
        },
        "input": {
            "X": "data/processed/X_train.csv",
            "y": "data/processed/y_train.csv",
            "n_samples": n_samples,
            "n_features": int(X.shape[1]),
            "n_positive": n_positive,
            "positive_rate": n_positive / n_samples,
            "split_info": "results/feature_selection/data_split_info.json",
        },
        "top_k": top_k,
        "full_ranking": [
            {"rank": int(r), "feature": f, "mi_score": float(s)}
            for r, f, s in zip(table["rank"], table["feature"], table["mi_score"])
        ],
        "script": "src/mutual_information.py",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "notes": [
            "只使用真实训练集，验证/测试集未读入。",
            "互信息为连续-离散 kNN 估计，数值单位 nats，量级普遍偏小，比较时看相对次序。",
            "分数相同者按冻结列序细分名次，故排序唯一可复现。",
        ],
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    plot_ranking(table, n_samples, n_positive)

    print("\n--- 完整排序（30 个特征，降序）---")
    for _, row in table.iterrows():
        print(f"{int(row['rank']):>3}. {row['feature']:<7} {row['mi_score']:.6f}")

    print("\n--- Top-k 特征列表 ---")
    for k in K_LIST:
        print(f"Top {k}（{k} 个）：{top_k[str(k)]}")

    print("\n已写出：")
    print(f"  {os.path.relpath(CSV_PATH, PROJECT_ROOT)}")
    print(f"  {os.path.relpath(PNG_PATH, PROJECT_ROOT)}")
    print(f"  {os.path.relpath(JSON_PATH, PROJECT_ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
