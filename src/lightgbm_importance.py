# -*- coding: utf-8 -*-
"""
src/lightgbm_importance.py
==========================
信用卡欺诈检测 · 第二阶段 · A（数据与特征优化负责人）· Day 2 方法二
LightGBM 特征重要性（gain 为主，split 为辅）。

口径说明
--------
1. 只用真实训练集训练，验证集/测试集完全不读入；**不使用 early stopping**，
   因为早停必须看验证集标签，一旦用了，特征选择就间接使用了验证集信息。
   因此树的数量固定为 500，属于"不调参的固定基线"。
2. class_weight="balanced"：在 0.17% 正例下，不加权时分裂增益几乎被多数类主导，
   加权后少数类主导的分裂才会进入排名。该选择会在报告中说明。
3. 重要性取 LightGBM 的 gain（分裂带来的总增益），并同时保存 split（被选为分裂点
   的次数）供交叉核对；gain 归一到 share（总和为 1）便于跨方法比较量级。
4. 单次训练的 gain 有随机性（行/列采样），固定 random_state=42 与 deterministic=True
   保证可复现；"重要性是否稳定"不在本轮范围，Day 5 用模型效果验证。

用法
----
    python src/lightgbm_importance.py

输出
----
    results/feature_selection/lightgbm_importance.csv
    results/feature_selection/lightgbm_importance.png
    results/feature_selection/lightgbm_importance_topk.json
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# 0. 全局配置：固定基线超参数（不做调参，避免用到验证集）
# ============================================================
SEED = 42
N_ESTIMATORS = 500
LEARNING_RATE = 0.05
NUM_LEAVES = 31
MIN_CHILD_SAMPLES = 20
SUBSAMPLE = 0.9
COLSAMPLE_BYTREE = 0.9
K_LIST = (10, 15, 20)
TOP_N_FIG = 20

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL_COL = "Class"

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "feature_selection")

CSV_PATH = os.path.join(OUT_DIR, "lightgbm_importance.csv")
PNG_PATH = os.path.join(OUT_DIR, "lightgbm_importance.png")
JSON_PATH = os.path.join(OUT_DIR, "lightgbm_importance_topk.json")


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
# 2. 训练基线模型
# ============================================================
def train_baseline(X: pd.DataFrame, y: pd.Series):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "未安装 lightgbm。请先执行：pip install -U lightgbm（Python 3.13 需 ≥4.6）"
        ) from exc

    model = LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        subsample=SUBSAMPLE,
        subsample_freq=1,
        colsample_bytree=COLSAMPLE_BYTREE,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    model.fit(X, y)
    return model


# ============================================================
# 3. 提取并排序重要性
# ============================================================
def extract_importance(model) -> pd.DataFrame:
    booster = model.booster_
    names = list(booster.feature_name())          # 与训练时列序一致
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)

    if names != FEATURE_ORDER:
        raise ValueError(f"booster 中的特征名与冻结列序不一致：{names}")
    total = gain.sum()
    if total <= 0:
        raise ValueError("所有特征的 gain 均为 0，模型可能未正常训练。")

    table = pd.DataFrame({
        "feature": names,
        "gain": gain,
        "gain_share": gain / total,
        "split": split.astype(int),
        "_idx": np.arange(len(names)),
    })
    table = table.sort_values(["gain", "_idx"], ascending=[False, True], kind="mergesort")
    table = table.reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["rank", "feature", "gain", "gain_share", "split"]]


# ============================================================
# 4. 出图
# ============================================================
def plot_top20(table: pd.DataFrame, n_samples: int, n_positive: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = table.head(TOP_N_FIG).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(d["feature"], d["gain_share"], color="#2c3e50")
    ax.set_xlabel("Gain share (normalized to sum = 1)")
    ax.set_ylabel("Feature")
    ax.set_title(
        f"LightGBM feature importance (top {TOP_N_FIG}, gain)\n"
        f"train only, n={n_samples:,}, positives={n_positive}, "
        f"trees={N_ESTIMATORS}, lr={LEARNING_RATE}, seed={SEED}"
    )
    for i, (_, row) in enumerate(d.iterrows()):
        ax.text(row["gain_share"], i, f" {row['gain_share']:.3%}", va="center", fontsize=8)
    ax.margins(x=0.15)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. 主流程
# ============================================================
def main() -> int:
    print("=" * 78)
    print("Day 2 · 方法二：LightGBM 特征重要性（只用真实训练集，无 early stopping）")
    print("=" * 78)

    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        X, y = load_train_only()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        return 2

    n_samples, n_positive = len(X), int(y.sum())
    print(f"训练集：{n_samples:,} 行 × {X.shape[1]} 特征；正例 {n_positive}（{n_positive/n_samples:.4%}）")
    print(f"参数：trees={N_ESTIMATORS}, lr={LEARNING_RATE}, leaves={NUM_LEAVES}, "
          f"class_weight=balanced, seed={SEED}（不做调参、不看验证集）")

    t0 = time.time()
    try:
        model = train_baseline(X, y)
    except Exception as exc:
        print(f"\n[ERROR] 训练失败：{type(exc).__name__}: {exc}")
        return 2
    fit_seconds = time.time() - t0
    print(f"训练完成，用时 {fit_seconds:.1f} 秒")

    try:
        table = extract_importance(model)
    except Exception as exc:
        print(f"\n[ERROR] 提取重要性失败：{type(exc).__name__}: {exc}")
        return 2

    table.to_csv(CSV_PATH, index=False, float_format="%.10f")
    top_k = {str(k): table["feature"].head(k).tolist() for k in K_LIST}

    # 仅作健全性检查：训练集 AP（不参与任何选择决策）
    try:
        from sklearn.metrics import average_precision_score
        train_ap = float(average_precision_score(y, model.predict_proba(X)[:, 1]))
    except Exception:
        train_ap = None

    try:
        import lightgbm
        lgbm_version = lightgbm.__version__
    except Exception:
        lgbm_version = "unknown"

    card = {
        "method": "lightgbm_importance",
        "estimator": "lightgbm.LGBMClassifier",
        "importance_type": {"primary": "gain", "secondary": "split"},
        "params": {
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "min_child_samples": MIN_CHILD_SAMPLES,
            "subsample": SUBSAMPLE,
            "subsample_freq": 1,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "class_weight": "balanced",
            "random_state": SEED,
            "early_stopping": None,
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
            {"rank": int(r), "feature": f, "gain": float(g),
             "gain_share": float(s), "split": int(sp)}
            for r, f, g, s, sp in zip(table["rank"], table["feature"],
                                      table["gain"], table["gain_share"], table["split"])
        ],
        "sanity": {"train_average_precision": train_ap, "fit_seconds": round(fit_seconds, 1)},
        "script": "src/lightgbm_importance.py",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "lightgbm_version": lgbm_version,
        "notes": [
            "只使用真实训练集；未使用 early stopping，避免通过验证集信息影响特征排序。",
            "hyperparameters 为固定基线，未做任何调参。",
            "gain 归一到 share（总和 1）便于跨方法比较；split 仅作交叉核对。",
        ],
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    plot_top20(table, n_samples, n_positive)

    print("\n--- 完整排序（30 个特征，按 gain 降序）---")
    for _, row in table.iterrows():
        print(f"{int(row['rank']):>3}. {row['feature']:<7} gain={row['gain']:>14.2f}  "
              f"share={row['gain_share']:.4%}  split={int(row['split'])}")

    print("\n--- Top-k 特征列表 ---")
    for k in K_LIST:
        print(f"Top {k}（{k} 个）：{top_k[str(k)]}")

    if train_ap is not None:
        print(f"\n健全性检查（仅训练集，不参与选择）：train AP = {train_ap:.4f}")
    zero_gain = int((table["gain"] <= 0).sum())
    print(f"gain 为 0 的特征数：{zero_gain}")

    print("\n已写出：")
    print(f"  {os.path.relpath(CSV_PATH, PROJECT_ROOT)}")
    print(f"  {os.path.relpath(PNG_PATH, PROJECT_ROOT)}")
    print(f"  {os.path.relpath(JSON_PATH, PROJECT_ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
