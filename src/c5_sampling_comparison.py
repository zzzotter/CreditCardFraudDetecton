# -*- coding: utf-8 -*-
"""C·DAY5 改进采样方法对照：Borderline-SMOTE 与 Conservative SMOTE。

- 只比较「表现较好的模型」：随机森林（RF）与 LightGBM，均使用与 DAY3/DAY4 相同的
  固定超参数且不加类别权重，避免穷举 (模型 × 采样) 组合。
- 采样方案（全部只在训练集执行，验证集保留真实分布）：
    none              无采样（参考基线，读自 DAY4 对照表）
    smote             普通 SMOTE（参考，读自 DAY4 对照表）
    borderline_smote  imblearn BorderlineSMOTE(kind='borderline-1', sampling_strategy='auto',
                        k_neighbors=5, m_neighbors=10, random_state=42)
    conservative_smote 原始训练集 + 517 条最终合成样本（Class=1，
                        来自 results/conservative_smote/final_synthetic_samples.csv）
- 统一用 src/evaluation.py 的 evaluate()；阈值 0.5；保留验证集预测概率。

输出：
    results/tables/c5_improved_sampling_comparison.csv
    results/tables/{family}_{scheme}_valid_predictions.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
SRC_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.insert(0, SRC_DIR)  # 使 `import evaluation` 可用

from evaluation import evaluate, save_validation_predictions  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
CONSERVATIVE_SYNTH = os.path.join(
    PROJECT_ROOT, "results", "conservative_smote", "final_synthetic_samples.csv"
)
C4_CSV = os.path.join(TABLES_DIR, "c4_smote_comparison.csv")
OUT_CSV = os.path.join(TABLES_DIR, "c5_improved_sampling_comparison.csv")

SEED = 42
THRESHOLD = 0.5
K_NEIGHBORS = 5
M_NEIGHBORS = 10
SAMPLING_STRATEGY = "auto"

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

RF_PARAMS = dict(n_estimators=100, random_state=SEED, n_jobs=-1)
LGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=SEED,
    n_jobs=-1,
    deterministic=True,
    force_col_wise=True,
    verbose=-1,
)

TABLE_COLUMNS = [
    "family", "scheme", "n_samples", "n_fraud", "TP", "FP", "FN", "TN",
    "AP", "Recall", "Precision", "F1", "MCC", "ROC_AUC", "fit_seconds",
]


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["Class"].astype(int)
    X_valid = pd.read_csv(os.path.join(DATA_DIR, "X_valid.csv"))
    y_valid = pd.read_csv(os.path.join(DATA_DIR, "y_valid.csv"))["Class"].astype(int)

    if list(X_train.columns) != FEATURE_ORDER:
        raise ValueError(f"训练集特征列序与冻结口径不一致：{list(X_train.columns)}")
    if "Class" in X_train.columns or "Class" in X_valid.columns:
        raise ValueError("特征矩阵中混入标签列 Class，存在数据泄漏。")
    if len(X_train) != len(y_train) or len(X_valid) != len(y_valid):
        raise ValueError("特征与标签行数不对齐。")
    return X_train, y_train, X_valid, y_valid


def build_model(family: str):
    if family == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**RF_PARAMS)
    if family == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**LGB_PARAMS)
    raise ValueError(f"未知模型族：{family}")


def make_samplers(X_train: pd.DataFrame, y_train: pd.Series):
    """返回 {scheme: (X, y)}，只作用于训练集。"""
    samplers = {}

    # Borderline-SMOTE
    bsmote = BorderlineSMOTE(
        sampling_strategy=SAMPLING_STRATEGY,
        k_neighbors=K_NEIGHBORS,
        m_neighbors=M_NEIGHBORS,
        random_state=SEED,
        kind="borderline-1",
    )
    X_bsm, y_bsm = bsmote.fit_resample(X_train, y_train)
    samplers["borderline_smote"] = (X_bsm, y_bsm)

    # Conservative SMOTE：原始训练集 + 517 条最终合成样本（Class=1）
    syn = pd.read_csv(CONSERVATIVE_SYNTH)
    if list(syn.columns) != FEATURE_ORDER:
        raise ValueError(f"Conservative SMOTE 合成样本列序不一致：{list(syn.columns)}")
    X_con = pd.concat([X_train, syn], ignore_index=True)
    y_con = pd.concat(
        [y_train.reset_index(drop=True), pd.Series([1] * len(syn), name="Class")],
        ignore_index=True,
    ).astype(int)
    samplers["conservative_smote"] = (X_con, y_con)

    return samplers


def main() -> int:
    os.makedirs(TABLES_DIR, exist_ok=True)
    X_train, y_train, X_valid, y_valid = load_data()

    print("=" * 90)
    print("C·DAY5 改进采样方法对照：Borderline-SMOTE vs Conservative SMOTE")
    print("=" * 90)
    print(f"训练集 {X_train.shape[0]:,} 行，欺诈 {int(y_train.sum())}；"
          f"验证集 {X_valid.shape[0]:,} 行，欺诈 {int(y_valid.sum())}")
    print(f"比较模型：Random Forest、LightGBM（固定超参数，不加类别权重）\n")

    samplers = make_samplers(X_train, y_train)
    for scheme, (X, y) in samplers.items():
        n_f = int(y.sum())
        print(f"[{scheme:<20}] 训练样本 {len(y):,} 行（正常 {len(y) - n_f:,}，欺诈 {n_f:,}，"
              f"新增 {n_f - int(y_train.sum()):,}）")
    print()

    new_rows = []
    for family in ("RandomForest", "LightGBM"):
        for scheme, (X, y) in samplers.items():
            file_tag = f"{family.lower().split(' ')[0]}_{scheme}"
            display_name = f"{family} ({scheme})"
            model = build_model(family)
            t0 = time.time()
            model.fit(X, y)
            y_score = model.predict_proba(X_valid)[:, 1]
            fit_seconds = time.time() - t0

            m = evaluate(y_valid, y_score, threshold=THRESHOLD, model=display_name)
            m["family"] = family
            m["scheme"] = scheme
            m["fit_seconds"] = round(fit_seconds, 1)
            new_rows.append(m)

            pred_path = save_validation_predictions(y_valid, y_score, THRESHOLD, file_tag, TABLES_DIR)
            print(f"[{display_name:<28}] AP={m['AP']:.4f} Recall={m['Recall']:.4f} "
                  f"Precision={m['Precision']:.4f} F1={m['F1']:.4f} MCC={m['MCC']:.4f} "
                  f"FP={m['FP']} FN={m['FN']}  ({fit_seconds:.1f}s)")

    # 参考行（无采样 / 普通 SMOTE）读自 DAY4 对照表
    ref = pd.read_csv(C4_CSV)
    ref = ref[ref["scheme"].isin(["none", "smote"])].copy()
    ref = ref[TABLE_COLUMNS]

    combined = pd.concat([ref, pd.DataFrame(new_rows)[TABLE_COLUMNS]], ignore_index=True)
    # 稳定排序：family 固定，scheme 按 none -> smote -> borderline -> conservative
    scheme_order = {"none": 0, "smote": 1, "borderline_smote": 2, "conservative_smote": 3}
    combined["_order"] = combined["scheme"].map(scheme_order)
    combined = combined.sort_values(["family", "_order"], kind="mergesort").drop(columns="_order")
    combined = combined.reset_index(drop=True)
    combined.to_csv(OUT_CSV, index=False)

    print("\n--- 改进采样方法对照表（验证集，threshold=0.5）---")
    show = combined.copy()
    for c in ["AP", "Recall", "Precision", "F1", "MCC"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    print(show.to_string(index=False))
    print(f"\n对照表已写出：{os.path.relpath(OUT_CSV, PROJECT_ROOT)}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
