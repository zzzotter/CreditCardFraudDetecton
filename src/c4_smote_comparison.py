# -*- coding: utf-8 -*-
"""C·DAY4 普通 SMOTE 模型对照：随机森林 vs LightGBM。

- 普通 SMOTE：imblearn `SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)`，
  **只在训练集执行**；验证集保留真实类别比例，仅做预测与评价。
- SMOTE 后的模型使用与 DAY3 相同的固定超参数，但**不再加类别权重**——
  重采样本身就是不平衡处理，二者不叠加（与既有 B4 / smote_experiment.py 口径一致）。
- 统一用 src/evaluation.py 的 evaluate() 计算 AP/Recall/Precision/F1/MCC/FP/FN，
  阈值 0.5；验证集预测概率经 save_validation_predictions() 保留，供 DAY6 阈值选择。
- 对照表合并 DAY3 基线（无处理 / 类别权重）与 DAY4 SMOTE 结果。

输出：
    results/tables/c4_smote_comparison.csv
    results/tables/{rf,lgbm}_smote_valid_predictions.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
SRC_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.insert(0, SRC_DIR)  # 使 `import evaluation` 可用

from evaluation import evaluate, metrics_to_dataframe, save_validation_predictions  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")

SEED = 42
THRESHOLD = 0.5
K_NEIGHBORS = 5
SAMPLING_STRATEGY = "auto"

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# 与 DAY3 一致的固定超参数（SMOTE 模型不加类别权重）
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

C3_CSV = os.path.join(TABLES_DIR, "c3_baseline_comparison.csv")
OUT_CSV = os.path.join(TABLES_DIR, "c4_smote_comparison.csv")


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


def main() -> int:
    os.makedirs(TABLES_DIR, exist_ok=True)
    X_train, y_train, X_valid, y_valid = load_data()

    print("=" * 90)
    print("C·DAY4 普通 SMOTE 模型对照：Random Forest vs LightGBM")
    print("=" * 90)
    print(f"训练集 {X_train.shape[0]:,} 行，欺诈 {int(y_train.sum())}；"
          f"验证集 {X_valid.shape[0]:,} 行，欺诈 {int(y_valid.sum())}")
    print(f"SMOTE：sampling_strategy={SAMPLING_STRATEGY}, k_neighbors={K_NEIGHBORS}, "
          f"random_state={SEED}（只在训练集执行）\n")

    # 1) 普通 SMOTE（只作用于训练集）
    smote = SMOTE(sampling_strategy=SAMPLING_STRATEGY, k_neighbors=K_NEIGHBORS, random_state=SEED)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    n_fraud_after = int(y_sm.sum())
    n_normal_after = int(len(y_sm) - n_fraud_after)
    print(f"SMOTE 后训练集：{len(y_sm):,} 行（正常 {n_normal_after:,}，欺诈 {n_fraud_after:,}，"
          f"合成 {n_fraud_after - int(y_train.sum()):,}）\n")

    rows = []
    for family, file_tag, display_name in [
        ("RandomForest", "rf_smote", "Random Forest (SMOTE)"),
        ("LightGBM", "lgbm_smote", "LightGBM (SMOTE)"),
    ]:
        model = build_model(family)
        t0 = time.time()
        model.fit(X_sm, y_sm)
        y_score = model.predict_proba(X_valid)[:, 1]
        fit_seconds = time.time() - t0

        m = evaluate(y_valid, y_score, threshold=THRESHOLD, model=display_name)
        m["family"] = family
        m["scheme"] = "smote"
        m["fit_seconds"] = round(fit_seconds, 1)
        rows.append(m)

        pred_path = save_validation_predictions(y_valid, y_score, THRESHOLD, file_tag, TABLES_DIR)
        print(f"[{display_name:<24}] AP={m['AP']:.4f} Recall={m['Recall']:.4f} "
              f"Precision={m['Precision']:.4f} F1={m['F1']:.4f} MCC={m['MCC']:.4f} "
              f"FP={m['FP']} FN={m['FN']}  ({fit_seconds:.1f}s)")
        print(f"    预测概率 -> {os.path.relpath(pred_path, PROJECT_ROOT)}")

    # 2) 合并 DAY3 基线（无处理 / 类别权重）
    day3 = pd.read_csv(C3_CSV)
    scheme_map = {"none": "none", "balanced": "class_weight"}
    day3 = day3.rename(columns={"class_weight": "_cw"})
    day3["scheme"] = day3["_cw"].map(scheme_map)
    day3 = day3.drop(columns=["_cw", "model", "threshold"])
    # DAY3 的 n_fraud 为 float，转 int
    for col in ["n_samples", "n_fraud", "TP", "FP", "FN", "TN"]:
        if col in day3.columns:
            day3[col] = day3[col].astype(int)

    combined = pd.concat([day3[TABLE_COLUMNS], pd.DataFrame(rows)[TABLE_COLUMNS]], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)

    print("\n--- 普通 SMOTE 模型对照表（验证集，threshold=0.5）---")
    show = combined.copy()
    for c in ["AP", "Recall", "Precision", "F1", "MCC"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    print(show.to_string(index=False))
    print(f"\n对照表已写出：{os.path.relpath(OUT_CSV, PROJECT_ROOT)}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
