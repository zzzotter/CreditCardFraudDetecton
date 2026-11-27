# -*- coding: utf-8 -*-
"""C·DAY3 基础模型对照：随机森林 vs LightGBM，无处理 vs 类别权重。

在相同数据划分（data/processed/ 的 70/15/15 分层抽样，随机种子 42）上，
训练 4 个「固定基线」，统一用 src/evaluation.py 的 evaluate() 计算指标：

    1. Random Forest, class_weight=None        （无处理）
    2. Random Forest, class_weight='balanced'  （类别权重）
    3. LightGBM,     class_weight=None         （无处理）
    4. LightGBM,     class_weight='balanced'   （类别权重）

- 阈值统一缺省 0.5；
- 指标统一为 AP、Recall、Precision、F1、MCC、FP、FN（另附 TP/TN/ROC-AUC）；
- 验证集预测概率经 save_validation_predictions() 保留，供 DAY6 阈值选择复用；
- 测试集本轮完全不读取（留待 DAY7 最终评价）。

输出：
    results/tables/c3_baseline_comparison.csv
    results/tables/{tag}_valid_predictions.csv   （4 个，保留验证集预测概率）
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
SRC_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.insert(0, SRC_DIR)  # 使 `import evaluation` 可用

from evaluation import evaluate, metrics_to_dataframe, save_validation_predictions  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")

SEED = 42
THRESHOLD = 0.5

# 冻结特征列序（与 data/processed/README.md 一致）
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# 固定基线超参数（不做调参，避免使用验证集信息）
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

# (family, class_weight, file_tag, display_name)
CONFIGS = [
    ("RandomForest", "none", "rf_none", "Random Forest (无处理)"),
    ("RandomForest", "balanced", "rf_balanced", "Random Forest (类别权重)"),
    ("LightGBM", "none", "lgbm_none", "LightGBM (无处理)"),
    ("LightGBM", "balanced", "lgbm_balanced", "LightGBM (类别权重)"),
]

TABLE_COLUMNS = [
    "family", "class_weight", "model", "threshold", "n_samples", "n_fraud",
    "TP", "FP", "FN", "TN", "AP", "Recall", "Precision", "F1", "MCC",
    "ROC_AUC", "fit_seconds",
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
    if X_train.isna().any().any() or X_valid.isna().any().any():
        raise ValueError("数据存在缺失值。")
    return X_train, y_train, X_valid, y_valid


def build_model(family: str, class_weight: str):
    if family == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        kw = dict(RF_PARAMS)
        if class_weight == "balanced":
            kw["class_weight"] = "balanced"
        return RandomForestClassifier(**kw)

    if family == "LightGBM":
        from lightgbm import LGBMClassifier

        kw = dict(LGB_PARAMS)
        if class_weight == "balanced":
            kw["class_weight"] = "balanced"
        return LGBMClassifier(**kw)

    raise ValueError(f"未知模型族：{family}")


def main() -> int:
    os.makedirs(TABLES_DIR, exist_ok=True)

    X_train, y_train, X_valid, y_valid = load_data()
    print("=" * 90)
    print("C·DAY3 基础模型对照：Random Forest vs LightGBM（无处理 vs 类别权重）")
    print("=" * 90)
    print(f"训练集 {X_train.shape[0]:,} 行 × {X_train.shape[1]} 特征，"
          f"欺诈 {int(y_train.sum())}（{y_train.mean():.4%}）")
    print(f"验证集 {X_valid.shape[0]:,} 行，欺诈 {int(y_valid.sum())}（{y_valid.mean():.4%}）")
    print(f"阈值 threshold={THRESHOLD}；随机种子 {SEED}；测试集本轮不读取。\n")

    rows = []
    for family, class_weight, file_tag, display_name in CONFIGS:
        model = build_model(family, class_weight)
        t0 = time.time()
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_valid)[:, 1]
        fit_seconds = time.time() - t0

        metrics = evaluate(y_valid, y_score, threshold=THRESHOLD, model=display_name)
        metrics["family"] = family
        metrics["class_weight"] = class_weight
        metrics["fit_seconds"] = round(fit_seconds, 1)
        rows.append(metrics)

        pred_path = save_validation_predictions(
            y_valid, y_score, THRESHOLD, file_tag, TABLES_DIR
        )
        print(f"[{display_name:<24}] AP={metrics['AP']:.4f} "
              f"Recall={metrics['Recall']:.4f} Precision={metrics['Precision']:.4f} "
              f"F1={metrics['F1']:.4f} MCC={metrics['MCC']:.4f} "
              f"FP={metrics['FP']} FN={metrics['FN']}  ({fit_seconds:.1f}s)")
        print(f"    预测概率 -> {os.path.relpath(pred_path, PROJECT_ROOT)}")

    table = metrics_to_dataframe(rows)[TABLE_COLUMNS]
    out_csv = os.path.join(TABLES_DIR, "c3_baseline_comparison.csv")
    table.to_csv(out_csv, index=False)

    print("\n--- 基础模型对照表（验证集，threshold=0.5）---")
    print(table.to_string(index=False))
    print(f"\n对照表已写出：{os.path.relpath(out_csv, PROJECT_ROOT)}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
