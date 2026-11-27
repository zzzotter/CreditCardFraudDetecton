# -*- coding: utf-8 -*-
"""统一评价函数与指标定义（C·DAY2）。

本模块是项目内所有监督 / 半监督 / 无监督方法**唯一**的评价口径来源。
后续 DAY3—DAY7 的模型对照、阈值选择与测试集评价都必须复用本模块，
保证横向比较时指标定义完全一致。

约定
----
- 正类 = 欺诈（label = 1），负类 = 正常（label = 0）。
- 连续得分 ``y_score`` 一律表示「该样本为欺诈」的得分 / 概率，越大越可能是欺诈：
    - 监督模型取 ``model.predict_proba(X)[:, 1]``；
    - 无监督方法取异常得分（越大越异常）。
- 二值预测 ``y_pred`` 由 ``y_score >= threshold`` 得到，阈值缺省 ``0.5``。
- 验证集与测试集的真实类别比例必须保留（不重采样），只在其上做预测与评价。

统一指标（与《第二次工作安排》第 10 条一致）
------------------------------------------
- AP        : Average Precision（平均精确率），即 ``sklearn.average_precision_score``，
              等价于 PR-AUC。
- Recall    : TP / (TP + FN)，真实欺诈中被检出的比例（召回率 / 查全率）。
- Precision : TP / (TP + FP)，预测为欺诈的样本中真实欺诈的比例（精确率 / 查准率）。
- F1        : 2·Precision·Recall / (Precision + Recall)，P 与 R 的调和平均。
- MCC       : (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))，
              综合混淆矩阵四类计数的相关系数，取值 [−1, 1]，对类别不平衡更稳健。
- FP        : 误报数（把正常交易判为欺诈的数量）。
- FN        : 漏报数（把真实欺诈漏掉的数量）。

此外返回 TP、TN、ROC-AUC（辅助）与样本数、欺诈数，便于核对混淆矩阵。
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 正类与负类标签
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

# 统一指标列名（按报告顺序）
METRIC_KEYS = [
    "n_samples",
    "n_fraud",
    "TP",
    "FP",
    "FN",
    "TN",
    "AP",
    "Recall",
    "Precision",
    "F1",
    "MCC",
    "ROC_AUC",
]


def _as_1d_float(y: Sequence) -> np.ndarray:
    """转为 float 一维数组，并做基本形状检查。"""
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("输入必须是长度一致、非空的一维数组")
    return arr


def _check_binary_labels(arr: np.ndarray, name: str) -> None:
    if not np.isin(arr, [NEGATIVE_LABEL, POSITIVE_LABEL]).all():
        raise ValueError(f"{name} 必须只含二值标签：0=正常，1=欺诈")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """由二值标签计算 TP / FP / FN / TN。"""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]
    ).ravel()
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def evaluate(
    y_true: Sequence,
    y_score: Sequence,
    threshold: float = 0.5,
    model: Optional[str] = None,
) -> Dict[str, Union[int, float, str, None]]:
    """统一评价函数：由真实标签与连续得分计算全部统一指标。

    参数
    ----
    y_true    : 真实标签（0/1，1 为欺诈）。
    y_score   : 连续得分 / 欺诈概率，越大越可能为欺诈。
    threshold : 二值化阈值，缺省 0.5。
    model     : 可选的方法名，写入返回字典，便于拼对比表。

    返回
    ----
    有序字典，键依次为：model, threshold, n_samples, n_fraud,
    TP, FP, FN, TN, AP, Recall, Precision, F1, MCC, ROC_AUC。
    """
    truth = _as_1d_float(y_true)
    score = _as_1d_float(y_score)
    if score.shape != truth.shape:
        raise ValueError("y_true 与 y_score 长度必须一致")
    _check_binary_labels(truth, "y_true")
    if not np.isfinite(score).all():
        raise ValueError("y_score 必须为有限数值")

    # 阈值二值化：得分 >= 阈值 判为欺诈
    y_pred = (score >= threshold).astype(int)

    both_classes = len(np.unique(truth)) == 2
    counts = confusion_counts(truth, y_pred)

    result: Dict[str, Union[int, float, str, None]] = {}
    if model is not None:
        result["model"] = model
    result["threshold"] = float(threshold)
    result["n_samples"] = int(truth.size)
    result["n_fraud"] = int(truth.sum())
    result.update(counts)

    result["AP"] = (
        float(average_precision_score(truth, score)) if both_classes else np.nan
    )
    result["Recall"] = float(recall_score(truth, y_pred, zero_division=0))
    result["Precision"] = float(precision_score(truth, y_pred, zero_division=0))
    result["F1"] = float(f1_score(truth, y_pred, zero_division=0))
    result["MCC"] = (
        float(matthews_corrcoef(truth, y_pred)) if both_classes else np.nan
    )
    result["ROC_AUC"] = (
        float(roc_auc_score(truth, score)) if both_classes else np.nan
    )
    return result


def evaluate_binary(
    y_true: Sequence, y_pred: Sequence, model: Optional[str] = None
) -> Dict[str, Union[int, float, str, None]]:
    """仅给定二值预测时计算统一指标（AP/ROC-AUC 置为 NaN，因缺少连续得分）。

    主要用于无监督硬标签或外部给定的 0/1 判定结果。
    """
    truth = _as_1d_float(y_true)
    pred = _as_1d_float(y_pred)
    if pred.shape != truth.shape:
        raise ValueError("y_true 与 y_pred 长度必须一致")
    _check_binary_labels(truth, "y_true")
    _check_binary_labels(pred, "y_pred")

    both_classes = len(np.unique(truth)) == 2
    counts = confusion_counts(truth, pred)

    result: Dict[str, Union[int, float, str, None]] = {}
    if model is not None:
        result["model"] = model
    result["threshold"] = None
    result["n_samples"] = int(truth.size)
    result["n_fraud"] = int(truth.sum())
    result.update(counts)
    result["AP"] = np.nan
    result["Recall"] = float(recall_score(truth, pred, zero_division=0))
    result["Precision"] = float(precision_score(truth, pred, zero_division=0))
    result["F1"] = float(f1_score(truth, pred, zero_division=0))
    result["MCC"] = (
        float(matthews_corrcoef(truth, pred)) if both_classes else np.nan
    )
    result["ROC_AUC"] = np.nan
    return result


def save_validation_predictions(
    y_true: Sequence,
    y_score: Sequence,
    threshold: float,
    model_name: str,
    out_dir: str,
) -> str:
    """保存验证集逐条预测概率（保留连续得分，不只留 0/1 标签）。

    写入列：y_true, y_score, y_pred, threshold。
    y_score 为「该样本为欺诈」的概率 / 得分，供后续阈值选择与 PR 曲线复用。
    返回写入的 CSV 路径。
    """
    truth = _as_1d_float(y_true)
    score = _as_1d_float(y_score)
    if score.shape != truth.shape:
        raise ValueError("y_true 与 y_score 长度必须一致")
    _check_binary_labels(truth, "y_true")
    y_pred = (score >= threshold).astype(int)

    os.makedirs(out_dir, exist_ok=True)
    safe_name = model_name.strip().lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(out_dir, f"{safe_name}_valid_predictions.csv")
    pd.DataFrame(
        {
            "y_true": truth.astype(int),
            "y_score": score,
            "y_pred": y_pred,
            "threshold": float(threshold),
        }
    ).to_csv(path, index=False)
    return path


def metrics_to_dataframe(
    metrics_list: Iterable[Dict[str, Union[int, float, str, None]]],
) -> pd.DataFrame:
    """把多个 evaluate() 结果拼成对比表（列顺序与统一指标一致）。"""
    rows = list(metrics_list)
    if not rows:
        return pd.DataFrame()

    # 以第一条的键为基准，保证列顺序稳定
    order = list(rows[0].keys())
    df = pd.DataFrame(rows)[order]
    return df


# ---------------------------------------------------------------------------
# 自检：验证统一函数能复现既有 LR / RF 报告指标，并做 toy 用例抽查
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TABLES = os.path.join(PROJECT_ROOT, "results", "tables")

    # 1) toy 用例：手工可验证的混淆矩阵
    yt = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 4 正常 + 4 欺诈
    ys = np.array([0.1, 0.2, 0.6, 0.8, 0.3, 0.6, 0.9, 0.95])
    m = evaluate(yt, ys, threshold=0.5, model="toy")
    # 阈值 0.5 -> y_pred = [0,0,1,1,0,1,1,1]
    assert m["TP"] == 3 and m["FP"] == 2 and m["FN"] == 1 and m["TN"] == 2, m
    assert abs(m["Recall"] - 3 / 4) < 1e-9
    assert abs(m["Precision"] - 3 / 5) < 1e-9
    print("toy 用例通过：", {k: m[k] for k in ["TP", "FP", "FN", "TN", "Recall", "Precision"]})

    # 2) 用既有保存的验证集预测概率复现 LR / RF 指标
    checks = [
        ("logistic_regression_predictions.csv", "Logistic Regression",
         {"AP": 0.6275251631550335, "Recall": 0.8783783783783784,
          "Precision": 0.06701030927835051, "F1": 0.12452107279693486,
          "MCC": 0.23927730311190964, "FP": 905, "FN": 9}),
        ("random_forest_predictions.csv", "Random Forest",
         {"AP": 0.8174249814253471, "Recall": 0.6891891891891891,
          "Precision": 0.9807692307692307, "F1": 0.8095238095238095,
          "MCC": 0.8219130552951682, "FP": 1, "FN": 23}),
    ]
    rows = []
    for fname, name, expected in checks:
        df = pd.read_csv(os.path.join(TABLES, fname))
        got = evaluate(df["y_true"].values, df["y_score"].values,
                       threshold=df["threshold_used"].iloc[0], model=name)
        for k, v in expected.items():
            assert abs(got[k] - v) < 1e-9, (name, k, got[k], v)
        rows.append(got)
        print(f"{name} 复现通过：AP={got['AP']:.4f} Recall={got['Recall']:.4f} "
              f"Precision={got['Precision']:.4f} F1={got['F1']:.4f} "
              f"MCC={got['MCC']:.4f} FP={got['FP']} FN={got['FN']}")

    # 3) 输出统一对照表，作为 C·DAY2 的复现凭证
    out_csv = os.path.join(TABLES, "c2_unified_metrics_reproduced.csv")
    metrics_to_dataframe(rows).to_csv(out_csv, index=False)
    print("统一对照表已写入：", out_csv)
    print("✅ C·DAY2 自检全部通过")
