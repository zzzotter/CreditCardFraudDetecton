# -*- coding: utf-8 -*-
"""
src/feature_count_table.py
==========================
Day 5 交付：特征数量对照表（互信息 Top-10 / Top-15 / Top-20 与全特征基线）。

实验设计
--------
- 特征集：MI Top-10、Top-15、Top-20、全部 30 个（基线）；
- 模型：RandomForest（100 树、balanced_subsample）与 LightGBM（300 树、15 叶、
  min_child_samples=50、reg_lambda=1、balanced），均为固定超参、固定种子 42；
- 每换一个特征集都重新训练，只有特征集变化，其余条件完全一致；
- 在**验证集**上评价；测试集不读入；
- 两套阈值策略并列报告：
    (a) 固定 0.5；(b) 在验证集上使 F1 最大的阈值（会偏乐观，已在报告注明）。

说明
----
本表是 A 侧**预演版**。正式对照以 C 的统一评价函数为准（新文档 Day2 由 C 建立）；
两者若因阈值策略不同而有差异，以统一口径为准。

用法
----
    python src/feature_count_table.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_io  # noqa: E402  统一数据读取接口

# ============================================================
# 配置
# ============================================================
K_LIST = (10, 15, 20)
SEED = data_io.SEED
FS_DIR = data_io.FS_DIR

CSV_PATH = FS_DIR / "feature_count_comparison.csv"
MD_PATH = FS_DIR / "feature_count_comparison.md"
PNG_PATH = FS_DIR / "feature_count_comparison.png"

RF_PARAMS = dict(
    n_estimators=100,
    class_weight="balanced_subsample",
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=SEED,
    n_jobs=-1,
)
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=50,
    colsample_bytree=0.9,
    subsample=0.9,
    subsample_freq=1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
    deterministic=True,
    force_col_wise=True,
    verbose=-1,
)


# ============================================================
# 指标计算
# ============================================================
def compute_metrics(y_true, score, threshold: float) -> dict:
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 matthews_corrcoef)

    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "AP": float(average_precision_score(y_true, score)),
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "MCC": float(matthews_corrcoef(y_true, pred)),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "TN": int(tn),
        "threshold": float(threshold),
    }


def best_f1_threshold(y_true, score) -> float:
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_true, score)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    return float(thr[int(np.argmax(f1[:-1]))])


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    print("=" * 82)
    print("Day 5 · 特征数量对照表（验证集，测试集不参与）")
    print("=" * 82)

    try:
        X_tr, y_tr = data_io.load_split("train")
        X_va, y_va = data_io.load_split("valid")
    except Exception as exc:
        print(f"[ERROR] 数据读取失败：{type(exc).__name__}: {exc}")
        return 2

    print(f"训练集 {len(X_tr):,}（正例 {int(y_tr.sum())}）；"
          f"验证集 {len(X_va):,}（正例 {int(y_va.sum())}）")

    feature_sets: list[tuple[str, list[str]]] = []
    for k in K_LIST:
        try:
            feature_sets.append((f"MI Top-{k}", data_io.load_topk("mi", k)))
        except Exception as exc:
            print(f"[ERROR] 读取 Top-{k} 失败：{exc}")
            return 2
    feature_sets.append(("All-30", list(data_io.FEATURE_ORDER)))

    from sklearn.ensemble import RandomForestClassifier
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("[ERROR] 未安装 lightgbm：pip install -U lightgbm")
        return 2

    models = {
        "RandomForest": (RandomForestClassifier(**RF_PARAMS), None),
        "LightGBM": (LGBMClassifier(**LGBM_PARAMS), None),
    }

    rows, t0 = [], time.time()
    for set_name, feats in feature_sets:
        cols = data_io.canonical_columns(feats)      # 强制规范列序
        for model_name, (model, _) in models.items():
            model.fit(X_tr[cols], y_tr)
            score = model.predict_proba(X_va[cols])[:, 1]
            tau = best_f1_threshold(y_va, score)
            for rule, thr in (("fixed_0.5", 0.5), ("valid_best_f1", tau)):
                met = compute_metrics(y_va, score, thr)
                rows.append({
                    "feature_set": set_name,
                    "k": len(feats),
                    "model": model_name,
                    "threshold_rule": rule,
                    **met,
                })
            print(f"  {set_name:<9} {model_name:<13} "
                  f"AP={rows[-1]['AP']:.4f}  F1@0.5={rows[-2]['F1']:.4f}  "
                  f"F1@{tau:.3f}={rows[-1]['F1']:.4f}  MCC@{tau:.3f}={rows[-1]['MCC']:.4f}")

    elapsed = time.time() - t0
    res = pd.DataFrame(rows)

    order = {"fixed_0.5": 0, "valid_best_f1": 1}
    res = res.sort_values(["model", "threshold_rule", "k"],
                          key=lambda s: s.map(order) if s.name == "threshold_rule" else s)
    res.to_csv(CSV_PATH, index=False, float_format="%.6f")

    # ---------- Markdown ----------
    cols_show = ["feature_set", "k", "model", "AP", "Recall", "Precision",
                 "F1", "MCC", "FP", "FN"]
    lines = [
        "# 特征数量对照表（验证集）",
        "",
        f"生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}　"
        f"训练集 {len(X_tr):,} 行 / 验证集 {len(X_va):,} 行（测试集未参与）",
        "",
        "特征集为嵌套关系（MI Top-10 ⊂ Top-15 ⊂ Top-20），All-30 为不筛选基线。",
        "RandomForest：100 树、`balanced_subsample`、`max_features=sqrt`；"
        "LightGBM：300 树、15 叶、`min_child_samples=50`、`reg_lambda=1`、`balanced`。"
        f"随机种子 {SEED}，各特征集独立重新训练。",
        "",
        "## 1 阈值固定为 0.5",
        "",
        "| " + " | ".join(cols_show) + " |",
        "|" + "|".join(["---"] * len(cols_show)) + "|",
    ]
    for _, r in res[res["threshold_rule"] == "fixed_0.5"][cols_show].iterrows():
        lines.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols_show) + " |")

    lines += ["", "## 2 阈值取验证集上使 F1 最大的值", "",
              "| " + " | ".join(cols_show + ["threshold"]) + " |",
              "|" + "|".join(["---"] * (len(cols_show) + 1)) + "|"]
    for _, r in res[res["threshold_rule"] == "valid_best_f1"].iterrows():
        lines.append("| " + " | ".join(
            [f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols_show]
            + [f"{r['threshold']:.3f}"]) + " |")

    lines += [
        "",
        "> 说明：第 2 表的阈值与特征数均在验证集上挑选，指标偏乐观，仅用于比较相对差异；",
        "> 最终性能须由 C 在独立测试集上评价一次。本表为 A 侧预演，正式对照以 C 的统一评价函数为准。",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- 出图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        style = {"RandomForest": ("#2c7fb8", "o"), "LightGBM": ("#dd8047", "s")}
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        for ax, rule, title in (
                (axes[0], "fixed_0.5", "AP (threshold-free)"),
                (axes[1], "valid_best_f1", "F1 at validation-optimal threshold")):
            for model_name, (color, marker) in style.items():
                d = res[(res["model"] == model_name) & (res["threshold_rule"] == rule)].sort_values("k")
                y = d["AP"] if rule == "fixed_0.5" else d["F1"]
                ax.plot(d["k"], y, marker=marker, color=color, linewidth=2, label=model_name)
            ax.set_xlabel("Number of features (k; 30 = all features)")
            ax.set_ylabel("AP" if rule == "fixed_0.5" else "F1")
            ax.set_title(title)
            ax.set_xticks([10, 15, 20, 30])
            ax.grid(alpha=0.3, linestyle=":")
            ax.legend(fontsize=9)
        fig.suptitle("Feature count vs validation performance (test set untouched)", fontsize=11)
        fig.tight_layout()
        fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] 出图失败（不影响表格）：{type(exc).__name__}: {exc}")

    print(f"\n总耗时 {elapsed:.1f} 秒")
    print("\n--- 阈值固定 0.5 ---")
    print(res[res["threshold_rule"] == "fixed_0.5"][cols_show].to_string(index=False))
    print("\n--- 验证集最优阈值 ---")
    print(res[res["threshold_rule"] == "valid_best_f1"][cols_show + ["threshold"]].to_string(index=False))

    print("\n已写出：")
    for p in (CSV_PATH, MD_PATH, PNG_PATH):
        print(f"  {os.path.relpath(p, data_io.PROJECT_ROOT)}")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
