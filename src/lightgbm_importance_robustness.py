# -*- coding: utf-8 -*-
"""
src/lightgbm_importance_robustness.py
=====================================
信用卡欺诈检测 · 第二阶段 · A · Day 2 补充：LightGBM 重要性的稳健性检查。

要回答的问题
------------
Day 2 的 LightGBM 基线 train AP ≈ 1.0000（严重过拟合），gain 高度集中在 V14（66.9%），
第 5–15 名挤在 0.6%–1.3% 之间。那么：**这些名次在换超参、换随机种子之后还站得住吗？**

做法
----
5 组预设配置 × 5 个固定种子 = 25 次独立训练，每次记录每个特征的 gain share 与名次，然后统计：
1. 每个特征的平均 share、标准差、平均名次、最好/最差名次；
2. 该特征进入 Top-10 的次数（25 次里有几次）；
3. 25 次训练之间的平均名次相关系数与 Top-10 集合的平均 Jaccard；
4. 与 Day 2 基线结果的一致性校验（config C0 + seed 42 应复现 66.88%）。

口径
----
只用真实训练集；不使用早停（早停需要验证集标签，会造成特征选择层面的泄漏）；
所有配置均为预设，不看验证集结果挑选，因此不存在“用验证集调特征选择”的问题。

用法
----
    python src/lightgbm_importance_robustness.py

输出
----
    results/feature_selection/lightgbm_importance_robustness.csv
    results/feature_selection/lightgbm_importance_robustness.json
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
# 0. 全局配置
# ============================================================
SEEDS = (42, 43, 44, 45, 46)
K_LIST = (10, 15, 20)
CONSENSUS_THRESHOLD = 0.8          # 进入 Top-k 的频率 ≥ 80% 才视为“共识特征”

BASE_PARAMS = dict(
    learning_rate=0.05,
    subsample=0.9,
    subsample_freq=1,
    class_weight="balanced",
    n_jobs=-1,
    deterministic=True,
    force_col_wise=True,
    verbose=-1,
)

CONFIGS = [
    {"name": "C0_baseline",
     "params": dict(n_estimators=500, num_leaves=31, min_child_samples=20,
                    colsample_bytree=0.9, reg_lambda=0.0)},
    {"name": "C1_regularized",
     "params": dict(n_estimators=300, num_leaves=15, min_child_samples=50,
                    colsample_bytree=0.9, reg_lambda=1.0)},
    {"name": "C2_shallow",
     "params": dict(n_estimators=200, num_leaves=7, min_child_samples=100,
                    colsample_bytree=0.9, reg_lambda=5.0)},
    {"name": "C3_no_colsample",
     "params": dict(n_estimators=300, num_leaves=15, min_child_samples=50,
                    colsample_bytree=1.0, reg_lambda=1.0, subsample=0.8)},
    {"name": "C4_unweighted",
     "params": dict(n_estimators=300, num_leaves=15, min_child_samples=50,
                    colsample_bytree=0.9, reg_lambda=1.0, class_weight=None)},
]

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL_COL = "Class"

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "feature_selection")

CSV_PATH = os.path.join(OUT_DIR, "lightgbm_importance_robustness.csv")
JSON_PATH = os.path.join(OUT_DIR, "lightgbm_importance_robustness.json")
DAY2_CSV = os.path.join(OUT_DIR, "lightgbm_importance.csv")


# ============================================================
# 1. 读数据（只读训练集）
# ============================================================
def load_train_only():
    x_path = os.path.join(DATA_DIR, "X_train.csv")
    y_path = os.path.join(DATA_DIR, "y_train.csv")
    for p in (x_path, y_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少文件：{p}")
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)[LABEL_COL].astype(int)
    if list(X.columns) != FEATURE_ORDER:
        raise ValueError("特征列与冻结口径不一致。")
    if len(X) != len(y):
        raise ValueError("X 与 y 行数不一致。")
    return X, y


# ============================================================
# 2. 单次训练 → gain share
# ============================================================
def fit_gain_share(cfg, seed, X, y) -> np.ndarray:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("未安装 lightgbm，请先执行：pip install -U lightgbm") from exc

    params = dict(BASE_PARAMS)
    params.update(cfg["params"])
    model = LGBMClassifier(random_state=seed, **params)
    model.fit(X, y)

    booster = model.booster_
    names = list(booster.feature_name())
    if names != FEATURE_ORDER:
        raise ValueError(f"booster 特征名与冻结列序不一致：{names}")
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    total = gain.sum()
    if total <= 0:
        raise ValueError("gain 全为 0，训练异常。")
    return gain / total


def to_ranks(share_row: np.ndarray) -> np.ndarray:
    """按 share 降序给名次；share 相同时按冻结列序细分（与 Day 2 口径一致）。"""
    order = np.lexsort((np.arange(len(share_row)), -share_row))
    ranks = np.empty(len(share_row), dtype=int)
    ranks[order] = np.arange(1, len(share_row) + 1)
    return ranks


# ============================================================
# 3. 主流程
# ============================================================
def main() -> int:
    print("=" * 88)
    print("Day 2 补充：LightGBM 重要性稳健性检查（5 配置 × 5 种子 = 25 次训练）")
    print("=" * 88)

    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        X, y = load_train_only()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2

    n_runs = len(CONFIGS) * len(SEEDS)
    share_mat = np.zeros((n_runs, len(FEATURE_ORDER)), dtype=float)
    rank_mat = np.zeros((n_runs, len(FEATURE_ORDER)), dtype=int)
    run_meta = []

    t_start = time.time()
    i = 0
    for cfg in CONFIGS:
        for seed in SEEDS:
            try:
                share = fit_gain_share(cfg, seed, X, y)
            except Exception as exc:
                print(f"[ERROR] {cfg['name']} / seed={seed} 训练失败：{type(exc).__name__}: {exc}")
                return 2
            share_mat[i] = share
            rank_mat[i] = to_ranks(share)
            run_meta.append({"config": cfg["name"], "seed": seed,
                             "top10": [FEATURE_ORDER[j] for j in np.argsort(rank_mat[i])[:10]]})
            i += 1
            print(f"  [{i:>2}/{n_runs}] {cfg['name']:<16} seed={seed}  完成")
    elapsed = time.time() - t_start
    print(f"\n全部训练完成，用时 {elapsed:.1f} 秒")

    # ---------- 聚合统计 ----------
    table = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "mean_share": share_mat.mean(axis=0),
        "std_share": share_mat.std(axis=0, ddof=0),
        "min_share": share_mat.min(axis=0),
        "max_share": share_mat.max(axis=0),
        "mean_rank": rank_mat.mean(axis=0),
        "best_rank": rank_mat.min(axis=0),
        "worst_rank": rank_mat.max(axis=0),
    })
    for k in K_LIST:
        table[f"top{k}_freq"] = (rank_mat <= k).sum(axis=0)
    table["top_k_freq_ratio"] = table["top10_freq"] / n_runs
    table = table.sort_values(["mean_rank", "feature"]).reset_index(drop=True)
    table["rank_by_mean"] = np.arange(1, len(table) + 1)

    # ---------- 稳定性指标 ----------
    R = rank_mat.astype(float)
    Rc = R - R.mean(axis=1, keepdims=True)
    corr = np.corrcoef(Rc)
    mean_rank_corr = float(corr[np.triu_indices(n_runs, 1)].mean())

    top10_sets = [set(FEATURE_ORDER[j] for j in np.argsort(rank_mat[r])[:10]) for r in range(n_runs)]
    jac = []
    for a in range(n_runs):
        for b in range(a + 1, n_runs):
            inter = len(top10_sets[a] & top10_sets[b])
            union = len(top10_sets[a] | top10_sets[b])
            jac.append(inter / union if union else 0.0)
    mean_jaccard = float(np.mean(jac))

    def consensus(k: int):
        need = int(np.ceil(CONSENSUS_THRESHOLD * n_runs))
        sel = table[table[f"top{k}_freq"] >= need]["feature"].tolist()
        return sel, need

    cons10, need10 = consensus(10)
    cons15, need15 = consensus(15)
    cons20, need20 = consensus(20)

    # ---------- 与 Day 2 基线一致性校验（按特征名逐一对齐，不用位置索引） ----------
    baseline_check = {"performed": False}
    day2_top10 = None
    if os.path.exists(DAY2_CSV):
        day2 = pd.read_csv(DAY2_CSV)
        day2_share = {str(f): float(s) for f, s in zip(day2["feature"], day2["gain_share"])}
        day2_top10 = day2.sort_values("rank")["feature"].head(10).tolist()

        run00 = next(i for i, m in enumerate(run_meta)
                     if m["config"] == "C0_baseline" and m["seed"] == 42)
        share00 = {f: float(share_mat[run00][j]) for j, f in enumerate(FEATURE_ORDER)}
        diff = max(abs(share00[f] - day2_share[f]) for f in FEATURE_ORDER)

        baseline_check = {
            "performed": True,
            "day2_top10": day2_top10,
            "c0_seed42_top10": run_meta[run00]["top10"],
            "top10_identical": day2_top10 == run_meta[run00]["top10"],
            "max_abs_share_diff": diff,
            "reproduced": bool(diff < 1e-6),
        }

    # ---------- 落盘 ----------
    table.to_csv(CSV_PATH, index=False, float_format="%.10f")

    card = {
        "method": "lightgbm_importance_robustness",
        "n_runs": n_runs,
        "configs": [{"name": c["name"], "params": {**BASE_PARAMS, **c["params"]}} for c in CONFIGS],
        "seeds": list(SEEDS),
        "input": {"X": "data/processed/X_train.csv", "y": "data/processed/y_train.csv",
                  "n_samples": int(len(X)), "n_positive": int(y.sum())},
        "stability": {
            "mean_pairwise_rank_correlation": mean_rank_corr,
            "mean_pairwise_top10_jaccard": mean_jaccard,
            "elapsed_seconds": round(elapsed, 1),
        },
        "consensus": {
            "rule": f"进入 Top-k 的次数 >= ceil({CONSENSUS_THRESHOLD} * {n_runs}) = {need10}",
            "top10": cons10,
            f"top15": cons15,
            "top20": cons20,
        },
        "baseline_vs_day2": baseline_check,
        "per_run_top10": run_meta,
        "aggregate": table.to_dict(orient="records"),
        "script": "src/lightgbm_importance_robustness.py",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "notes": [
            "只用训练集；不使用早停，避免通过验证集影响特征排序。",
            "5 组配置均为预设，未依据任何验证结果挑选。",
            "平均名次相关性与 Top-10 Jaccard 衡量排名的稳定性，越接近 1 越稳。",
        ],
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    # ---------- 打印 ----------
    print("\n--- 排名稳定性总览（按平均名次排序，展示前 20）---")
    print(f"{'rank':>4} {'feature':<7} {'mean_share':>11} {'±std':>9} "
          f"{'mean_rank':>9} {'rank_range':>10} {'Top10次数':>9}")
    for _, row in table.head(20).iterrows():
        print(f"{int(row['rank_by_mean']):>4} {row['feature']:<7} {row['mean_share']:>11.4%} "
              f"{row['std_share']:>9.4%} {row['mean_rank']:>9.2f} "
              f"{int(row['best_rank']):>4}-{int(row['worst_rank']):<5} "
              f"{int(row['top10_freq']):>5}/{n_runs}")

    print("\n--- 稳定性指标 ---")
    print(f"25 次训练之间的平均名次相关系数：{mean_rank_corr:.3f}（1.0 = 完全一致）")
    print(f"25 次训练之间 Top-10 集合的平均 Jaccard：{mean_jaccard:.3f}")

    print(f"\n--- 共识特征（进入 Top-k 至少 {need10}/{n_runs} 次）---")
    print(f"共识 Top-10（{len(cons10)} 个）：{cons10}")
    print(f"共识 Top-15（{len(cons15)} 个）：{cons15}")
    print(f"共识 Top-20（{len(cons20)} 个）：{cons20}")

    if baseline_check["performed"]:
        print("\n--- 与 Day 2 基线的一致性校验 ---")
        print(f"Day 2 Top-10 : {baseline_check['day2_top10']}")
        print(f"本次 C0/42   : {baseline_check['c0_seed42_top10']}")
        print(f"Top-10 是否相同：{baseline_check['top10_identical']}；"
              f"最大 share 偏差：{baseline_check['max_abs_share_diff']:.2e}"
              f"（{'复现成功' if baseline_check['reproduced'] else '未完全复现，注意线程数/版本差异'}）")

    print("\n已写出：")
    print(f"  {os.path.relpath(CSV_PATH, PROJECT_ROOT)}")
    print(f"  {os.path.relpath(JSON_PATH, PROJECT_ROOT)}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
