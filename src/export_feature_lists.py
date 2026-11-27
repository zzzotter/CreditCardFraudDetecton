# -*- coding: utf-8 -*-
"""
src/export_feature_lists.py
===========================
Day 4 交付：从互信息排名整理 Top 10 / Top 15 / Top 20 三组候选特征列表。

做什么
------
1. 读取已有结果 results/feature_selection/mutual_information.csv（不重算互信息）；
2. 校验：30 行、名次 1..30 唯一、MI 严格降序、特征名与冻结口径一致；
3. 生成三组候选列表（.txt，一行一个特征）、合并表（.csv）与报告用表格（.md）；
4. 与 results/feature_selection/mutual_information_topk.json 交叉校验，确保两处名单一致。

不做什么
--------
不重新计算互信息、不修改已有任何文件、不触碰验证集与测试集。

用法
----
    python src/export_feature_lists.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

# ============================================================
# 配置
# ============================================================
K_LIST = (10, 15, 20)
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
FS_DIR = os.path.join(PROJECT_ROOT, "results", "feature_selection")

MI_CSV = os.path.join(FS_DIR, "mutual_information.csv")
MI_JSON = os.path.join(FS_DIR, "mutual_information_topk.json")


# ============================================================
# 读取与校验
# ============================================================
def load_mi_table() -> pd.DataFrame:
    if not os.path.exists(MI_CSV):
        raise FileNotFoundError(f"缺少互信息结果：{MI_CSV}（请先运行 src/mutual_information.py）")

    df = pd.read_csv(MI_CSV)
    need = {"rank", "feature", "mi_score"}
    if not need.issubset(df.columns):
        raise ValueError(f"互信息结果缺少列：{sorted(need - set(df.columns))}")

    df = df.sort_values("rank").reset_index(drop=True)

    if len(df) != len(FEATURE_ORDER):
        raise ValueError(f"特征数量异常：期望 {len(FEATURE_ORDER)}，实际 {len(df)}")
    if sorted(df["feature"].tolist()) != sorted(FEATURE_ORDER):
        raise ValueError("特征名与冻结口径不一致。")
    if df["rank"].tolist() != list(range(1, len(df) + 1)):
        raise ValueError("名次不是 1..30 的连续整数。")
    if df["mi_score"].isna().any():
        raise ValueError("mi_score 存在缺失值。")
    if not (df["mi_score"].diff().dropna() <= 0).all():
        raise ValueError("mi_score 不是降序排列，排名可能有问题。")

    return df


def cross_check(df: pd.DataFrame) -> str:
    """与 mutual_information_topk.json 交叉校验；返回一句话结论。"""
    if not os.path.exists(MI_JSON):
        return "未找到 mutual_information_topk.json，跳过交叉校验。"

    card = json.loads(open(MI_JSON, encoding="utf-8").read())
    top_k = card.get("top_k", {})
    mismatch = []
    for k in K_LIST:
        saved = top_k.get(str(k)) or top_k.get(f"top{k}")
        if saved is None:
            mismatch.append(f"Top-{k} 在 json 中缺失")
            continue
        current = df["feature"].head(k).tolist()
        if list(saved) != current:
            mismatch.append(f"Top-{k} 不一致：json={saved} / csv={current}")

    if mismatch:
        raise ValueError("交叉校验失败：\n  " + "\n  ".join(mismatch))
    return f"与 mutual_information_topk.json 交叉校验通过（Top {list(K_LIST)} 完全一致）"


# ============================================================
# 写出
# ============================================================
def write_txt(df: pd.DataFrame) -> list[str]:
    written = []
    for k in K_LIST:
        path = os.path.join(FS_DIR, f"mi_top{k}.txt")
        feats = df["feature"].head(k).tolist()
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(feats) + "\n")
        written.append(path)
    return written


def write_combined(df: pd.DataFrame) -> tuple[str, str]:
    rows = []
    for k in K_LIST:
        for order, (_, r) in enumerate(df.head(k).iterrows(), start=1):
            rows.append({
                "k": k,
                "order_in_list": order,
                "feature": r["feature"],
                "rank_in_full": int(r["rank"]),
                "mi_score": float(r["mi_score"]),
            })
    out = pd.DataFrame(rows)

    csv_path = os.path.join(FS_DIR, "mi_candidate_features.csv")
    out.to_csv(csv_path, index=False, float_format="%.10f")

    md_path = os.path.join(FS_DIR, "mi_candidate_features.md")
    lines = [
        "# 互信息候选特征列表（Top 10 / 15 / 20）",
        "",
        "来源：`results/feature_selection/mutual_information.csv`（训练集 199,364 行，344 正例，"
        "`mutual_info_classif`, `n_neighbors=3`, `random_state=42`）。",
        "三组为嵌套关系：Top-10 ⊂ Top-15 ⊂ Top-20，均为按互信息降序取前 k 个。",
        "",
        "| k | 序号 | 特征 | 全局名次 | MI 值 |",
        "|---:|---:|:---|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['k']} | {r['order_in_list']} | {r['feature']} | "
                     f"{r['rank_in_full']} | {r['mi_score']:.6f} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return csv_path, md_path


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    print("=" * 78)
    print("Day 4 · 整理互信息 Top 10 / 15 / 20 候选特征列表")
    print("=" * 78)

    os.makedirs(FS_DIR, exist_ok=True)
    try:
        df = load_mi_table()
        check_msg = cross_check(df)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2

    print(f"读取 {len(df)} 个特征的互信息排名，校验通过（名次连续、MI 降序）。")

    print("\n--- 三组候选特征 ---")
    prev = set()
    for k in K_LIST:
        feats = df["feature"].head(k).tolist()
        added = [f for f in feats if f not in prev]
        print(f"\nTop-{k}（{k} 个）: {', '.join(feats)}")
        if prev:
            print(f"  相较上一组新增 {len(added)} 个: {', '.join(added)}")
        prev = set(feats)

    print(f"\n交叉校验：{check_msg}")

    try:
        txts = write_txt(df)
        csv_path, md_path = write_combined(df)
    except Exception as exc:
        print(f"[ERROR] 写出失败：{type(exc).__name__}: {exc}")
        return 2

    print("\n已写出：")
    for p in txts:
        print(f"  {os.path.relpath(p, PROJECT_ROOT)}")
    print(f"  {os.path.relpath(csv_path, PROJECT_ROOT)}")
    print(f"  {os.path.relpath(md_path, PROJECT_ROOT)}")

    print("\n给 C 的两种读法（等价）：")
    print('  import sys; sys.path.insert(0, "src"); import data_io')
    print('  feats = data_io.load_topk("mi", 10)            # 读 json，含校验')
    print('  feats = open("results/feature_selection/mi_top10.txt").read().split()')
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
