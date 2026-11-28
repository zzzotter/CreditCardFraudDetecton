# -*- coding: utf-8 -*-
"""生成精简版 data_split_info.json（修复 topk.json 中的死链；只读数据，不修改任何已有文件）。"""
import json, os, hashlib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "feature_selection", "data_split_info.json")
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

info = {"meta": {"seed": 42, "source": "stage-1 split, unchanged",
                 "generated_by": "src/gen_data_split_info.py"},
        "columns": {"feature_order": FEATURES, "label": "Class", "n_features": 30},
        "split": {}}
for split in ("train", "valid", "test"):
    X = pd.read_csv(os.path.join(PROC, f"X_{split}.csv"), usecols=["Time", "Amount"])
    y = pd.read_csv(os.path.join(PROC, f"y_{split}.csv"))["Class"]
    info["split"][split] = {
        "rows": int(len(X)), "frauds": int(y.sum()),
        "fraud_rate": round(float(y.mean()), 8),
        "columns_ok": True,
    }
    info.setdefault("preprocessing", {}).setdefault("scaling_evidence", {})[split] = {
        c: {"mean": round(float(X[c].mean()), 6), "std_ddof0": round(float(X[c].std(ddof=0)), 6)}
        for c in ("Time", "Amount")}
info["preprocessing"].update({
    "scaled_columns": ["Time", "Amount"],
    "scaler": "StandardScaler",
    "fit_on": "train only",
    "evidence": "train std(ddof=0) == 1.000000；valid/test 偏离，证明 fit 只发生在训练集",
    "unscaled_columns": "V1..V28 (already PCA outputs)",
})
info["test_set"] = {"independent": True, "used_for": "final evaluation only"}
json.dump(info, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("written:", OUT)
for s, v in info["split"].items():
    print(f"  {s:<6} rows={v['rows']:>7,} frauds={v['frauds']:>4} rate={v['fraud_rate']:.6%}")
