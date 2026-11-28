# -*- coding: utf-8 -*-
"""
src/data_io.py
==============
第二阶段统一数据读取接口（A 交付给 B、C 使用）。

设计原则
--------
1. **只读**：本模块不划分数据、不缩放、不去重、不写任何文件（除命令行打印）。
2. **列序唯一**：所有读取结果一律按 FEATURE_ORDER 排列。任何按名字取子集的操作，
   返回的列序也是规范的——避免"同一特征集、不同列序"导致树模型列采样差异
   （LightGBM 的 colsample_bytree 依赖列位置，列序一变结果会微变）。
3. **读取即校验**：行数、列名、列序、缺失值、标签取值在一次读取中全部检查，
   不符合预期立刻报错，而不是等训练出问题再排查。
4. **测试集默认不读**：`load_split("test")` 会打印提示，需显式 `allow_test=True`；
   命令行自检默认跳过测试集内容（只看文件是否存在）。

数据来源（阶段一已生成，本模块不修改）
--------------------------------------
    data/processed/X_train.csv / y_train.csv    199,364 行 / 344 正例
    data/processed/X_valid.csv / y_valid.csv     42,721 行 /  74 正例
    data/processed/X_test.csv  / y_test.csv      42,722 行 /  74 正例

    特征 30 个，顺序固定为 Time, V1..V28, Amount；标签列名 Class（0/1）。
    Time、Amount 已用 StandardScaler 在训练集上拟合后标准化
    （训练集上 std = 1.000000，验证/测试集偏离，可证明 fit 只发生在训练集），
    V1..V28 未做二次缩放。**严禁对全量数据重新 fit 缩放**。

用法
----
    import sys
    sys.path.insert(0, "src")
    import data_io

    X_train, y_train = data_io.load_split("train")
    X_valid, y_valid = data_io.load_split("valid")

    # 只取前 10 个互信息特征（返回的列序仍是规范顺序）
    feats = data_io.load_topk("mi", 10)
    X_tr_sub, y_tr_sub = data_io.load_split("train", columns=feats)

    # 可用档位查询与数据版本指纹
    print(data_io.available_topk("mi"))      # [10, 15, 20]
    print(data_io.data_fingerprint())

命令行自检
----------
    python src/data_io.py              # 校验 train/valid，测试集只检查文件存在
    python src/data_io.py --with-test  # 连测试集内容一起校验（仍不训练、不写文件）
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "FEATURE_ORDER", "LABEL_COL", "SEED", "SPLITS", "EXPECTED_ROWS",
    "EXPECTED_FRAUDS", "PROCESSED_DIR", "FS_DIR", "PROJECT_ROOT",
    "DataInterfaceError",
    "load_split", "load_x", "load_y", "canonical_columns",
    "load_topk", "available_topk", "data_fingerprint", "validate_split",
]

# ============================================================
# 常量（与阶段一冻结口径一致，不要修改）
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
FS_DIR: Path = PROJECT_ROOT / "results" / "feature_selection"

FEATURE_ORDER: tuple[str, ...] = (
    ("Time",) + tuple(f"V{i}" for i in range(1, 29)) + ("Amount",)
)
LABEL_COL: str = "Class"
SEED: int = 42
SPLITS: tuple[str, ...] = ("train", "valid", "test")

# 阶段一划分的实测值（行数、欺诈样本数），用于读取时校验
EXPECTED_ROWS: dict[str, int] = {"train": 199_364, "valid": 42_721, "test": 42_722}
EXPECTED_FRAUDS: dict[str, int] = {"train": 344, "valid": 74, "test": 74}

# 各方法特征列表文件（results/feature_selection/ 下）
TOP_K_FILES: dict[str, str] = {
    "mi": "mutual_information_topk.json",
    "lightgbm": "lightgbm_importance_topk.json",
    "fidsom": "fidsom_importance_topk.json",
}


class DataInterfaceError(RuntimeError):
    """数据文件缺失、列序不符、行数异常等问题的统一异常。"""


# ============================================================
# 内部工具
# ============================================================
def _x_path(split: str) -> Path:
    return PROCESSED_DIR / f"X_{split}.csv"


def _y_path(split: str) -> Path:
    return PROCESSED_DIR / f"y_{split}.csv"


def _check_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，收到：{split!r}")
    return split


def _read_header(path: Path) -> list[str]:
    if not path.exists():
        raise DataInterfaceError(f"缺少数据文件：{path}")
    try:
        return pd.read_csv(path, nrows=0).columns.tolist()
    except Exception as exc:  # 文件损坏 / 编码问题
        raise DataInterfaceError(f"无法读取表头：{path}（{exc}）") from exc


# ============================================================
# 列序
# ============================================================
def canonical_columns(columns: Iterable[str] | None = None) -> list[str]:
    """把任意特征名集合整理成规范列序；含未知特征名时直接报错。"""
    if columns is None:
        return list(FEATURE_ORDER)
    requested = list(dict.fromkeys(columns))          # 去重且保序
    unknown = [c for c in requested if c not in FEATURE_ORDER]
    if unknown:
        raise DataInterfaceError(
            f"存在不属于冻结特征集的名字：{unknown}\n可用特征：{list(FEATURE_ORDER)}")
    if not requested:
        raise DataInterfaceError("请求的特征列表为空。")
    keep = set(requested)
    return [c for c in FEATURE_ORDER if c in keep]     # 强制规范顺序


# ============================================================
# 读取
# ============================================================
def load_split(
    split: str,
    columns: Sequence[str] | None = None,
    with_y: bool = True,
    validate: bool = True,
    allow_test: bool = False,
    quiet: bool = False,
):
    """读取一个划分。

    参数
    ----
    split     : "train" | "valid" | "test"
    columns   : 需要的特征子集（按名字），返回结果自动按规范列序排列
    with_y    : 是否同时返回标签
    validate  : 是否执行完整性校验（默认开启）
    allow_test: 读取测试集时必须显式置 True（项目约定：测试集只用于最终评价）
    quiet     : 关闭提示信息

    返回
    ----
    (X, y) 或 X
    """
    split = _check_split(split)
    if split == "test" and not allow_test and not quiet:
        print("[data_io][WARN] 正在读取测试集。按项目约定，"
              "测试集只用于最终方案的一次性评价，不得参与特征/阈值/模型选择。")

    x_path, y_path = _x_path(split), _y_path(split)
    header = _read_header(x_path)

    if columns is None:
        if validate and header != list(FEATURE_ORDER):
            raise DataInterfaceError(
                f"X_{split}.csv 的列与冻结口径不一致。\n"
                f"  期望：{list(FEATURE_ORDER)}\n  实际：{header}")
        cols = list(FEATURE_ORDER)
    else:
        cols = canonical_columns(columns)
        missing = [c for c in cols if c not in header]
        if missing:
            raise DataInterfaceError(f"X_{split}.csv 中不存在这些列：{missing}")

    X = pd.read_csv(x_path, usecols=cols).loc[:, cols]

    if validate:
        _validate_x(X, split)

    if not with_y:
        return X

    if not y_path.exists():
        raise DataInterfaceError(f"缺少标签文件：{y_path}")
    y_df = pd.read_csv(y_path)
    if LABEL_COL not in y_df.columns:
        raise DataInterfaceError(
            f"y_{split}.csv 中缺少标签列 {LABEL_COL!r}，实际列：{y_df.columns.tolist()}")
    y = y_df[LABEL_COL]

    if validate:
        _validate_y(y, split, n_x=len(X))
    return X, y


def load_x(split: str, columns: Sequence[str] | None = None, **kwargs):
    """只读特征矩阵。"""
    kwargs["with_y"] = False
    return load_split(split, columns=columns, **kwargs)


def load_y(split: str, validate: bool = True, allow_test: bool = False,
           quiet: bool = False):
    """只读标签（不与 X 一起读时使用）。"""
    split = _check_split(split)
    if split == "test" and not allow_test and not quiet:
        print("[data_io][WARN] 正在读取测试集标签。")
    y_path = _y_path(split)
    if not y_path.exists():
        raise DataInterfaceError(f"缺少标签文件：{y_path}")
    y = pd.read_csv(y_path)[LABEL_COL]
    if validate:
        _validate_y(y, split, n_x=EXPECTED_ROWS[split])
    return y


# ============================================================
# 校验
# ============================================================
def _validate_x(X: pd.DataFrame, split: str) -> None:
    n_expected = EXPECTED_ROWS.get(split)
    if n_expected is not None and len(X) != n_expected:
        raise DataInterfaceError(
            f"X_{split}.csv 行数异常：期望 {n_expected:,}，实际 {len(X):,}。"
            "（是否被重新划分或覆盖？）")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise DataInterfaceError(f"X_{split}.csv 存在缺失值，涉及列：{bad}")
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise DataInterfaceError(f"X_{split}.csv 存在非数值列：{non_numeric}")


def _validate_y(y: pd.Series, split: str, n_x: int | None = None) -> None:
    if n_x is not None and len(y) != n_x:
        raise DataInterfaceError(
            f"y_{split}.csv 与 X 行数不一致：X {n_x:,} 行，y {len(y):,} 行。")
    if y.isna().any():
        raise DataInterfaceError(f"y_{split}.csv 存在缺失标签。")
    values = set(pd.unique(y))
    if not values.issubset({0, 1}):
        raise DataInterfaceError(
            f"y_{split}.csv 标签取值异常：{sorted(values)}（应为 0/1）。")
    n_pos = int((y == 1).sum())
    n_expected = EXPECTED_FRAUDS.get(split)
    if n_expected is not None and n_pos != n_expected:
        raise DataInterfaceError(
            f"y_{split}.csv 欺诈样本数异常：期望 {n_expected}，实际 {n_pos}。")


def validate_split(split: str) -> dict:
    """完整校验一个划分，返回统计字典（供自检与报表使用）。"""
    X, y = load_split(split, validate=True, allow_test=True, quiet=True)
    return {
        "split": split,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_fraud": int((y == 1).sum()),
        "fraud_rate": float((y == 1).mean()),
        "n_missing": int(X.isna().sum().sum()),
        "columns_ok": list(X.columns) == list(FEATURE_ORDER),
    }


# ============================================================
# 特征列表（A 的交付物，供 B/C 直接调用）
# ============================================================
def _topk_map(method: str) -> dict[int, list[str]]:
    """把 topk.json 的 top_k 统一成 {int: [feature, ...]}。

    各方法保存的键格式不一致：互信息与 LightGBM 用 "10"、"15"、"20"，
    FID-SOM 用 "top10"、"top15"、"top20"。这里统一归一化为整数键，
    无法解析的键直接跳过。
    """
    path = FS_DIR / TOP_K_FILES.get(method, "")
    if not path.exists():
        raise DataInterfaceError(
            f"找不到 {method} 的特征列表文件：{path}\n"
            f"可用方法：{list(TOP_K_FILES)}（需先运行对应脚本生成 topk.json）")
    card = json.loads(path.read_text(encoding="utf-8"))
    raw = card.get("top_k", {})
    out: dict[int, list[str]] = {}
    for key, feats in raw.items():
        k_str = str(key).strip().lower()
        if k_str.startswith("top"):
            k_str = k_str[3:]
        try:
            k = int(k_str)
        except ValueError:
            continue
        if isinstance(feats, list) and feats:
            out[k] = [str(f) for f in feats]
    if not out:
        raise DataInterfaceError(f"{path.name} 中未找到可解析的 top_k 字段。")
    return out


def available_topk(method: str = "mi") -> list[int]:
    """列出某个方法已保存的 Top-k 档位（自动兼容 "10" 与 "top10" 两种键）。"""
    return sorted(_topk_map(method))


def load_topk(method: str = "mi", k: int = 10) -> list[str]:
    """读取方法 method 的 Top-k 特征名单（按该方法的重要性顺序，长度 k）。

    返回顺序是**该方法的重要性顺序**；若直接用于读数据，
    `load_split(..., columns=load_topk(...))` 会自动转成规范列序。
    """
    mapping = _topk_map(method)
    if k not in mapping:
        raise DataInterfaceError(
            f"{TOP_K_FILES.get(method, method)} 中没有 Top-{k}。"
            f"可用档位：{sorted(mapping)}")
    feats = mapping[k]
    unknown = [f for f in feats if f not in FEATURE_ORDER]
    if unknown:
        raise DataInterfaceError(
            f"{TOP_K_FILES.get(method, method)} 的 Top-{k} 含未知特征：{unknown}")
    return feats


# ============================================================
# 数据版本指纹
# ============================================================
def data_fingerprint(with_hash: bool = False) -> dict:
    """记录数据版本（统一要求：记录数据版本）。

    默认只取文件大小与修改时间（本地时区）；with_hash=True 时额外计算 sha256 前 16 位。
    """
    out: dict[str, dict] = {}
    for split in SPLITS:
        for kind in ("X", "y"):
            path = PROCESSED_DIR / f"{kind}_{split}.csv"
            if not path.exists():
                out[path.name] = {"exists": False}
                continue
            st = path.stat()
            item = {
                "exists": True,
                "bytes": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
            if with_hash:
                h = hashlib.sha256()
                with path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                item["sha256"] = h.hexdigest()[:16]
            out[path.name] = item
    return out


# ============================================================
# 命令行自检
# ============================================================
def _stat_line(split: str, st: dict) -> tuple[str, bool]:
    good = bool(st["columns_ok"]) and st["n_missing"] == 0
    line = (f"  {split:<8} {'OK  ' if good else 'FAIL'}  "
            f"行数={st['n_rows']:>7,}  特征={st['n_features']:>2}  "
            f"欺诈={st['n_fraud']:>4}  ({st['fraud_rate']:.4%})  "
            f"缺失={st['n_missing']}")
    return line, good


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="统一数据读取接口自检（只读）")
    parser.add_argument("--with-test", action="store_true",
                        help="连测试集内容一起校验（默认只检查文件存在）")
    parser.add_argument("--hash", action="store_true",
                        help="计算 sha256 指纹（较慢）")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("统一数据读取接口自检（只读，不修改任何文件）")
    print("=" * 78)
    print(f"项目根目录 : {PROJECT_ROOT}")
    print(f"数据目录   : {PROCESSED_DIR}")
    print(f"特征数     : {len(FEATURE_ORDER)}   标签列: {LABEL_COL}   种子: {SEED}")
    print(f"特征顺序   : {FEATURE_ORDER[0]}, {FEATURE_ORDER[1]}, ..., {FEATURE_ORDER[-1]}")

    print("\n--- 数据版本指纹 ---")
    for name, info in data_fingerprint(with_hash=args.hash).items():
        if not info.get("exists"):
            print(f"  {name:<12} 缺失")
        else:
            extra = f"  sha256={info['sha256']}" if "sha256" in info else ""
            print(f"  {name:<12} {info['bytes']:>12,} B  {info['mtime']}{extra}")

    ok = True
    print("\n--- 划分校验 ---")
    for split in ("train", "valid"):
        try:
            line, good = _stat_line(split, validate_split(split))
            print(line)
            ok = ok and good
        except DataInterfaceError as exc:
            ok = False
            print(f"  {split:<8} FAIL  {exc}")

    test_exists = _x_path("test").exists() and _y_path("test").exists()
    if args.with_test and test_exists:
        try:
            line, good = _stat_line("test", validate_split("test"))
            print(line)
            ok = ok and good
        except DataInterfaceError as exc:
            ok = False
            print(f"  {'test':<8} FAIL  {exc}")
    else:
        print(f"  {'test':<8} 文件存在={test_exists}"
              "（内容未读取；加 --with-test 可校验）")
        ok = ok and test_exists

    print("\n--- 特征列表可用性 ---")
    for method in TOP_K_FILES:
        try:
            print(f"  {method:<9} Top-k 档位：{available_topk(method)}")
        except Exception as exc:
            print(f"  {method:<9} 不可用（{type(exc).__name__}: {exc}）")

    print("\n" + ("✅ 自检通过" if ok else "❌ 自检未通过，请检查上面的 FAIL 项"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
