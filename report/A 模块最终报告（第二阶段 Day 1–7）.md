# A 模块最终报告（第二阶段 Day 1–7）

**角色**：A · 数据与特征分析
**范围**：数据核对 → 预处理接口 → 互信息特征排序 → 候选特征列表 → 特征数量对照 → 方案建议
**数据**：ULB 信用卡欺诈数据集，沿用阶段一的 70/15/15 分层划分，未重新划分
**随机种子**：42　**测试集**：全程未参与任何选择环节

---

## 0 摘要

本模块在阶段一既有划分的基础上，完成了四件事：

1. **确认数据可用**：三集行数、特征数、欺诈数、类别比例、列名、标签位置与划分规则全部核对一致，并证明特征缩放只在训练集上拟合；
2. **给出统一读取接口**：`src/data_io.py` 将列序、标签、种子、路径与校验集中到一处，B、C 不必各自手写 `read_csv`；
3. **建立互信息特征基准**：30 个特征按互信息降序排列，并给出 Top 10 / 15 / 20 三组候选；
4. **回答"特征筛选是否有用"**：在验证集上用两个模型比较四种特征集，结论是**高度依赖模型**——RandomForest 减少特征反而在固定阈值下更好，LightGBM 减少特征则明显变差。

推荐方案为 **MI Top-20**（特征量减少 1/3）：RandomForest 上 AP 与全特征持平（0.8177 vs 0.8174），LightGBM 上保留全特征 AP 的 98.1%。

---

## 1 任务与完成度

| 日 | 文档要求 | 状态 | 交付物 |
|---|---|---|---|
| D1 | 核对三集行数/特征数/欺诈数/比例/列名/标签位置/划分 | ✅ | `data_split_info.json`（机读核对表） |
| D2 | 检查预处理代码，确认缩放仅用训练集拟合；记录接口与种子 | ✅ | `src/data_io.py`；缩放口径证据见 D1 的 json |
| D3 | 在真实训练集计算互信息并排序 | ✅ | `mutual_information.csv`、`.png`、`mutual_information_topk.json` |
| D4 | 整理 Top 10 / 15 / 20 三组候选列表 + 生成代码 | ✅ | `mutual_information_topk.json`（权威）、`mi_candidate_features.md`、`export_feature_lists.py` |
| D5 | 交 C 在验证集比较，汇总不同特征数量的指标 | ✅ | `feature_count_comparison.csv` / `.md` / `.png`、`feature_count_table.py` |
| D6 | 整理建议特征方案，说明是否改善模型表现 | ✅ | 见本报告第 3 节；详见 `Day5-6_特征数量对照与筛选方案建议.md` |
| D7 | A 模块最终报告与文件清单 | ✅ | 本文件 |

**完成度 7/7。** 另有非必做的补充分析（LightGBM 重要性稳健性检验），作为附录性证据保留，见第 6 节。

---

## 2 数据核对结论（D1）

| 划分 | 行数 | 特征数 | 欺诈数 | 欺诈比例 |
|---|---:|---:|---:|---:|
| train | 199,364 | 30 | 344 | 0.1725% |
| valid | 42,721 | 30 | 74 | 0.1732% |
| test | 42,722 | 30 | 74 | 0.1732% |

- 特征列序固定为 `Time, V1…V28, Amount`；标签列 `Class` 单独存放于 `y_*.csv`，不在特征矩阵内；
- 划分规则：先 `train_test_split(test_size=0.3, stratify=y, random_state=42)`，再对剩余 30% 做 `test_size=0.5`，得到 70/15/15；
- 三集欺诈比例一致（0.1725% / 0.1732% / 0.1732%），分层抽样生效。

以上结论的机读记录：`results/feature_selection/data_split_info.json`。

---

## 3 关键结论

### 3.1 缩放只发生在训练集（D2）

| 指标 | train | valid | test |
|---|---:|---:|---:|
| Time 均值 | −0.000000 | −0.003969 | −0.007873 |
| **Time 标准差** | **1.000000** | 0.997663 | 1.002058 |
| Amount 均值 | +0.000000 | +0.001813 | −0.002740 |
| **Amount 标准差** | **1.000000** | 0.979910 | 0.940993 |

若缩放器在验证集或全量数据上拟合，valid/test 的标准差也会恰为 1。现在只有 train 恰好为 1，valid/test 各自偏离——**这是"用训练集参数 transform 其余数据"的数学指纹**，比任何代码注释都更可靠。V1–V28 未做二次缩放（它们本身已是 PCA 结果）。

### 3.2 互信息排序（D3）

| k | 特征（按 MI 降序） |
|---|---|
| 10 | V17, V14, V12, V10, V11, V16, V4, V3, V9, V18 |
| 15 | 上述 10 个 + V7, V2, V21, V5, V27 |
| 20 | 上述 15 个 + V6, V1, V8, Time, V28 |

- 标签熵 H(Y) = 0.012702 nats 是理论上限；第一名 V17 的 MI = 0.008431，相当于 H(Y) 的 66.4%；
- 前 6 名（V17, V14, V12, V10, V11, V16）构成一个高相关梯队，彼此差距小于 kNN 估计噪声，只能表述为"同属第一梯队"，不能排序；
- `Amount`（第 21）与 `Time`（第 19）落在中游，Top 10 恰好把两者排除——这是数据驱动结果，不是人为设定。

### 3.3 特征数量对性能的影响（D5）

| k | LightGBM AP | 保留率 | RandomForest AP | 保留率 |
|---:|---:|---:|---:|---:|
| 10 | 0.6946 | 83.0% | 0.7977 | 97.6% |
| 15 | 0.8014 | 95.7% | 0.8016 | 98.1% |
| 20 | 0.8210 | 98.1% | 0.8177 | 100.0% |
| 30 | 0.8370 | 100% | 0.8174 | 100% |

**两条曲线的形状完全不同**：LightGBM 对特征数量敏感且单调，RandomForest 几乎不敏感（Top-20 与全特征的 AP 差 0.0002，混淆矩阵完全相同）。

### 3.4 RandomForest + Top-10 是唯一"筛选带来真实收益"的组合（D5）

阈值固定 0.5：

| 配置 | FP | FN | Recall | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| RF + Top-10 | 1 | **19** | 0.7432 | 0.9821 | 0.8462 | 0.8542 |
| RF + All-30 | 1 | 23 | 0.6892 | 0.9808 | 0.8095 | 0.8219 |

**少用 20 个特征，多抓回 4 个欺诈交易，误报数不变。** F1 提升 3.7 个百分点、MCC 提升 3.2 个百分点，且 F1 与 MCC 同向，说明是真实判别力提升而非"预测得更多"。

### 3.5 模型选择的影响大于特征数量

| 对比 | AP 差异 |
|---|---:|
| 同一特征集（Top-10）下，RF 优于 LGBM | +0.1031 |
| 同一特征集（All-30）下，LGBM 优于 RF | +0.0196 |
| LGBM 自身 10 → 30 个特征 | +0.1424 |
| RF 自身 10 → 30 个特征 | +0.0197 |

因此"选几个特征"必须先确定模型才能回答。

### 3.6 推荐方案（D6）

| 方案 | 特征集 | 模型 | 验证集（最优阈值） | 适用前提 |
|---|---|---|---|---|
| A | MI Top-10 | RandomForest | AP 0.7977，F1 0.8571，MCC 0.8624，FP 2 / FN 17 | 误报代价高 |
| B | MI Top-20 | LightGBM | AP 0.8210，F1 0.8112，MCC 0.8114，FP 11 / FN 16 | 需要较高召回 |
| **C（推荐）** | **MI Top-20** | **RandomForest** | **AP 0.8177，F1 0.8397，MCC 0.8466，FP 2 / FN 19** | 无明显短板，特征量减 1/3 |
| ✗ | MI Top-10 | LightGBM | AP 0.6946，F1 0.7778，FP 14 / FN 18 | 特征不足，不可用 |

**结论：特征筛选是否改善模型表现，取决于模型。** 对 RandomForest 改善（固定阈值下 FN 23→19、F1 +3.7pp），对 LightGBM 未改善（全特征最优，Top-10 损失 17% AP）。推荐 C（Top-20 + RF）；若团队最终模型为 LightGBM，退化为 B（Top-20），AP 保留 98.1%。

---

## 4 交付接口（给 B、C）

### 4.1 统一数据读取

```python
import sys; sys.path.insert(0, "src")
import data_io

X_tr, y_tr = data_io.load_split("train")          # 30 列，规范列序，读取即校验
X_va, y_va = data_io.load_split("valid")

feats = data_io.load_topk("mi", 20)               # 三组候选特征，从 json 读取并校验
X_sub, y_sub = data_io.load_split("train", columns=feats)
```

`load_split` 会在读取时校验：行数（199,364 / 42,721 / 42,722）、30 列且列序正确、无缺失、标签只含 0/1、欺诈数（344 / 74 / 74）。任何不符立刻抛 `DataInterfaceError`。

### 4.2 候选特征

三组名单以 `results/feature_selection/mutual_information_topk.json` 的 `top_k` 字段为唯一权威来源；`mi_candidate_features.md` 是同一内容的人可读版本。**不依赖任何 txt 文件。**

### 4.3 阈值范围（重要）

| 模型 | 验证集最优阈值 | 说明 |
|---|---|---|
| LightGBM（`class_weight="balanced"`） | **0.946 – 0.997** | 加权使输出概率呈双峰，最优判据被推向 0.95 以上 |
| RandomForest（`balanced_subsample`） | **0.350 – 0.400** | 概率分布更平缓 |

**两者不可共用同一搜索区间。** 若沿用 0.05–0.70，LightGBM 会整段落在最优值左侧，得出"阈值越低 F1 越高"的错误结论。

---

## 5 防泄漏设计

| 环节 | 约束 | 执行方式 |
|---|---|---|
| 特征选择 | 只用训练集 | `mutual_information.py` 仅读 `X_train.csv` / `y_train.csv` |
| 特征缩放 | 只在训练集拟合 | 阶段一完成；本文档第 3.1 节给出数据层面的证明 |
| LightGBM 重要性 | 不使用早停 | 早停需验证集标签，会使特征排序间接"看过"验证集，故树数固定 |
| 特征数量与阈值 | 只用验证集 | 测试集未读入 |
| 测试集 | 只评一次 | 全部脚本默认不读测试集，`load_split("test")` 需显式 `allow_test=True` |

---

## 6 补充分析与未纳入项

### 6.1 补充分析：LightGBM 重要性的稳健性检验

新文档未要求，作为交叉验证保留。5 组配置 × 5 个种子共 25 次训练：平均名次相关 0.796、Top-10 集合平均 Jaccard 0.664；前 4 名（V14/V10/V4/V12）在 25 次训练中全部进入 Top-10，第 8 名之后命中率降至 36%–72%。

结论：**LightGBM 重要性的顶端可信、中段不可细究**。该分析用于说明"为什么 MI 是主方法"，不参与最终特征集选取。

### 6.2 未纳入：FID-SOM

新文档明确"FID-SOM 暂不列为本周必做；如需加入，先明确论文依据、算法定义及其与本项目目标的关系"。

已完成的部分：
- 定位到正确文献：Breskuvienė & Dzemyda, *FID-SOM: feature selection for imbalanced data using self-organizing map*, Journal of Big Data (2024) 11:182, doi:10.1186/s40537-024-01059-5；
- 核对该文算法（min-max 输入 → 矩形 SOM → 取不同 BMU 的权重向量构成矩阵 → 列归一化 → 列方差排序），发现早期实现有三处偏差（z-score 代替 min-max、按样本加权代替 BMU 去重、缺列归一化）；
- 因时间安排未按论文口径重跑，故**不纳入本次交付**，相关工作留作后续。

### 6.3 未随包提交的辅助文件

`mi_top10/15/20.txt` 三个纯文本列表可由 `src/export_feature_lists.py` 秒级重新生成，报告中已无引用，故未提交。

---

## 7 局限性

1. **验证集正例极少**（74 个）：1 个 FN 对应 1.35pp 的 Recall，AP 差 0.02 可能只涉及 1–2 个样本的排序变化（如 RF 的 Top-20 与 All-30 混淆矩阵完全相同，不可区分）。
2. **双重乐观**：特征数量与阈值均在同一个验证集上挑选，指标偏乐观；最终配置必须在测试集上只评一次。
3. **重复记录未处理**：数据含 773 组 / 1,081 条重复（与参考论文一致），保留原因为标签无冲突且删除会改变类别比例；此为已知局限。
4. **本模块的对照表为 A 侧预演**：评价函数由 A 实现，与 C 的统一评价函数可能存在口径差异，正式对照以 C 的结果为准。
5. **未覆盖采样方法**：本模块在未采样的训练集上训练。若与 B 的采样方法叠加，最优特征数量可能改变，需联合评审确认。
6. **特征重要性不等于因果**：Top-k 是"在这份数据、这套模型下的排序"，不是数据的内在属性。

---

## 8 复现步骤

按顺序执行（均在项目根目录）：

- `python src/gen_data_split_info.py` —— 重建数据核对表（约 1 秒）
- `python src/mutual_information.py` —— 互信息排序（约 1–3 分钟）
- `python src/export_feature_lists.py` —— 生成候选列表（秒级）
- `python src/feature_count_table.py` —— 特征数量对照表（约 55 秒）
- `python src/data_io.py` —— 数据接口自检（秒级，预期输出"自检通过"）

补充分析：
- `python src/lightgbm_importance.py` —— LightGBM gain 重要性（约 10 秒）
- `python src/lightgbm_importance_robustness.py` —— 25 次稳健性检验（约 60 秒）

可复现性：固定种子 42；无随机重采样；特征列序由 `data_io.canonical_columns` 强制规范；重复运行除输出文件中的时间戳外结果一致。

---

## 9 文件清单

### 9.1 代码（7 个）

| 文件 | 对应 Day | 作用 |
|---|---|---|
| `src/data_io.py` | D2 | 统一数据读取接口（列序强制、读取即校验、测试集需授权） |
| `src/gen_data_split_info.py` | D1 | 生成数据核对表 json（只读数据） |
| `src/mutual_information.py` | D3 | 互信息计算与排序 |
| `src/export_feature_lists.py` | D4 | 从互信息结果整理三组候选列表 |
| `src/feature_count_table.py` | D5 | 特征数量对照实验（4 特征集 × 2 模型 × 2 阈值策略） |
| `src/lightgbm_importance.py` | 补充 | LightGBM gain 重要性 |
| `src/lightgbm_importance_robustness.py` | 补充 | 重要性稳健性检验（5 配置 × 5 种子） |

### 9.2 结果与元数据（14 个，`results/feature_selection/`）

| 文件 | 对应 Day | 内容 |
|---|---|---|
| `data_split_info.json` | D1 | 三集行数/特征数/欺诈数/比例/列名/标签/划分 + 缩放证据 |
| `mutual_information.csv` | D3 | 30 行完整 MI 排序（rank, feature, mi_score） |
| `mutual_information_ranking.png` | D3 | 图 2-1：MI 降序条形图 |
| `mutual_information_topk.json` | D3/D4 | 三组候选特征（权威来源）+ 参数与样本信息 |
| `mi_candidate_features.md` | D4 | 三组候选择可读表格（含 MI 值与全局名次） |
| `mi_candidate_features.csv` | D4 | 同上，机器可读版本 |
| `feature_count_comparison.csv` | D5 | 长表：8 配置 × 2 阈值策略 × 13 列指标 |
| `feature_count_comparison.md` | D5 | 报告用双表（阈值 0.5 / 验证集最优） |
| `feature_count_comparison.png` | D5 | 图 5-1：AP 与 F1 随特征数量的变化 |
| `lightgbm_importance.csv` | 补充 | 30 行 gain 排序（含 gain_share、split） |
| `lightgbm_importance.png` | 补充 | 图 3-1：前 20 名 gain share |
| `lightgbm_importance_topk.json` | 补充 | Top-k + 超参 + 训练集 AP |
| `lightgbm_importance_robustness.csv` | 补充 | 稳定性聚合表（平均 share/标准差/名次区间/命中次数） |
| `lightgbm_importance_robustness.json` | 补充 | 25 次运行明细与稳定性指标 |

### 9.3 报告（3 个，`results/`）

| 文件 | 内容 |
|---|---|
| `建立特征选择基准.md` | Day 2–3 详细报告：互信息 + LightGBM 重要性 + 稳健性 + 两法对照 |
| `Day5-6_特征数量对照与筛选方案建议.md` | Day 5–6 详细报告：对照表 + 方案建议 + 与 C 的对接约定 |
| `A模块最终报告_Day1-7.md` | 本文件：总览、结论、接口、清单 |

---

## 10 与其他文档的关系

- **本文件是总览**：说明 A 做了什么、结论是什么、文件在哪、B/C 怎么用；
- **两份详细报告**提供方法细节（参数、动机、推导、局限性），答辩追问时以它们为准；
- **数据与结果文件**是与 C 对接的实际接口，C 的正式对照结果若与本模块的预演表有差异，以 C 的统一评价函数为准。