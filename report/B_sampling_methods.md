# B 模块采样实验与交接

本轮在同一套信用卡欺诈训练数据上比较普通 SMOTE、Borderline-SMOTE 和 Conservative SMOTE。Conservative SMOTE 最终保留了 517 条合成欺诈样本。初步随机森林对照显示，它以约为普通 SMOTE 0.26% 的合成样本量，在验证集上取得了接近无采样基线的表现。

## 数据与既有实验复核

本轮使用项目提交 `3960c796` 里的预处理数据和现有划分。原始训练集有 199,364 条交易、30 个特征，包含 199,020 条正常交易和 344 条欺诈交易。验证集有 42,721 条交易，其中 74 条是欺诈。采样脚本从 `data/processed/X_train.csv` 和 `y_train.csv` 读取原始训练数据。 [项目数据说明](../data/processed/README.md)

既有普通 SMOTE 脚本使用 `sampling_strategy="auto"`、`k_neighbors=5` 和随机种子 `42`，将欺诈类扩充至 199,020 条。本次复现的样本数与 B4 一致。既有 B4 中，随机森林在 0.5 阈值下使用普通 SMOTE 后，召回率从 75.68% 升至 77.03%，同时误报从 2 笔增至 10 笔。 [既有代码](../src/smote_experiment.py#L85-L99) · [B4 对照报告](B4.md)

## 方法与参数

| 方法 | 生成规则 | 本轮参数 |
| --- | --- | --- |
| 普通 SMOTE | 在真实欺诈样本与其欺诈近邻之间插值 | `sampling_strategy="auto"`，`k_neighbors=5`，`random_state=42` |
| Borderline-SMOTE | 以处于类别边界附近的欺诈点为起点，与其欺诈近邻之间插值 | `kind="borderline-1"`，`sampling_strategy="auto"`，`k_neighbors=5`，`m_neighbors=10`（库默认值），`random_state=42` |
| Conservative SMOTE | 排除孤立插值起点，限量生成，再依次做几何过滤和可信度过滤 | `k_out=5`，`q_out=0.95`，`alpha=2`，生成上限 `M=5000`，插值近邻数 `5`，`k_f=5`，`q_f=0.75`，`t_p=0.75`，`random_state=42` |

Borderline-SMOTE 使用 Han 等人的 Borderline-1 定义。Conservative SMOTE 依据 Becerra-Suarez 等人的 Algorithm 1 和公式（4）、（7）至（11）。近邻数和生成上限是本轮实现选用的设置。 [Borderline-SMOTE](https://doi.org/10.1007/11538059_91) · [Conservative SMOTE](https://doi.org/10.3389/frai.2026.1871972)

1. **异常点清理。** 对 344 条真实欺诈样本计算到 5 个其他真实欺诈近邻的平均欧氏距离，按第 95 百分位标记 18 条局部异常点。剩余 326 条用于插值。训练集保留全部 344 条真实欺诈交易。如果清理后不足 6 条，则用完整真实欺诈集合尝试插值。
2. **限量生成。** 以原始真实欺诈数计算生成量，得到 `min(5000, 2 × 344) = 688` 条候选。
3. **几何过滤。** 用全部 344 条真实欺诈样本的平均近邻距离分布取第 75 百分位，得到距离阈值 6.56047。对每条候选计算其到 5 个真实欺诈近邻的平均距离，有 662 条不超过该阈值。
4. **可信度过滤。** 用全部真实训练数据拟合 `HistGradientBoostingClassifier(random_state=42)`，其余参数采用库默认值。在几何过滤通过的候选中，保留预测欺诈概率至少为 0.75 的 517 条。

## 采样结果

| 方案 | 正常交易 | 欺诈交易 | 训练总数 | 新增合成数 | 欺诈比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始训练集 | 199,020 | 344 | 199,364 | 0 | 0.1725% |
| 普通 SMOTE | 199,020 | 199,020 | 398,040 | 198,676 | 50.0000% |
| Borderline-SMOTE | 199,020 | 199,020 | 398,040 | 198,676 | 50.0000% |
| Conservative SMOTE | 199,020 | 861 | 199,881 | 517 | 0.4308% |

[普通 SMOTE 统计](../results/sampling/smote_summary.csv) · [Borderline-SMOTE 统计](../results/sampling/borderline_smote_summary.csv) · [Conservative SMOTE 统计与阈值](../results/conservative_smote/conservative_smote_summary.csv)

## 采样流程检查与复现

两份脚本都从同一版原始训练集读取数据，三个采样方案的随机种子均为 42。已逐项核对导出的特征列和顺序，均与原始训练集的 30 个特征一致。`Class` 单独作为标签读取。验证集保留原始类别比例，用于后续模型评价。

复现环境使用 `pandas 2.2.3`、`scikit-learn 1.6.1` 和 `imbalanced-learn 0.13.0`。在仓库根目录运行以下命令。

```bash
python -m src.sampling_methods
python -m src.conservative_smote --full
```

普通 SMOTE 和 Borderline-SMOTE 的命令加 `--save-data` 可同时导出重采样后的特征和标签。Conservative SMOTE 的增强训练集由原始训练集和 517 条最终合成样本组成。接入 C 的模型时，将最终合成特征按列追加到原始训练特征，并按相同顺序追加 517 个 `Class=1`。 [采样脚本](../src/sampling_methods.py) · [Conservative SMOTE 脚本](../src/conservative_smote.py) · [最终合成样本](../results/conservative_smote/final_synthetic_samples.csv)

## 随机森林验证集初步对照

四组使用同一份验证集和相同的随机森林配置，树数为 100，随机种子为 42，分类阈值为 0.5。AP 由 `average_precision_score` 计算。

| 方案 | AP | 召回率 | 精确率 | F1 | MCC | 误报 | 漏报 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 无采样 | 0.8178 | 75.68% | 96.55% | 0.8485 | 0.8546 | 2 | 18 |
| 普通 SMOTE | 0.8173 | 77.03% | 85.07% | 0.8085 | 0.8092 | 10 | 17 |
| Borderline-SMOTE | 0.8163 | 72.97% | 98.18% | 0.8372 | 0.8462 | 1 | 20 |
| Conservative SMOTE | 0.8196 | 74.32% | 96.49% | 0.8397 | 0.8466 | 2 | 19 |

普通 SMOTE 在本次阈值下检出的欺诈最多，同时增加了误报。Borderline-SMOTE 的误报最少，召回率也最低。Conservative SMOTE 以较少合成样本取得了最高 AP，和无采样的 AP 差距为 0.0018，在固定阈值下多漏报 1 笔。该差异来自单次验证集对照。正式选型由 C 在统一验证集上比较模型和阈值后确定。
