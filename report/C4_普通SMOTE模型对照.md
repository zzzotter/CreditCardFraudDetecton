# 普通 SMOTE 模型对照（C·DAY4）：随机森林 vs LightGBM

更新：2026-11-27。

## 一、目标与范围

接入**普通 SMOTE**，在相同数据划分上比较随机森林（RF）与 LightGBM 重采样后的验证集表现，并与 DAY3 基线（无处理 / 类别权重）合并成完整对照，为 DAY5 的改进采样对照（Borderline-SMOTE、Conservative SMOTE）建立 SMOTE 基线。

## 二、实验设置

| 项目 | 取值 |
| --- | --- |
| 数据划分 | `data/processed/` 70/15/15 分层抽样，随机种子 42，不重新划分 |
| 训练集 | 199,364 行 × 30 特征，欺诈 344（0.1725%） |
| 验证集 | 42,721 行，欺诈 74（0.1732%） |
| 测试集 | 本轮不读取（留待 DAY7） |
| 普通 SMOTE | `SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)`，**只在训练集执行** |
| SMOTE 后训练集 | 398,040 行（正常 199,020 / 欺诈 199,020，合成 198,676） |
| 模型 | RF（`n_estimators=100`）、LightGBM（500 树固定基线），**均不加类别权重**（重采样本身即不平衡处理） |
| 阈值 / 指标 | 0.5；AP、Recall、Precision、F1、MCC、FP、FN（统一 `src/evaluation.py`） |

## 三、普通 SMOTE 模型对照表（验证集，threshold=0.5）

| family | 方案 | AP | Recall | Precision | F1 | MCC | FP | FN | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 无处理 | 0.8178 | 0.7568 | 0.9655 | 0.8485 | 0.8546 | 2 | 18 | 0.9307 |
| Random Forest | 类别权重 | 0.8102 | 0.7432 | 0.8730 | 0.8029 | 0.8052 | 8 | 19 | 0.9429 |
| Random Forest | **SMOTE** | 0.8170 | 0.7703 | 0.8507 | 0.8085 | 0.8092 | 10 | 17 | 0.9581 |
| LightGBM | 无处理 | 0.7159 | 0.6892 | 0.7846 | 0.7338 | 0.7349 | 14 | 23 | 0.9644 |
| LightGBM | 类别权重 | 0.8163 | 0.7703 | 0.9344 | 0.8444 | 0.8482 | 4 | 17 | 0.9521 |
| LightGBM | **SMOTE** | 0.8142 | **0.8243** | 0.8472 | 0.8356 | 0.8354 | 11 | **13** | 0.9718 |

> 完整数值见 [`results/tables/c4_smote_comparison.csv`](../results/tables/c4_smote_comparison.csv)。

## 四、结果分析

1. **SMOTE 对 LightGBM 提升显著**：Recall 从无处理的 0.6892 大幅升至 0.8243，F1 从 0.7338 升至 0.8356，漏报从 23 降到 13（六组最少）。SMOTE 弥补了 LightGBM 在不加权时被多数类主导的问题。

2. **SMOTE 对 RF 提升有限且有代价**：Recall 从 0.7568 升到 0.7703，但 Precision 从 0.9655 降到 0.8507、误报从 2 升到 10、F1 从 0.8485 降到 0.8085。RF 本身对不平衡已有一定稳健性，SMOTE 的合成样本在这里主要带来额外误报。

3. **LightGBM+SMOTE 的召回是六组最高（0.8243）**，但精度（0.8472）明显低于 LightGBM 类别权重（0.9344）和 RF 无处理（0.9655）。这说明它是「更积极地抓欺诈、但误报更多」的配置，最终取舍需 DAY6 结合阈值与业务成本决定。

4. **复现核对**：RF+SMOTE 的 Recall/Precision/F1/MCC/FP 与既有 B4、B 采样报告一致（0.7703 / 0.8507 / 0.8085 / 0.8092 / 10）；AP 为 0.8170，与历史 0.8173 的 0.0003 差异来自 imbalanced-learn 版本（本轮 0.14.2，历史 0.13.0），可忽略。

5. **准备改进采样对照**：B 模块已产出 Borderline-SMOTE 与 Conservative SMOTE 的样本统计及合成样本（见 [`results/sampling/`](../results/sampling/)、[`results/conservative_smote/`](../results/conservative_smote/)）。DAY5 将把这两种采样接入，优先在表现较好的 RF 与 LightGBM 上对照，避免穷举组合。

## 五、交付文件

| 文件 | 内容 |
| --- | --- |
| [`src/c4_smote_comparison.py`](../src/c4_smote_comparison.py) | SMOTE 重采样 + RF/LightGBM 训练与对照脚本 |
| [`results/tables/c4_smote_comparison.csv`](../results/tables/c4_smote_comparison.csv) | 普通 SMOTE 模型对照表（含 DAY3 基线，6 组） |
| [`results/tables/rf_smote_valid_predictions.csv`](../results/tables/rf_smote_valid_predictions.csv) | RF + SMOTE · 验证集预测概率 |
| [`results/tables/lgbm_smote_valid_predictions.csv`](../results/tables/lgbm_smote_valid_predictions.csv) | LightGBM + SMOTE · 验证集预测概率 |
