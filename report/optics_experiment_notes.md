# OPTICS：42,721 条数据整批聚类实验

## 一、数据与主要结果

本轮使用 `data/verified/X_valid(1).csv` 的全部 **42,721 条、30 维特征**，其中欺诈 74 条。每条样本均参与密度结构计算，无抽样。cohort_id=`60b7a821795b61d4`。

主结果：**xi=0.005、min_samples=10**，异常候选 **30259** 条，检出 **74/74**，误报 **30185**，漏报 **0**；Recall=100.00%，Precision=0.24%，F1=0.0049，MCC=0.0267，辅助 AP=0.1153。

## 二、统一设置与实验口径

- 用原训练集全部 199,364 行拟合全特征 StandardScaler，再变换本批数据。训练集只提供缩放统计。
- 欧氏距离，min_samples 包含样本自身；cluster_label=-1 为异常候选。PCA 仅用于绘图。
- Class 不进入距离、拟合、簇提取。标签用于事后评价与候选配置选择，因此属于同一批数据上的探索性调参。
- 本轮直接对整批数据聚类，尚未得到独立测试结论；与 B 使用相同行也不代表训练评价协议相同。

沿用原 OPTICS 方案：**`cluster_method="xi"`**，调用 `cluster_optics_xi`；**max_eps=∞、min_cluster_size=500、predecessor_correction=True** 在两组中固定。xi 表示可达距离的相对陡峭程度，不是 DBSCAN 的 eps。

本轮复用已核验数据与参数来源的无限 max_eps 原生 OPTICS 图，重新提取 xi 标签。未使用有限 max_eps 的同半径对照图。

主要评分为 **(MCC + F1) / 2**，同分依次按 MCC、F1、辅助 AP、较少误报、较小 xi、较小 min_samples 决定。第二组按用户指定固定 **xi=0.005**，并非第一组评分最优值（第一组最优为 xi=0.04，F1=0.0820、MCC=0.0826）。第二组内部仍按评分选择主结果。

## 三、第一组：固定 min_samples=10，改变 xi

| xi | min_samples | 簇数 | 异常候选 | 检出欺诈 | 误报 | 漏报 | Recall | Precision | F1 | MCC | 辅助 AP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 10 | 18 | 30259 | 74/74 | 30185 | 0 | 100.00% | 0.24% | 0.0049 | 0.0267 | 0.1153 |
| 0.01 | 10 | 7 | 38351 | 74/74 | 38277 | 0 | 100.00% | 0.19% | 0.0039 | 0.0141 | 0.1153 |
| 0.015 | 10 | 5 | 39445 | 74/74 | 39371 | 0 | 100.00% | 0.19% | 0.0037 | 0.0120 | 0.1153 |
| 0.02 | 10 | 1 | 42174 | 74/74 | 42100 | 0 | 100.00% | 0.18% | 0.0035 | 0.0047 | 0.1153 |
| 0.04 | 10 | 1 | 48 | 5/74 | 43 | 69 | 6.76% | 10.42% | 0.0820 | 0.0826 | 0.1153 |

## 四、第二组：固定 xi=0.005，改变 min_samples

| xi | min_samples | 簇数 | 异常候选 | 检出欺诈 | 误报 | 漏报 | Recall | Precision | F1 | MCC | 辅助 AP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 5 | 20 | 30739 | 74/74 | 30665 | 0 | 100.00% | 0.24% | 0.0048 | 0.0260 | 0.0864 |
| 0.005 | 10 | 18 | 30259 | 74/74 | 30185 | 0 | 100.00% | 0.24% | 0.0049 | 0.0267 | 0.1153 |
| 0.005 | 20 | 15 | 33053 | 74/74 | 32979 | 0 | 100.00% | 0.22% | 0.0045 | 0.0225 | 0.1326 |
| 0.005 | 30 | 11 | 34884 | 74/74 | 34810 | 0 | 100.00% | 0.21% | 0.0042 | 0.0197 | 0.1470 |
| 0.005 | 50 | 7 | 38204 | 73/74 | 38131 | 1 | 98.65% | 0.19% | 0.0038 | 0.0125 | 0.1720 |

两组共 13 条记录、12 个不同组合；交叉处重复组合不作为独立重复实验。本页按要求只展开检出大于 0 的 10 条记录，省略 3 条检出为 0/74 的记录。完整 CSV 保留全部结果，展示筛选不参与参数评分。

省略的组合不输出噪声候选，全部真实欺诈均被漏报，不能将“零异常候选”当作效果好。xi 控制层级密度区域的提取，噪声数不保证随 xi 单调变化，结果可能发生明显跳变。

## 五、图示与辅助排序指标

![OPTICS 主配置聚类投影](figures/optics_clusters.png)

![OPTICS 可达距离图](figures/optics_reachability.png)

图按 OPTICS 访问顺序排列。xi 用于识别可达距离曲线中相对陡峭的变化，不对应一条水平 eps 阈值线。无限距离仅在图中放在显示上界，CSV 保留 inf。

辅助分数为到第 `min_samples-1` 个其他样本的距离，越大越稀疏。AP 使用 `average_precision_score`，评价的是辅助 kNN 排序，不是原生 OPTICS 异常概率，也不是 Precision/Recall 的比值。固定 min_samples 时，分数与 AP 不随 xi（或 DBSCAN 的 eps）改变。

下图固定 min_samples=10，辅助 AP=0.1153，Recall@100=20.27%、Recall@500=72.97%、Recall@1000=81.08%。它与其他 min_samples 的主结果 AP 可能不同；同分按行位置升序。

![固定 min_samples=10 的辅助 PR 曲线](figures/optics_pr_curve.png)

## 六、运行记录与交付文件

- 主结果：[逐行聚类标签](tables/optics_cluster_results.csv)，row_index 为输入 CSV 的零起始行位置。
- [实验总表](tables/optics_result/optics_parameter_experiments.csv)。
- 两组对照：[xi 表](tables/optics_result/optics_xi_sweep_metrics.csv)、[min_samples 表](tables/optics_result/optics_min_samples_sweep_metrics.csv)。
- [全部参数逐行标签](tables/optics_result/optics_all_cluster_labels.csv)。
- 算法与实验入口：`src/unsupervised.py`；CSV、图和报告集中在 `src/unsupervised_reporting.py`。
- [综合报告](../report/03_unsupervised_and_comparison.md)。

仅含展示行的表格：[xi 对照表](tables/optics_result/optics_xi_sweep_display.csv)、[min_samples 对照表](tables/optics_result/optics_min_samples_sweep_display.csv)。上面的实验总表与两组 metrics.csv 均为完整记录。

此前固定 xi=0.04 的版本另存为[历史实验总表](tables/optics_result/archive_before_fixed_xi0005/optics_parameter_experiments.csv)，不与本轮第二组混算。

每个 min_samples 建图一次，各 xi 从同一图提取标签。建图耗时为原生 OPTICS 实际计算时间；缓存复现保留首次耗时。

| min_samples | 建图秒数 |
| ---: | ---: |
| 5 | 615.157 |
| 10 | 640.471 |
| 20 | 660.990 |
| 30 | 664.033 |
| 50 | 674.556 |

建图任务耗时合计 **3255.208 秒**；并行计算时该总和不等于总等待时间。每次 xi 提取的耗时另存 CSV 的 extraction_seconds。

DBSCAN 通过固定半径判断核心点连通；本轮 OPTICS 通过 xi 提取可达距离的密度结构。因此二者的参数含义和异常划分规则不同。具体检出、误报及 F1/MCC 对照见综合报告。

<!-- OPTICS_EXTRA_XI01_M2_10_START -->
## 额外实验：xi=0.1，min_samples=2～10

固定 min_cluster_size=500、max_eps=∞、predecessor_correction=True；使用同一批全部 42,721 条数据与原缩放方式。以下 9 组单独记为额外实验，不替换原两组主实验、主结果或模型比较表。

| xi | min_samples | 簇数 | 异常候选 | 检出欺诈 | 误报 | 漏报 | Recall | Precision | F1 | MCC | 辅助 AP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 2 | 1 | 17 | 4/74 | 13 | 70 | 5.41% | 23.53% | 0.0879 | 0.1121 | 0.0581 |
| 0.1 | 3 | 1 | 17 | 4/74 | 13 | 70 | 5.41% | 23.53% | 0.0879 | 0.1121 | 0.0718 |
| 0.1 | 4 | 1 | 0 | 0/74 | 0 | 74 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.0755 |
| 0.1 | 5 | 1 | 0 | 0/74 | 0 | 74 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.0864 |
| 0.1 | 6 | 1 | 0 | 0/74 | 0 | 74 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.0938 |
| 0.1 | 7 | 1 | 0 | 0/74 | 0 | 74 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.0993 |
| 0.1 | 8 | 1 | 0 | 0/74 | 0 | 74 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.1039 |
| 0.1 | 9 | 1 | 0 | 0/74 | 0 | 74 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.1084 |
| 0.1 | 10 | 1 | 17 | 4/74 | 13 | 70 | 5.41% | 23.53% | 0.0879 | 0.1121 | 0.1153 |

[额外实验指标 CSV](tables/optics_result/optics_extra_xi01_min_samples_2_10_metrics.csv) · [9 组逐行标签](tables/optics_result/optics_extra_xi01_min_samples_2_10_labels.csv)

[额外实验说明与复现命令](optics_extra_experiment_notes.md)

<!-- OPTICS_EXTRA_XI01_M2_10_END -->

<!-- OPTICS_DBSCAN_CONTROL_EPS6_M10_START -->
## 同半径控制实验：eps=6、min_samples=10

OPTICS 显式使用 cluster_method=dbscan、max_eps=eps=6，与 DBSCAN 相同半径对照；不替换上述 xi 结果。

| 方法 | 参数 | min_samples | 检出欺诈 | 误报 | 漏报 | Recall | Precision | F1 | MCC | 辅助 AP | Recall@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DBSCAN | eps=6 | 10 | 58/74 | 550 | 16 | 78.38% | 9.54% | 0.1701 | 0.2706 | 0.1153 | 20.27% |
| OPTICS（dbscan 提取） | eps=6 | 10 | 58/74 | 550 | 16 | 78.38% | 9.54% | 0.1701 | 0.2706 | 0.1153 | 20.27% |

异常标记差异 0 条，分区 ARI=1.000000。

[控制实验说明及逐行对照](optics_dbscan_control_notes.md)
<!-- OPTICS_DBSCAN_CONTROL_EPS6_M10_END -->
