# DBSCAN：42,721 条数据整批聚类实验

## 一、具体流程


输入为 `data\verified\X_valid(1).csv`，共 42,721 行、30 个特征；聚类完成后读取同目录 y_valid(1).csv，真实欺诈 74 条、正常 42,647 条。cohort_id=`60b7a821795b61d4`；row_index 是这份 CSV 从 0 开始的行位置。

当前保留 **21 条参数记录、17 个不同组合**：原始六组、固定 min_samples=10 的 eps 对照，以及固定 eps=6 的 min_samples 对照。重复组合的簇划分相同，不作为独立重复实验。第五部分展示 **eps=6**。

按用户确定的主展示配置 **eps=6、min_samples=10**，异常候选 **608** 条，检出 **58/74** 条欺诈，误报 **550** 条，漏报 **16** 条。Recall=78.38%，Precision=9.54%，F1=0.1701，MCC=0.2706。主结果 CSV、聚类投影图和辅助 PR 图均使用这一配置。它是本次选定的展示配置，不代表所有参数组合中的最高 F1。

## 二、统一设置与计算方法

- 聚类输入：全部 42,721 条、30 维特征。Class 不进入特征，不参与拟合或异常判定。
- 缩放口径与旧实验一致：只用原 X_train 拟合 StandardScaler，然后变换本次全部输入。X_train 仅提供缩放统计，不提供本次 DBSCAN 的核心点或邻居。
- 欧氏距离，min_samples 包含样本自身；cluster_label=-1 为噪声候选。
- 当前机器内存不足以同时保存大 eps 下的所有半径邻域，因此通过精确近邻距离、互可达最小生成树还原 DBSCAN 核心点连通分量，再分块接入边界点。没有采样、降维或近似距离。
- 核心判据为第 min_samples 个近邻距离≤eps。边界点可属于多个簇时，按最小核心点行位置的簇优先接纳，与标准 DBSCAN 的扫描顺序一致。
- 最小生成树仅是计算核心连通性的工具；没有调用 HDBSCAN 的簇选择规则，没有改用 HDBSCAN 算法。内部计算接口固定为 scikit-learn 1.7.2。

PCA 只用于作图。所有参数的簇标签计算完毕后，才读取真实标签评价。


## 四、固定 min_samples=10，改变 eps

| eps | min_samples | 核心点 | 簇数 | 异常候选 | 检出欺诈 | 误报 | 漏报 | Recall | Precision | F1 | MCC | 辅助 PR-AUC（AP） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 10 | 20055 | 113 | 17829 | 71/74 | 17758 | 3 | 95.95% | 0.40% | 0.0079 | 0.0458 | 0.1153 |
| 2.5 | 10 | 27216 | 38 | 10748 | 68/74 | 10680 | 6 | 91.89% | 0.63% | 0.0126 | 0.0641 | 0.1153 |
| 3 | 10 | 32822 | 28 | 6268 | 63/74 | 6205 | 11 | 85.14% | 1.01% | 0.0199 | 0.0830 | 0.1153 |
| 3.5 | 10 | 36453 | 8 | 3792 | 63/74 | 3729 | 11 | 85.14% | 1.66% | 0.0326 | 0.1117 | 0.1153 |
| 4 | 10 | 38674 | 2 | 2402 | 61/74 | 2341 | 13 | 82.43% | 2.54% | 0.0493 | 0.1389 | 0.1153 |
| 4.5 | 10 | 40007 | 3 | 1638 | 61/74 | 1577 | 13 | 82.43% | 3.72% | 0.0713 | 0.1705 | 0.1153 |
| 5 | 10 | 40843 | 1 | 1119 | 60/74 | 1059 | 14 | 81.08% | 5.36% | 0.1006 | 0.2046 | 0.1153 |
| 6 | 10 | 41709 | 1 | 608 | 58/74 | 550 | 16 | 78.38% | 9.54% | 0.1701 | 0.2706 | 0.1153 |
| 7 | 10 | 42136 | 3 | 362 | 39/74 | 323 | 35 | 52.70% | 10.77% | 0.1789 | 0.2357 | 0.1153 |
| 10 | 10 | 42516 | 2 | 111 | 18/74 | 93 | 56 | 24.32% | 16.22% | 0.1946 | 0.1969 | 0.1153 |

![eps 对照实验](https://raw.githubusercontent.com/zzzotter/CreditCardFraudDetecton/main/results/figures/dbscan_batch_eps_sweep.png)


由图表可见eps为6时各项指标都很好，下一组实验选择固定eps=6


## 五、固定 eps=6，改变 min_samples

| eps | min_samples | 核心点 | 簇数 | 异常候选 | 检出欺诈 | 误报 | 漏报 | Recall | Precision | F1 | MCC | 辅助 PR-AUC（AP） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 5 | 41993 | 8 | 499 | 37/74 | 462 | 37 | 50.00% | 7.41% | 0.1291 | 0.1893 | 0.0864 |
| 6 | 10 | 41709 | 1 | 608 | 58/74 | 550 | 16 | 78.38% | 9.54% | 0.1701 | 0.2706 | 0.1153 |
| 6 | 20 | 41407 | 1 | 695 | 59/74 | 636 | 15 | 79.73% | 8.49% | 0.1534 | 0.2572 | 0.1326 |
| 6 | 30 | 41217 | 1 | 750 | 60/74 | 690 | 14 | 81.08% | 8.00% | 0.1456 | 0.2516 | 0.1470 |
| 6 | 50 | 40966 | 1 | 827 | 60/74 | 767 | 14 | 81.08% | 7.26% | 0.1332 | 0.2393 | 0.1720 |

本轮保持原全特征 StandardScaler，只改变 min_samples。

仅比较固定 eps=6 的五组，**min_samples=10** 的 F1 与 MCC 最高：F1=0.1701、MCC=0.2706，检出欺诈 58/74 条，误报 550 条，漏报 16 条。

![eps 对照实验](https://raw.githubusercontent.com/zzzotter/CreditCardFraudDetecton/main/results/figures/dbscan_batch_min_samples_sweep.png)


## 六、PR-AUC 与 Recall@K 

沿用原训练实验的辅助评分口径：`auxiliary_knn_score` 是到第 k=min_samples−1 个其他样本的欧氏距离，越大越稀疏。以它计算 PR-AUC（Average Precision，AP）和 Recall@K。固定 min_samples 时，评分不随 eps 改变，因此辅助 AP 重复是预期现象。

**辅助 PR-AUC 评价的是 kNN 排序，不是 DBSCAN 原生异常概率，也不是 Precision/Recall 相除。** DBSCAN 的 Recall、Precision、F1、MCC 使用实际簇标签中的噪声来计算。辅助 PR 曲线不标注 DBSCAN 的工作点，因为二者不是同一条阈值判定规则。

下面使用**固定 min_samples=10** 对应的辅助 kNN 分数（k=9）绘制一条 PR 曲线：辅助 PR-AUC（AP）=0.1153，Recall@100=20.27%，Recall@500=72.97%，Recall@1000=81.08%。同分按 CSV 行位置升序。固定 min_samples=10 的各个 eps 使用相同分数，因此它们的辅助 PR 曲线重合，图中只展示一条。

![辅助 kNN PR 曲线](https://raw.githubusercontent.com/zzzotter/CreditCardFraudDetecton/main/results/figures/dbscan_batch_pr_curve.png)


## 七、结果文件

![本轮候选聚类投影](https://raw.githubusercontent.com/zzzotter/CreditCardFraudDetecton/main/results/figures/dbscan_clusters.png)

- [全部参数指标](../results/validation/dbscan_batch/tables/dbscan_batch_metrics.csv)
- [原始六组指标](../results/validation/dbscan_batch/tables/dbscan_batch_initial_grid_metrics.csv)
- [eps 对照指标](../results/validation/dbscan_batch/tables/dbscan_batch_eps_sweep_metrics.csv)
- [min_samples 对照指标](../results/validation/dbscan_batch/tables/dbscan_batch_min_samples_sweep_metrics.csv)
- [本轮候选的逐样本簇标签与核心点标记](../results/validation/dbscan_batch/tables/dbscan_cluster_results.csv)
- [eps=6、min_samples=10 的逐样本结果](../results/validation/dbscan_batch/tables/dbscan_results_eps6_m10.csv)
- [全部已完成组合的标签宽表](../results/validation/dbscan_batch/tables/dbscan_batch_all_labels.csv)
- [五种 k 的辅助分数](../results/validation/dbscan_batch/tables/dbscan_batch_auxiliary_scores.csv)


## 八、耗时与评价边界

近邻距离计算 5.97 秒；5 个 min_samples 的精确连通树合计 245.20 秒；各阶段连通分量提取与边界接入累计 69.45 秒。共享计算只计一次，不伪造每个组合独立拟合的耗时。元数据和各连通树保存在 artifacts/，不必上传数值缓存。

历史补充 eps=6 时复用了已经计算的精确近邻距离和五棵连通树。这些结构只依赖相同数据、缩放和 min_samples，不依赖 eps；当时重新按 eps=6 提取簇及分配边界点，新增计算耗时单独记录在运行元数据中。切换报告展示的 eps 不需要重新聚类。

之前已独立使用标准 scikit-learn DBSCAN，在全部 42,721 条样本上对 eps=2、min_samples=10 运行对照；全部簇标签和核心点索引完全一致。本次已重算 21 条记录的所有指标，并抽样核对新增 eps=6 结果的近邻距离、核心点和边界/噪声规则；既有其他实验文件校验通过。

环境：Python 3.13.5，NumPy 2.2.6，scikit-learn 1.7.2。

```bash
python src/dbscan_batch_experiments.py
```



- [DBSCAN 核心点与边界定义](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.DBSCAN.html)
- [互可达距离与 DBSCAN* 连通分量](https://scikit-learn.org/1.7/modules/clustering.html#hdbscan)
