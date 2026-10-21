import os
import pandas as pd
import numpy as np

# 0. 全局路径配置 (绝对路径动态定位)
# ==========================================
# 1. 获取当前脚本（supervised.py）所在的绝对路径 (即 src 文件夹)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 向上退一级，获取项目根目录 (即 credit-card-fraud 文件夹)
project_root = os.path.dirname(current_dir)

# 3. 拼接出数据目录和结果目录
DATA_DIR = os.path.join(project_root, "data", "processed")
RESULTS_TABLE_DIR = os.path.join(project_root, "results", "tables")
RESULTS_FIGURE_DIR = os.path.join(project_root, "results", "figures")

# 确保结果目录存在
os.makedirs(RESULTS_TABLE_DIR, exist_ok=True)
os.makedirs(RESULTS_FIGURE_DIR, exist_ok=True)

# ==========================================
# 任务 B1: 建立代码框架与数据检查
# ==========================================
def load_and_check_data():
    print(">>> 任务 B1: 开始加载与检查数据...")
    
    # 1. 读取数据
    # 注意：y_train.csv 读出来是 DataFrame，用 ['Class'] 提取为 Series
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")['Class']
    
    X_valid = pd.read_csv(f"{DATA_DIR}/X_valid.csv")
    y_valid = pd.read_csv(f"{DATA_DIR}/y_valid.csv")['Class']

    # 2. 检查：特征矩阵和标签数量是否对应
    assert len(X_train) == len(y_train), "❌ 训练集特征与标签数量不匹配！"
    assert len(X_valid) == len(y_valid), "❌ 验证集特征与标签数量不匹配！"
    print("✅ 检查通过：特征与标签数量一致。")

    # 3. 检查：输入特征中是否错误包含了 Class
    assert 'Class' not in X_train.columns, "❌ 训练集中错误包含了 Class 标签列！"
    assert 'Class' not in X_valid.columns, "❌ 验证集中错误包含了 Class 标签列！"
    print("✅ 检查通过：特征中未包含 Class 标签，无数据泄漏。")
    
    # 4. 检查：数据中是否有缺失值
    if X_train.isnull().sum().sum() > 0 or X_valid.isnull().sum().sum() > 0:
        raise ValueError("❌ 数据中包含缺失值，请检查！")
    print("✅ 检查通过：数据中无缺失值。")
    
    # 5. 检查：是否存在无法输入模型的非数值类型
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"❌ 发现非数值特征: {non_numeric_cols.tolist()}")
    print("✅ 检查通过：所有特征均为数值型，可输入模型。")

    # 打印数据概况，方便你核对
    print("\n--- 数据概况 ---")
    print(f"训练集大小: {X_train.shape} | 欺诈样本数: {y_train.sum()} | 占比: {y_train.mean():.4%}")
    print(f"验证集大小: {X_valid.shape} | 欺诈样本数: {y_valid.sum()} | 占比: {y_valid.mean():.4%}")
    
    return X_train, y_train, X_valid, y_valid


# 在文件开头补充导入需要的库（如果你前面没导全的话）
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, recall_score, 
    precision_score, f1_score, matthews_corrcoef, 
    confusion_matrix, precision_recall_curve
)

# ==========================================
# 任务 B2: 训练逻辑回归模型与评估
# ==========================================
def train_logistic_regression(X_train, y_train, X_valid, y_valid):
    print("\n>>> 任务 B2: 开始训练逻辑回归模型...")
    
    # 1. 初始化模型 
    # 使用 class_weight='balanced' 自动处理类别不平衡
    lr_model = LogisticRegression(
        class_weight='balanced', 
        max_iter=2000, 
        random_state=42
    )
    
    # 2. 训练模型
    lr_model.fit(X_train, y_train)
    print("模型训练完成，开始验证集预测...")
    
    # 3. 预测连续概率分数 (极其重要：保留分数，不要只留 0/1)
    y_scores = lr_model.predict_proba(X_valid)[:, 1]
    
    # 4. 设定初始阈值 0.5 (后续B5任务会调整这个阈值)
    threshold = 0.5
    y_pred = (y_scores >= threshold).astype(int)
    
    # 5. 计算各项指标
    metrics = {
        'Model': 'Logistic Regression',
        'Threshold': threshold,
        'PR_AUC': average_precision_score(y_valid, y_scores),
        'ROC_AUC': roc_auc_score(y_valid, y_scores),
        'Recall': recall_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'F1': f1_score(y_valid, y_pred),
        'MCC': matthews_corrcoef(y_valid, y_pred)
    }
    
    # 计算混淆矩阵与误报数量
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
    metrics['False_Positives (误报数)'] = fp
    metrics['True_Positives (抓到的欺诈)'] = tp
    metrics['False_Negatives (漏报数)'] = fn
    
    # 打印指标
    print("\n--- 验证集评估指标 (阈值=0.5) ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<25}: {v:.4f}")
        else:
            print(f"{k:<25}: {v}")
    print("---------------------------------\n")
    
    return y_scores, metrics


# ==========================================
# 保存结果与可视化
# ==========================================
def save_results(y_valid, y_scores, metrics):
    # 1. 保存指标到 CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(RESULTS_TABLE_DIR, "logistic_regression_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ 指标已保存至: {metrics_path}")
    
    # 2. 保存预测分数到 CSV 
    predictions_df = pd.DataFrame({
        'y_true': y_valid,
        'y_score': y_scores,
        'threshold_used': metrics['Threshold']
    })
    preds_path = os.path.join(RESULTS_TABLE_DIR, "logistic_regression_predictions.csv")
    predictions_df.to_csv(preds_path, index=False)
    print(f"✅ 预测分数已保存至: {preds_path}")
    
    # 3. 绘制并保存 PR 曲线
    precision, recall, _ = precision_recall_curve(y_valid, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label=f"LR (PR-AUC = {metrics['PR_AUC']:.4f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Logistic Regression')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    fig_path = os.path.join(RESULTS_FIGURE_DIR, "logistic_regression_pr_curve.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ PR 曲线图已保存至: {fig_path}")
    
# ==========================================
# 主函数执行
# ==========================================
# ==========================================
# 主函数执行
# ==========================================
if __name__ == "__main__":
    # B1: 加载与检查
    X_train, y_train, X_valid, y_valid = load_and_check_data()
    
    # B2: 训练与评估
    y_scores, metrics = train_logistic_regression(X_train, y_train, X_valid, y_valid)
    
    # 保存结果
    save_results(y_valid, y_scores, metrics)
    
    print("\n🎉 B2 验收完成！你可以去 results 文件夹查看生成的表格和图片了。")
