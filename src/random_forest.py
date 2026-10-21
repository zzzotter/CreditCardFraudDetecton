import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, recall_score, 
    precision_score, f1_score, matthews_corrcoef, 
    confusion_matrix, precision_recall_curve
)

# ==========================================
# 0. 全局路径配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_DIR = os.path.join(project_root, "data", "processed")
RESULTS_TABLE_DIR = os.path.join(project_root, "results", "tables")
RESULTS_FIGURE_DIR = os.path.join(project_root, "results", "figures")

os.makedirs(RESULTS_TABLE_DIR, exist_ok=True)
os.makedirs(RESULTS_FIGURE_DIR, exist_ok=True)

# ==========================================
# B1: 加载与检查数据
# ==========================================
def load_and_check_data():
    print(">>> 任务 B1: 开始加载与检查数据...")
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))['Class']
    X_valid = pd.read_csv(os.path.join(DATA_DIR, "X_valid.csv"))
    y_valid = pd.read_csv(os.path.join(DATA_DIR, "y_valid.csv"))['Class']

    assert len(X_train) == len(y_train), "训练集特征与标签数量不匹配！"
    assert 'Class' not in X_train.columns, "数据泄漏：训练集中包含了 Class！"
    assert X_train.isnull().sum().sum() == 0, "数据中包含缺失值！"
    print("✅ B1 检查通过：数据准备就绪。")
    return X_train, y_train, X_valid, y_valid

# ==========================================
# 统一的评估与保存函数 (供 B2/B3 复用)
# ==========================================
def evaluate_and_save(model, model_name, X_valid, y_valid, threshold=0.5):
    # 1. 预测分数
    y_scores = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    
    # 2. 计算指标
    metrics = {
        'Model': model_name,
        'Threshold': threshold,
        'PR_AUC': average_precision_score(y_valid, y_scores),
        'ROC_AUC': roc_auc_score(y_valid, y_scores),
        'Recall': recall_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'F1': f1_score(y_valid, y_pred),
        'MCC': matthews_corrcoef(y_valid, y_pred)
    }
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
    metrics['False_Positives'] = fp
    metrics['True_Positives'] = tp
    metrics['False_Negatives'] = fn
    
    # 3. 打印指标
    print(f"\n--- {model_name} 验证集评估 (阈值={threshold}) ---")
    for k in ['PR_AUC', 'ROC_AUC', 'Recall', 'Precision', 'F1', 'MCC', 'False_Positives']:
        print(f"{k:<20}: {metrics[k]:.4f}" if isinstance(metrics[k], float) else f"{k:<20}: {metrics[k]}")
    
    # 4. 保存预测分数
    preds_df = pd.DataFrame({'y_true': y_valid, 'y_score': y_scores, 'threshold_used': threshold})
    preds_df.to_csv(os.path.join(RESULTS_TABLE_DIR, f"{model_name.lower().replace(' ', '_')}_predictions.csv"), index=False)
    
    # 5. 返回结果供绘图和对比
    return metrics, y_scores

# ==========================================
# B3: 随机森林训练
# ==========================================
def train_random_forest(X_train, y_train, X_valid, y_valid):
    print("\n>>> 任务 B3: 开始训练随机森林...")
    # n_jobs=-1 表示使用所有CPU核心加速训练
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced_subsample', # 树的每个节点使用平衡权重
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    metrics, y_scores = evaluate_and_save(rf_model, "Random Forest", X_valid, y_valid)
    return metrics, y_scores

# ==========================================
# 汇总对比与可视化 (B3 任务)
# ==========================================
def compare_and_plot(metrics_lr, scores_lr, metrics_rf, scores_rf, y_valid):
    # 1. 生成对比表
    comparison_df = pd.DataFrame([metrics_lr, metrics_rf])
    comparison_df.to_csv(os.path.join(RESULTS_TABLE_DIR, "supervised_baseline_comparison.csv"), index=False)
    print("\n✅ 模型对比表已保存。")

    # 2. 绘制叠加的 PR 曲线
    plt.figure(figsize=(8, 6))
    for name, scores in [("Logistic Regression", scores_lr), ("Random Forest", scores_rf)]:
        precision, recall, _ = precision_recall_curve(y_valid, scores)
        auc_val = average_precision_score(y_valid, scores)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {auc_val:.4f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_FIGURE_DIR, "supervised_pr_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 对比 PR 曲线图已保存。")

# ==========================================
# 主函数执行
# ==========================================
if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid = load_and_check_data()
    
    # B2: 逻辑回归
    print("\n>>> 任务 B2: 训练逻辑回归...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    lr_model.fit(X_train, y_train)
    metrics_lr, scores_lr = evaluate_and_save(lr_model, "Logistic Regression", X_valid, y_valid)
    
    # B3: 随机森林
    metrics_rf, scores_rf = train_random_forest(X_train, y_train, X_valid, y_valid)
    
    # 汇总对比
    compare_and_plot(metrics_lr, scores_lr, metrics_rf, scores_rf, y_valid)
    
    print("\n🎉 B2 和 B3 任务全部完成！")
