import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, recall_score, 
    precision_score, f1_score, matthews_corrcoef, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# ==========================================
# 0. 路径配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_DIR = os.path.join(project_root, "data", "processed")
RESULTS_TABLE_DIR = os.path.join(project_root, "results", "tables")
os.makedirs(RESULTS_TABLE_DIR, exist_ok=True)

# ==========================================
# 1. 统一评估函数
# ==========================================
def evaluate_model(model, model_name, X_test, y_test, method_name, threshold=0.5):
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    return {
        'Method': method_name,
        'Model': model_name,
        'PR_AUC': average_precision_score(y_test, y_scores),
        'ROC_AUC': roc_auc_score(y_test, y_scores),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'False_Positives': fp
    }

# ==========================================
# 2. 加载数据 (严格控制变量)
# ==========================================
print("正在加载数据...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))['Class']
X_valid = pd.read_csv(os.path.join(DATA_DIR, "X_valid.csv"))
y_valid = pd.read_csv(os.path.join(DATA_DIR, "y_valid.csv"))['Class']

# ==========================================
# 3. 执行对照实验
# ==========================================
results = []

# 模型初始化 (控制变量：除特定不平衡处理外，模型超参数保持一致)
lr_model = LogisticRegression(max_iter=2000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# ----------------- 组别 A：无处理基线 (Baseline) -----------------
print("\n>>> 正在跑 组别 A：无处理基线")
# LR
lr_a = LogisticRegression(max_iter=2000, random_state=42)
lr_a.fit(X_train, y_train)
results.append(evaluate_model(lr_a, "Logistic Regression", X_valid, y_valid, "A. Baseline"))

# RF
rf_a = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_a.fit(X_train, y_train)
results.append(evaluate_model(rf_a, "Random Forest", X_valid, y_valid, "A. Baseline"))


# ----------------- 组别 B：类别权重 (Class Weight) -----------------
print("\n>>> 正在跑 组别 B：类别权重")
# LR
lr_b = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
lr_b.fit(X_train, y_train)
results.append(evaluate_model(lr_b, "Logistic Regression", X_valid, y_valid, "B. Class Weight"))

# RF
rf_b = RandomForestClassifier(class_weight='balanced_subsample', n_estimators=100, random_state=42, n_jobs=-1)
rf_b.fit(X_train, y_train)
results.append(evaluate_model(rf_b, "Random Forest", X_valid, y_valid, "B. Class Weight"))


# ----------------- 组别 C：SMOTE 重采样 -----------------
print("\n>>> 正在对训练集执行 SMOTE... (严格限制只在训练集执行)")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE 后训练集样本数: {len(y_train_smote)} (欺诈样本数: {sum(y_train_smote)})")

# LR
lr_c = LogisticRegression(max_iter=2000, random_state=42)
lr_c.fit(X_train_smote, y_train_smote)
results.append(evaluate_model(lr_c, "Logistic Regression", X_valid, y_valid, "C. SMOTE"))

# RF
rf_c = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_c.fit(X_train_smote, y_train_smote)
results.append(evaluate_model(rf_c, "Random Forest", X_valid, y_valid, "C. SMOTE"))

# ==========================================
# 4. 结果汇总与保存
# ==========================================
results_df = pd.DataFrame(results)

# 调整列的顺序以便阅读
columns_order = ['Method', 'Model', 'PR_AUC', 'ROC_AUC', 'Recall', 'Precision', 'F1', 'MCC', 'False_Positives']
results_df = results_df[columns_order]

print("\n" + "="*60)
print("B4 任务：类别不平衡处理方法严谨对比实验结果")
print("="*60)
print(results_df.to_string(index=False))

# 保存为 CSV 文件
save_path = os.path.join(RESULTS_TABLE_DIR, "b4_imbalance_comparison.csv")
results_df.to_csv(save_path, index=False)
print(f"\n✅ 结果已保存至: {save_path}")
