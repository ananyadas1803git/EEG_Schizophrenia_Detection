import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for clean saving in VS Code
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# The 5 Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# ==============================================================================
# PATHS & DATASET
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "segmented_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

input_file = OUTPUT_DIR / "schizophrenia_6s_10segments.csv"
if not input_file.exists():
    raise FileNotFoundError(f"Cannot find {input_file.name}. Ensure the segmented dataset is present in output/.")

print(f"Loading Segmented Dataset: {input_file.name}...")
df = pd.read_csv(input_file)
print(f"Shape: {df.shape} (840 total 6-second segments across {df.shape[1] - 3} features)\n")

# Separate Metadata & Features
meta_cols = ['subject_id', 'segment_id', 'label']
X = df.drop(columns=meta_cols)
y = df['label'].values
groups = df['subject_id'].values

# ==============================================================================
# DEFINE THE 5 MODELS WITH TUNED PARAMETERS
# ==============================================================================
models = {
    "Naïve Bayes": GaussianNB(var_smoothing=1e-5),
    "Tuned SVM (RBF)": SVC(C=2.5, kernel='rbf', gamma='scale', probability=True, random_state=42),
    "Tuned Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
    "Tuned XGBoost": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.04,
                                   subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                                   random_state=42, n_jobs=-1),
    "Tuned ANN (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), activation='tanh', alpha=0.01,
                                     early_stopping=True, max_iter=400, random_state=42)
}

# ==============================================================================
# 5-FOLD EVALUATION STRICTLY FOR SEGMENT METRICS
# ==============================================================================
print("=" * 90)
print("COMPUTING PURE SEGMENT-LEVEL METRICS (840 EEG SEGMENTS)")
print("Validation Scheme: 5-Fold GroupKFold (Guaranteed Subject Isolation)")
print("=" * 90)

gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X, y, groups=groups))

results = []
all_preds = {}
all_probs = {}

for model_name, model in models.items():
    print(f"• Evaluating {model_name:<22} ... ", end="", flush=True)
    
    seg_preds = np.zeros(len(y))
    seg_probs = np.zeros(len(y))
    
    for train_idx, test_idx in cv_splits:
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test = X.iloc[test_idx]
        
        # Scaling inside fold to avoid data leakage
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        seg_preds[test_idx] = model.predict(X_test_scaled)
        
        if hasattr(model, "predict_proba"):
            seg_probs[test_idx] = model.predict_proba(X_test_scaled)[:, 1]
        else:
            seg_probs[test_idx] = seg_preds[test_idx]
            
    # Calculate pure segment-level metrics (840 samples)
    acc = accuracy_score(y, seg_preds) * 100
    prec = precision_score(y, seg_preds, zero_division=0)
    rec = recall_score(y, seg_preds, zero_division=0)
    f1 = f1_score(y, seg_preds, zero_division=0)
    auc = roc_auc_score(y, seg_probs)
    
    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y, seg_preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    all_preds[model_name] = seg_preds
    all_probs[model_name] = seg_probs
    
    print(f"Done! [Accuracy: {acc:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | Specificity: {spec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}]")
    
    results.append({
        "Model": model_name,
        "Segment Accuracy (%)": round(acc, 2),
        "Segment Precision": round(prec, 4),
        "Segment Recall (Sensitivity)": round(rec, 4),
        "Segment Specificity": round(spec, 4),
        "Segment F1-Score": round(f1, 4),
        "Segment ROC-AUC": round(auc, 4)
    })

# ==============================================================================
# SAVE PURE SEGMENT RESULTS TABLE
# ==============================================================================
results_df = pd.DataFrame(results).sort_values("Segment Accuracy (%)", ascending=False)

csv_path = OUTPUT_DIR / "segmented_only_metrics.csv"
results_df.to_csv(csv_path, index=False)

print("\n" + "=" * 90)
print("FINAL PURE SEGMENTATION METRICS LEADERBOARD (N = 840 Segments):")
print("=" * 90)
print(results_df.to_string(index=False))
print("=" * 90)
print(f"Saved metrics table to: {csv_path.name}\n")

# ==============================================================================
# VISUALIZATION 1: SEGMENT ACCURACY & F1 BAR CHART
# ==============================================================================
print("Generating Segmentation plots...")

fig, ax = plt.subplots(figsize=(10, 5))
results_df.set_index("Model")[["Segment Accuracy (%)"]].plot(kind="bar", color="#1f77b4", ax=ax, legend=False)
plt.title("Pure Segment-Level Classification Accuracy (840 Segments)", fontsize=13, weight='bold')
plt.ylabel("Accuracy (%)", fontsize=11)
plt.xlabel("Model", fontsize=11)
plt.ylim(50, 95)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle=':', alpha=0.6)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.8),
                ha='center', va='bottom', weight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "1_segment_accuracy_bar.png", bbox_inches='tight', dpi=300)
plt.close()

# ==============================================================================
# VISUALIZATION 2: CONFUSION MATRICES (ALL 5 MODELS FOR 840 SEGMENTS)
# ==============================================================================
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
class_names = ['Healthy (0)', 'Schizo (1)']

for i, (name, preds) in enumerate(all_preds.items()):
    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    axes[i].set_title(f"{name}", fontsize=11, weight='bold')
    axes[i].set_ylabel("True Label" if i == 0 else "")
    axes[i].set_xlabel("Predicted Label")

plt.suptitle("Confusion Matrices: Pure Segment-Level Classification (840 Segments)", fontsize=14, weight='bold', y=1.05)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "2_segment_confusion_matrices.png", bbox_inches='tight', dpi=300)
plt.close()

# ==============================================================================
# VISUALIZATION 3: ROC CURVES FOR ALL 5 MODELS
# ==============================================================================
plt.figure(figsize=(9, 7))

for name, probs in all_probs.items():
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = roc_auc_score(y, probs)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
plt.title("ROC Curves: Pure Segment-Level Classification (840 Segments)", fontsize=13, weight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "3_segment_roc_curves.png", bbox_inches='tight', dpi=300)
plt.close()

print(f"All plots successfully saved to: {PLOTS_DIR}/")
print("=" * 90)
