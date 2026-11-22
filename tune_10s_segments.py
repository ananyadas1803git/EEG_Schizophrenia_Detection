from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

# 5 Classifiers
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
PLOTS_DIR = OUTPUT_DIR / "segmented_10s_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

input_file = OUTPUT_DIR / "schizophrenia_10s_6segments.csv"
if not input_file.exists():
    raise FileNotFoundError("Run segment_10s_features.py first to create schizophrenia_10s_6segments.csv!")

print(f"Loading 10-second segmented dataset: {input_file.name}...")
df = pd.read_csv(input_file)
print(f"Loaded Shape: {df.shape} (504 total segments across {df.shape[1] - 3} features)\n")

X = df.drop(columns=['subject_id', 'segment_id', 'label'])
y = df['label'].values
groups = df['subject_id'].values

# ==============================================================================
# DEFINE TUNING GRIDS
# ==============================================================================
def get_tuning_configs():
    return {
        "Naïve Bayes": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('nb', GaussianNB())]),
            "params": {'nb__var_smoothing': [1e-9, 1e-6, 1e-4, 1e-2, 1e-1]}
        },
        "Tuned SVM (RBF)": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('svm', SVC(kernel='rbf', probability=True, random_state=42))]),
            "params": {
                'svm__C': [1.0, 2.5, 5.0, 10.0],
                'svm__gamma': ['scale', 0.005, 0.01, 0.03]
            }
        },
        "Tuned Random Forest": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))]),
            "params": {
                'rf__n_estimators': [100, 150],
                'rf__max_depth': [6, 8, 10],
                'rf__min_samples_split': [2, 4],
                'rf__max_features': ['sqrt', 0.35]
            }
        },
        "Tuned XGBoost": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('xgb', XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1))]),
            "params": {
                'xgb__n_estimators': [100, 150],
                'xgb__max_depth': [3, 4, 5],
                'xgb__learning_rate': [0.02, 0.04, 0.08],
                'xgb__subsample': [0.8, 1.0]
            }
        },
        "Tuned ANN (MLP)": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('ann', MLPClassifier(early_stopping=True, max_iter=400, random_state=42))]),
            "params": {
                'ann__hidden_layer_sizes': [(64, 32), (128, 64)],
                'ann__activation': ['tanh', 'relu'],
                'ann__alpha': [0.005, 0.01, 0.05]
            }
        }
    }

# ==============================================================================
# 5-FOLD SUBJECT-ISOLATED TUNING & EVALUATION (PURE SEGMENT LEVEL)
# ==============================================================================
print("=" * 90)
print("EVALUATING PURE SEGMENT-LEVEL PERFORMANCE (N = 504 SEGMENTS)")
print("Validation: 5-Fold GroupKFold (Guaranteed Subject Isolation)")
print("=" * 90)

gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X, y, groups=groups))

configs = get_tuning_configs()
results = []
all_preds = {}
all_probs = {}

for model_name, cfg in configs.items():
    print(f"• Tuning & Evaluating {model_name:<22} ... ", end="", flush=True)
    
    seg_preds = np.zeros(len(y))
    seg_probs = np.zeros(len(y))
    
    for train_idx, test_idx in cv_splits:
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        train_groups = groups[train_idx]
        
        # Inner CV for tuning
        inner_gkf = GroupKFold(n_splits=3)
        inner_splits = list(inner_gkf.split(X_train, y_train, groups=train_groups))
        
        searcher = GridSearchCV(
            estimator=cfg["pipeline"],
            param_grid=cfg["params"],
            cv=inner_splits,
            scoring='f1',
            n_jobs=-1
        )
        searcher.fit(X_train, y_train)
        best_model = searcher.best_estimator_
        
        # Predict on outer test fold
        seg_preds[test_idx] = best_model.predict(X_test)
        if hasattr(best_model, "predict_proba"):
            seg_probs[test_idx] = best_model.predict_proba(X_test)[:, 1]
        else:
            seg_probs[test_idx] = seg_preds[test_idx]
            
    # Pure Segment-Level Metrics
    acc = accuracy_score(y, seg_preds) * 100
    prec = precision_score(y, seg_preds, zero_division=0)
    rec = recall_score(y, seg_preds, zero_division=0)
    f1 = f1_score(y, seg_preds, zero_division=0)
    auc = roc_auc_score(y, seg_probs)
    
    cm = confusion_matrix(y, seg_preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    all_preds[model_name] = seg_preds
    all_probs[model_name] = seg_probs
    
    print(f"Done! [Segment Acc: {acc:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}]")
    
    results.append({
        "Model": model_name,
        "Segment Accuracy (%)": round(acc, 2),
        "Segment Precision": round(prec, 4),
        "Segment Recall (Sensitivity)": round(rec, 4),
        "Segment Specificity": round(spec, 4),
        "Segment F1-Score": round(f1, 4),
        "Segment ROC-AUC": round(auc, 4)
    })

res_df = pd.DataFrame(results).sort_values("Segment Accuracy (%)", ascending=False)
save_csv = OUTPUT_DIR / "segmented_10s_pure_metrics.csv"
res_df.to_csv(save_csv, index=False)

print("\n" + "=" * 90)
print("FINAL PURE SEGMENT-LEVEL METRICS LEADERBOARD (N = 504 SEGMENTS):")
print("=" * 90)
print(res_df.to_string(index=False))
print("=" * 90)
print(f"Saved results table to: {save_csv.name}\n")

# ==============================================================================
# VISUALIZATION 1: PURE SEGMENT ACCURACY BAR CHART
# ==============================================================================
print("Generating Plots...")
plt.figure(figsize=(10, 5))
ax = res_df.set_index("Model")["Segment Accuracy (%)"].plot(
    kind="bar", color="#1f77b4", edgecolor="black", linewidth=0.8
)

plt.title("Pure Segment-Level Classification Accuracy (10s Windows, 504 Segments)", fontsize=13, weight='bold', pad=15)
plt.ylabel("Segment Accuracy (%)", fontsize=11)
plt.xlabel("Model", fontsize=11)
plt.xticks(rotation=15, fontsize=10)
plt.ylim(50, 95)
plt.grid(axis='y', linestyle=':', alpha=0.6)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height() + 0.8),
                ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "1_pure_segment_accuracy_bar.png", dpi=300)
plt.close()

# ==============================================================================
# VISUALIZATION 2: CONFUSION MATRICES (ALL 5 MODELS FOR 504 SEGMENTS)
# ==============================================================================
fig, axes = plt.subplots(1, 5, figsize=(23, 4.2))
class_labels = ['Healthy', 'Schizo']

for i, (name, preds) in enumerate(all_preds.items()):
    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                annot_kws={"size": 13, "weight": "bold"},
                xticklabels=class_labels, yticklabels=class_labels)
    axes[i].set_title(name, fontsize=11, weight='bold', pad=10)
    axes[i].set_xlabel("Predicted Label", fontsize=10)
    if i == 0:
        axes[i].set_ylabel("True Label", fontsize=11)
    else:
        axes[i].set_ylabel("")

plt.suptitle("Confusion Matrices: Pure Segment-Level Classification (504 Total Segments)", 
             fontsize=14, weight='bold', y=1.06)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "2_pure_segment_confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# VISUALIZATION 3: PURE SEGMENT ROC-AUC CURVES
# ==============================================================================
plt.figure(figsize=(9, 7))
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']

for (name, probs), color in zip(all_probs.items(), colors):
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = roc_auc_score(y, probs)
    plt.plot(fpr, tpr, lw=2.2, color=color, label=f"{name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label="Chance Level (AUC = 0.5000)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
plt.title("ROC Curves: Pure Segment-Level Classification (10s Windows, N=504 Segments)", 
          fontsize=13, weight='bold')
plt.legend(loc="lower right", fontsize=10, frameon=True, facecolor='white')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "3_pure_segment_roc_curves.png", dpi=300)
plt.close()

print(f"All 3 segment plots successfully saved in: {PLOTS_DIR}/")
print("=" * 90)