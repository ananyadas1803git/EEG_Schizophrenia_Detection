from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for clean saving in VS Code
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
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
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
BASE_PLOTS_DIR = OUTPUT_DIR / "segment_slice_plots"
BASE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

input_file = OUTPUT_DIR / "schizophrenia_10s_6segments.csv"
if not input_file.exists():
    raise FileNotFoundError("Could not find schizophrenia_10s_6segments.csv in output/!")

print(f"Loading dataset: {input_file.name}...")
df = pd.read_csv(input_file)
print(f"Shape: {df.shape} (504 rows total across 6 segments)\n")

# Time window labels
SEGMENT_INFO = {
    0: ("Segment_0_0-10s", "Segment 0 (0s – 10s)"),
    1: ("Segment_1_10-20s", "Segment 1 (10s – 20s)"),
    2: ("Segment_2_20-30s", "Segment 2 (20s – 30s)"),
    3: ("Segment_3_30-40s", "Segment 3 (30s – 40s)"),
    4: ("Segment_4_40-50s", "Segment 4 (40s – 50s)"),
    5: ("Segment_5_50-60s", "Segment 5 (50s – 60s)")
}

# The 5 Models (Tuned Settings)
def get_models():
    return {
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.04, 
                                 subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', 
                                 random_state=42, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), activation='tanh', alpha=0.01, 
                                  early_stopping=True, max_iter=400, random_state=42),
        "Naïve Bayes": GaussianNB(var_smoothing=1e-5),
        "SVM (RBF)": SVC(C=2.5, kernel='rbf', gamma='scale', probability=True, random_state=42)
    }

# ==============================================================================
# MAIN LOOP: EVALUATE & GENERATE 3 PLOTS SEPARATELY FOR EACH SEGMENT
# ==============================================================================
print("=" * 95)
print("GENERATING SEPARATE BAR GRAPHS, CONFUSION MATRICES & ROC CURVES FOR ALL 6 SEGMENTS")
print("Validation: 5-Fold Stratified K-Fold (84 Subjects per Segment)")
print("=" * 95)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_master_rows = []

for seg_id in range(6):
    folder_name, display_title = SEGMENT_INFO[seg_id]
    seg_folder = BASE_PLOTS_DIR / folder_name
    seg_folder.mkdir(parents=True, exist_ok=True)
    
    df_slice = df[df['segment_id'] == seg_id].copy()
    X = df_slice.drop(columns=['subject_id', 'segment_id', 'label'])
    y = df_slice['label'].values
    
    print(f"\n{'='*30} PROCESSING {display_title.upper()} {'='*30}")
    print(f"Target Folder: {folder_name}/")
    
    models = get_models()
    seg_metrics = []
    all_preds = {}
    all_probs = {}
    
    for model_name, model in models.items():
        preds = np.zeros(len(y))
        probs = np.zeros(len(y))
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_test = X.iloc[test_idx]
            
            scaler = RobustScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_te_sc = scaler.transform(X_test)
            
            model.fit(X_tr_sc, y_train)
            preds[test_idx] = model.predict(X_te_sc)
            if hasattr(model, "predict_proba"):
                probs[test_idx] = model.predict_proba(X_te_sc)[:, 1]
            else:
                probs[test_idx] = preds[test_idx]
                
        acc = accuracy_score(y, preds) * 100
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, probs)
        
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        all_preds[model_name] = preds
        all_probs[model_name] = probs
        
        row_dict = {
            "Segment": display_title,
            "Model": model_name,
            "Accuracy (%)": round(acc, 2),
            "Precision": round(prec, 4),
            "Recall (Sensitivity)": round(rec, 4),
            "Specificity": round(spec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        }
        seg_metrics.append(row_dict)
        all_master_rows.append(row_dict)
        
    seg_df = pd.DataFrame(seg_metrics).sort_values("Accuracy (%)", ascending=False)
    seg_df.to_csv(seg_folder / "metrics.csv", index=False)
    print(seg_df[["Model", "Accuracy (%)", "F1-Score", "ROC-AUC"]].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # PLOT 1: ACCURACY BAR GRAPH FOR THIS SEGMENT
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 4.8))
    ax = seg_df.set_index("Model")["Accuracy (%)"].plot(
        kind="bar", color="#20639b", edgecolor="black", linewidth=0.8
    )
    plt.title(f"Classification Accuracy: {display_title}", fontsize=13, weight='bold', pad=12)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xlabel("Classifier", fontsize=11)
    plt.xticks(rotation=15, fontsize=10)
    plt.ylim(45, 95)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.8),
                    ha='center', va='bottom', fontsize=10, weight='bold')
                    
    plt.tight_layout()
    bar_path = seg_folder / "1_accuracy_bar.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # PLOT 2: CONFUSION MATRICES (1x5 PANEL) FOR THIS SEGMENT
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.2))
    class_labels = ['Healthy', 'Schizo']
    
    for i, (name, p_vals) in enumerate(all_preds.items()):
        cm = confusion_matrix(y, p_vals)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    annot_kws={"size": 13, "weight": "bold"},
                    xticklabels=class_labels, yticklabels=class_labels)
        axes[i].set_title(name, fontsize=11, weight='bold', pad=10)
        axes[i].set_xlabel("Predicted", fontsize=10)
        axes[i].set_ylabel("True" if i == 0 else "")
        
    plt.suptitle(f"Confusion Matrices: {display_title} (N = 84 Subjects)", 
                 fontsize=14, weight='bold', y=1.06)
    plt.tight_layout()
    cm_path = seg_folder / "2_confusion_matrices.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # PLOT 3: ROC-AUC CURVES FOR THIS SEGMENT
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8.5, 6.5))
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']
    
    for (name, pr_vals), color in zip(all_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y, pr_vals)
        auc_score = roc_auc_score(y, pr_vals)
        plt.plot(fpr, tpr, lw=2.2, color=color, label=f"{name} (AUC = {auc_score:.4f})")
        
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label="Chance Level")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title(f"ROC Curves: {display_title} (N = 84 Subjects)", fontsize=13, weight='bold')
    plt.legend(loc="lower right", fontsize=10, frameon=True, facecolor='white')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    roc_path = seg_folder / "3_roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    print(f"   ✓ Saved 3 plots in {folder_name}/")

# ==============================================================================
# SAVE MASTER COMPARISON CSV & PRINT LEADERBOARD
# ==============================================================================
master_df = pd.DataFrame(all_master_rows)
master_csv = BASE_PLOTS_DIR / "master_segment_comparison.csv"
master_df.to_csv(master_csv, index=False)

print("\n" + "=" * 95)
print("FINAL SUMMARY: PIVOT ACCURACY TABLE (TIME WINDOW vs. CLASSIFIER)")
print("=" * 95)
pivot_table = master_df.pivot(index="Segment", columns="Model", values="Accuracy (%)")
print(pivot_table.to_string())

# Find the winning segment
best_segment = master_df.groupby("Segment")["Accuracy (%)"].mean().sort_values(ascending=False).index[0]
best_avg = master_df.groupby("Segment")["Accuracy (%)"].mean().max()
top_single = master_df.sort_values("Accuracy (%)", ascending=False).iloc[0]

print("\n" + "=" * 95)
print(f"🥇 BEST OVERALL TIME WINDOW: {best_segment} (Average Accuracy: {best_avg:.2f}%)")
print(f"🏆 HIGHEST PEAK SCORE:       {top_single['Model']} on {top_single['Segment']} -> {top_single['Accuracy (%)']}% Acc | F1: {top_single['F1-Score']}")
print("=" * 95)
print(f"All individual plots and tables saved in: {BASE_PLOTS_DIR}/")
print("=" * 95)