import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for clean saving in VS Code
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# All 5 Models
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# ==============================================================================
# PATHS - ALL 4 DATASETS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "tuning_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "Time Domain (160 feat)": OUTPUT_DIR / "schizophrenia_6s_time.csv",
    "Frequency Domain (208 feat)": OUTPUT_DIR / "schizophrenia_6s_frequency.csv",
    "Entropy Domain (80 feat)": OUTPUT_DIR / "schizophrenia_6s_entropy.csv",
    "Combined Features (448 feat)": OUTPUT_DIR / "schizophrenia_6s_10segments.csv"
}

# ==============================================================================
# DEFINE TUNING SPACES FOR ALL 5 MODELS
# ==============================================================================
def get_tuning_configs():
    return {
        "Naïve Bayes": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('nb', GaussianNB())]),
            "params": {'nb__var_smoothing': np.logspace(0, -8, 8)},
            "search_type": "grid"
        },
        "Tuned SVM (RBF)": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('svm', SVC(kernel='rbf', probability=True, random_state=42))]),
            "params": {
                'svm__C': [1.0, 2.5, 5.0, 10.0, 20.0],
                'svm__gamma': ['scale', 'auto', 0.005, 0.01, 0.03],
                'svm__class_weight': [None, 'balanced']
            },
            "search_type": "grid"
        },
        "Tuned Random Forest": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))]),
            "params": {
                'rf__n_estimators': [100, 150],
                'rf__max_depth': [6, 8, 10],
                'rf__min_samples_split': [2, 4],
                'rf__max_features': ['sqrt', 0.35]
            },
            "search_type": "random",
            "n_iter": 8
        },
        "Tuned XGBoost": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('xgb', XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1))]),
            "params": {
                'xgb__n_estimators': [100, 150],
                'xgb__max_depth': [3, 4, 5],
                'xgb__learning_rate': [0.03, 0.06],
                'xgb__subsample': [0.8, 1.0],
                'xgb__colsample_bytree': [0.75, 1.0]
            },
            "search_type": "random",
            "n_iter": 8
        },
        "Tuned ANN (MLP)": {
            "pipeline": Pipeline([('scaler', RobustScaler()), ('ann', MLPClassifier(early_stopping=True, max_iter=400, random_state=42))]),
            "params": {
                'ann__hidden_layer_sizes': [(64, 32), (128, 64), (64, 64)],
                'ann__activation': ['tanh', 'relu'],
                'ann__alpha': [0.001, 0.01, 0.05],
                'ann__learning_rate_init': [0.001, 0.005]
            },
            "search_type": "random",
            "n_iter": 8
        }
    }

# ==============================================================================
# PATIENT-LEVEL EVALUATION (Majority Probability Averaging)
# ==============================================================================
def evaluate_patient_level(best_model, X, y, groups, cv_splits):
    segment_preds = np.zeros(len(y))
    segment_probs = np.zeros(len(y))
    
    for train_idx, test_idx in cv_splits:
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test = X.iloc[test_idx]
        
        best_model.fit(X_train, y_train)
        segment_preds[test_idx] = best_model.predict(X_test)
        
        if hasattr(best_model, "predict_proba"):
            segment_probs[test_idx] = best_model.predict_proba(X_test)[:, 1]
        else:
            segment_probs[test_idx] = segment_preds[test_idx]
            
    seg_acc = accuracy_score(y, segment_preds) * 100
    
    # Soft probability voting per patient
    df_eval = pd.DataFrame({'subject_id': groups, 'true_label': y, 'prob': segment_probs})
    patient_df = df_eval.groupby('subject_id').agg({'true_label': 'first', 'prob': 'mean'}).reset_index()
    patient_df['pred_label'] = (patient_df['prob'] >= 0.50).astype(int)
    
    pat_true = patient_df['true_label']
    pat_pred = patient_df['pred_label']
    
    pat_acc = accuracy_score(pat_true, pat_pred) * 100
    pat_prec = precision_score(pat_true, pat_pred, zero_division=0)
    pat_rec = recall_score(pat_true, pat_pred, zero_division=0)
    pat_f1 = f1_score(pat_true, pat_pred, zero_division=0)
    
    return pat_acc, pat_prec, pat_rec, pat_f1, seg_acc

# ==============================================================================
# MAIN MULTI-DATASET TUNING LOOP
# ==============================================================================
def main():
    print("=" * 95)
    print("TUNING ALL 5 MODELS ACROSS ALL 4 DATASETS (TIME, FREQ, ENTROPY, COMBINED)")
    print("Cross-Validation: 5-Fold GroupKFold (Strict Subject Isolation)")
    print("=" * 95)
    
    meta_cols = ['subject_id', 'segment_id', 'label']
    all_results = []
    
    for domain_name, filepath in DATASETS.items():
        if not filepath.exists():
            print(f"Skipping {domain_name} (file not found)")
            continue
            
        print(f"\n" + "#" * 60)
        print(f"### TUNING ON: {domain_name.upper()}")
        print("#" * 60)
        
        df = pd.read_csv(filepath)
        X = df.drop(columns=[c for c in meta_cols if c in df.columns])
        y = df['label'].values
        groups = df['subject_id'].values
        
        configs = get_tuning_configs()
        gkf = GroupKFold(n_splits=5)
        # CRITICAL FIX: Convert generator to a list so joblib can pickle it across CPU cores
        cv_splits = list(gkf.split(X, y, groups=groups))
        
        for model_name, cfg in configs.items():
            print(f"  • Tuning {model_name:<22} ... ", end="", flush=True)
            
            if cfg["search_type"] == "grid":
                searcher = GridSearchCV(
                    estimator=cfg["pipeline"],
                    param_grid=cfg["params"],
                    cv=cv_splits,
                    scoring='f1',
                    n_jobs=-1
                )
            else:
                searcher = RandomizedSearchCV(
                    estimator=cfg["pipeline"],
                    param_distributions=cfg["params"],
                    n_iter=cfg["n_iter"],
                    cv=cv_splits,
                    scoring='f1',
                    random_state=42,
                    n_jobs=-1
                )
                
            searcher.fit(X, y)
            best_model = searcher.best_estimator_
            
            # Evaluate on patient diagnosis
            pat_acc, pat_prec, pat_rec, pat_f1, seg_acc = evaluate_patient_level(best_model, X, y, groups, cv_splits)
            print(f"Done! [Pat Acc: {pat_acc:.2f}% | F1: {pat_f1:.4f} | Recall: {pat_rec:.4f}]")
            
            all_results.append({
                "Feature Set": domain_name,
                "Classifier": model_name,
                "Patient Accuracy (%)": round(pat_acc, 2),
                "Patient Precision": round(pat_prec, 4),
                "Patient Recall": round(pat_rec, 4),
                "Patient F1-Score": round(pat_f1, 4),
                "Segment Accuracy (%)": round(seg_acc, 2)
            })

    # ==============================================================================
    # OUTPUTS & SUMMARY TABLE
    # ==============================================================================
    results_df = pd.DataFrame(all_results)
    
    csv_path = OUTPUT_DIR / "all_datasets_all_models_tuned.csv"
    results_df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 95)
    print("FINAL MASTER LEADERBOARD (ALL 4 DATASETS × ALL 5 MODELS FULLY TUNED):")
    print("=" * 95)
    print(results_df.to_string(index=False))
    print("=" * 95)
    print(f"Saved complete tuned results to: {csv_path.name}")

    # ==============================================================================
    # 4-PANEL VISUALIZATION CHART
    # ==============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    metric_configs = [
        ("Patient Accuracy (%)", "Tuned Patient Accuracy (%)", axes[0, 0]),
        ("Patient Precision", "Tuned Patient Precision", axes[0, 1]),
        ("Patient Recall", "Tuned Patient Recall (Sensitivity)", axes[1, 0]),
        ("Patient F1-Score", "Tuned Patient F1-Score", axes[1, 1])
    ]
    
    for metric_col, title, ax in metric_configs:
        sns.barplot(
            data=results_df,
            x="Feature Set",
            y=metric_col,
            hue="Classifier",
            palette="tab10",
            ax=ax
        )
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel("Feature Set", fontsize=10)
        ax.set_ylabel(metric_col, fontsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        if metric_col == "Patient Accuracy (%)":
            ax.set_ylim(50, 105)
        else:
            ax.set_ylim(0.45, 1.05)
        ax.legend(title="Tuned Model", loc='lower right', fontsize=8)
        
    plt.tight_layout()
    chart_path = PLOTS_DIR / "all_datasets_all_models_tuned_chart.png"
    plt.savefig(chart_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved comprehensive 4-metric comparison chart to: {chart_path.name}\n")


if __name__ == "__main__":
    main()