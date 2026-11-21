import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# ==============================================================================
# PATHS - ALL 4 FEATURE SETS INCLUDED
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

DATASETS = {
    "Time Domain (160 feat)": OUTPUT_DIR / "schizophrenia_6s_time.csv",
    "Frequency Domain (208 feat)": OUTPUT_DIR / "schizophrenia_6s_frequency.csv",
    "Entropy Domain (80 feat)": OUTPUT_DIR / "schizophrenia_6s_entropy.csv",
    "Combined Features (448 feat)": OUTPUT_DIR / "schizophrenia_6s_10segments.csv"
}

# ==============================================================================
# OPTIMIZED / TUNED MODELS
# ==============================================================================
def get_optimized_models():
    # Tuned SVM (C=2.5, RBF kernel with soft probability estimation)
    tuned_svm = SVC(C=2.5, kernel='rbf', gamma='scale', probability=True, random_state=42)
    
    # Tuned Random Forest
    tuned_rf = RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_split=3, 
                                      random_state=42, n_jobs=-1)
    
    # Tuned XGBoost
    tuned_xgb = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.04, 
                              subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', 
                              random_state=42, n_jobs=-1)
    
    # Tuned ANN (Feed-forward with tanh activation and early stopping)
    tuned_ann = MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', alpha=0.01,
                              early_stopping=True, max_iter=500, random_state=42)
    
    # Soft Voting Ensemble Model
    ensemble = VotingClassifier(
        estimators=[('svm', tuned_svm), ('rf', tuned_rf), ('xgb', tuned_xgb)],
        voting='soft'
    )
    
    return {
        "Naïve Bayes": GaussianNB(),
        "Tuned SVM (RBF)": tuned_svm,
        "Tuned Random Forest": tuned_rf,
        "Tuned XGBoost": tuned_xgb,
        "Tuned ANN (MLP)": tuned_ann,
        "⭐ Ensemble (SVM+RF+XGB)": ensemble
    }

# ==============================================================================
# EVALUATION PIPELINE WITH ROBUST SCALING & SOFT PROBABILITY VOTING
# ==============================================================================
def evaluate_enhanced_pipeline(clf, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    
    segment_preds = np.zeros(len(y))
    segment_probs = np.zeros(len(y))
    
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test = X.iloc[test_idx]
        
        # RobustScaler (Median and IQR: handles voltage spikes and outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf.fit(X_train_scaled, y_train)
        
        segment_preds[test_idx] = clf.predict(X_test_scaled)
        if hasattr(clf, "predict_proba"):
            segment_probs[test_idx] = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            segment_probs[test_idx] = segment_preds[test_idx]
            
    # Segment-Level Metrics
    seg_acc = accuracy_score(y, segment_preds) * 100
    seg_f1 = f1_score(y, segment_preds, zero_division=0)
    
    # Patient-Level Diagnosis (Soft Probability Averaging across 10 segments)
    df_eval = pd.DataFrame({
        'subject_id': groups,
        'true_label': y,
        'prob_sz': segment_probs
    })
    
    patient_df = df_eval.groupby('subject_id').agg({
        'true_label': 'first',
        'prob_sz': 'mean'
    }).reset_index()
    
    patient_df['pred_label'] = (patient_df['prob_sz'] >= 0.5).astype(int)
    
    pat_true = patient_df['true_label']
    pat_pred = patient_df['pred_label']
    
    pat_acc = accuracy_score(pat_true, pat_pred) * 100
    pat_prec = precision_score(pat_true, pat_pred, zero_division=0)
    pat_rec = recall_score(pat_true, pat_pred, zero_division=0)
    pat_f1 = f1_score(pat_true, pat_pred, zero_division=0)
    
    return {
        "Patient_Acc": pat_acc,
        "Patient_Prec": pat_prec,
        "Patient_Rec": pat_rec,
        "Patient_F1": pat_f1,
        "Segment_Acc": seg_acc,
        "Segment_F1": seg_f1
    }

# ==============================================================================
# MAIN BENCHMARK LOOP
# ==============================================================================
def main():
    print("=" * 95)
    print("RUNNING BENCHMARK ACROSS ALL 4 DATASETS: TIME, FREQUENCY, ENTROPY, AND COMBINED")
    print("=" * 95)
    
    meta_cols = ['subject_id', 'segment_id', 'label']
    final_rows = []
    
    for domain_name, filepath in DATASETS.items():
        if not filepath.exists():
            print(f"Skipping {domain_name} (File not found: {filepath.name})")
            continue
            
        print(f"\n>>> PROCESSING: {domain_name.upper()} ({filepath.name})")
        df = pd.read_csv(filepath)
        
        X = df.drop(columns=[c for c in meta_cols if c in df.columns])
        y = df['label'].values
        groups = df['subject_id'].values
        
        models = get_optimized_models()
        
        for name, clf in models.items():
            print(f"  • Running {name:<25} ... ", end="", flush=True)
            res = evaluate_enhanced_pipeline(clf, X, y, groups)
            print(f"Done! [Patient Acc: {res['Patient_Acc']:.2f}% | F1: {res['Patient_F1']:.4f}]")
            
            final_rows.append({
                "Dataset": domain_name,
                "Classifier": name,
                "Patient Accuracy (%)": round(res["Patient_Acc"], 2),
                "Patient Precision": round(res["Patient_Prec"], 4),
                "Patient Recall": round(res["Patient_Rec"], 4),
                "Patient F1-Score": round(res["Patient_F1"], 4),
                "Segment Accuracy (%)": round(res["Segment_Acc"], 2),
                "Segment F1-Score": round(res["Segment_F1"], 4)
            })
            
    res_df = pd.DataFrame(final_rows)
    output_csv = OUTPUT_DIR / "improved_model_results.csv"
    res_df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 95)
    print("ALL 4 FEATURE SETS COMPLETE - UPDATED RESULTS TABLE:")
    print("=" * 95)
    print(res_df.to_string(index=False))
    print("=" * 95)
    print(f"Saved full results to: {output_csv}\n")

if __name__ == "__main__":
    main()