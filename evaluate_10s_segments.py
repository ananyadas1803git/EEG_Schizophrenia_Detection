from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "segmented_10s_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

input_file = OUTPUT_DIR / "schizophrenia_10s_6segments.csv"
if not input_file.exists():
    raise FileNotFoundError("Run segment_10s_features.py first!")

print(f"Loading 10-second segmented dataset: {input_file.name}...")
df = pd.read_csv(input_file)
print(f"Shape: {df.shape} (Total 504 segments across {df.shape[1] - 3} features)\n")

X = df.drop(columns=['subject_id', 'segment_id', 'label'])
y = df['label'].values
groups = df['subject_id'].values

# The 5 Classifiers
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

gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X, y, groups=groups))

results = []
all_preds = {}
all_probs = {}

print("=" * 95)
print("EVALUATING 5 MODELS ON 10-SECOND SEGMENTS (504 TOTAL SEGMENTS)")
print("Validation: 5-Fold GroupKFold (Subject-Isolated, 0% Data Leakage)")
print("=" * 95)

for model_name, model in models.items():
    print(f"• Evaluating {model_name:<22} ... ", end="", flush=True)
    seg_preds = np.zeros(len(y))
    seg_probs = np.zeros(len(y))
    
    for train_idx, test_idx in cv_splits:
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test = X.iloc[test_idx]
        
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)
        
        model.fit(X_tr_sc, y_train)
        seg_preds[test_idx] = model.predict(X_te_sc)
        if hasattr(model, "predict_proba"):
            seg_probs[test_idx] = model.predict_proba(X_te_sc)[:, 1]
        else:
            seg_probs[test_idx] = seg_preds[test_idx]
            
    # Pure Segment-Level Metrics (504 samples)
    acc = accuracy_score(y, seg_preds) * 100
    prec = precision_score(y, seg_preds, zero_division=0)
    rec = recall_score(y, seg_preds, zero_division=0)
    f1 = f1_score(y, seg_preds, zero_division=0)
    auc = roc_auc_score(y, seg_probs)
    
    cm = confusion_matrix(y, seg_preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Patient-Level Diagnosis (Soft vote across 6 segments per subject)
    df_eval = pd.DataFrame({'subject_id': groups, 'true_label': y, 'prob': seg_probs})
    pat_df = df_eval.groupby('subject_id').agg({'true_label': 'first', 'prob': 'mean'}).reset_index()
    pat_preds = (pat_df['prob'] >= 0.5).astype(int)
    pat_acc = accuracy_score(pat_df['true_label'], pat_preds) * 100
    
    all_preds[model_name] = seg_preds
    all_probs[model_name] = seg_probs
    
    print(f"Done! [Segment Acc: {acc:.2f}% | Patient Acc: {pat_acc:.2f}% | F1: {f1:.4f}]")
    
    results.append({
        "Model": model_name,
        "Segment Accuracy (%)": round(acc, 2),
        "Patient Accuracy (%)": round(pat_acc, 2),
        "Segment Precision": round(prec, 4),
        "Segment Recall (Sensitivity)": round(rec, 4),
        "Segment Specificity": round(spec, 4),
        "Segment F1-Score": round(f1, 4),
        "Segment ROC-AUC": round(auc, 4)
    })

res_df = pd.DataFrame(results).sort_values("Segment Accuracy (%)", ascending=False)
save_csv = OUTPUT_DIR / "segmented_10s_metrics.csv"
res_df.to_csv(save_csv, index=False)

print("\n" + "=" * 95)
print("FINAL RESULTS TABLE FOR 10-SECOND WINDOWS (6 SEGMENTS PER SUBJECT):")
print("=" * 95)
print(res_df.to_string(index=False))
print("=" * 95)
print(f"Saved results table to: {save_csv.name}\n")