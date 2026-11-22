from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import skew, kurtosis
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

# The 5 Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "band_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = [
    'F7', 'F3', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4',
    'T4', 'T5', 'P3', 'Pz',
    'P4', 'T6', 'O1', 'O2'
]
SFREQ = 128

BANDS = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0)
}

# ==============================================================================
# 1. BUTTERWORTH ZERO-PHASE BANDPASS FILTER
# ==============================================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=-1)

# ==============================================================================
# 2. FEATURE EXTRACTION: TIME (T), FREQUENCY (F), ENTROPY (E)
# ==============================================================================
def extract_time_domain(sig):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    var_0 = np.var(sig)
    var_1 = np.var(d1)
    var_2 = np.var(d2)
    power = np.mean(sig ** 2)
    mob = np.sqrt(var_1 / (var_0 + 1e-10))
    comp = np.sqrt(var_2 / (var_1 + 1e-10)) / (mob + 1e-10)
    
    return {
        'mean': np.mean(sig),
        'power': power,
        'rms': np.sqrt(power),
        'line_length': np.sum(np.abs(d1)),
        'hjorth_act': var_0,
        'hjorth_mob': mob,
        'hjorth_comp': comp,
        'skew': skew(sig),
        'kurtosis': kurtosis(sig)
    }

def extract_frequency_domain(sig, fs):
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), 256))
    peak_idx = np.argmax(psd)
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spec_ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    return {
        'peak_power': psd[peak_idx],
        'peak_freq': freqs[peak_idx],
        'mean_power': np.mean(psd),
        'spec_entropy': spec_ent
    }

def extract_entropy_domain(sig):
    counts, _ = np.histogram(sig, bins=30, density=True)
    p = counts / (np.sum(counts) + 1e-10)
    p = p[p > 0]
    shannon = -np.sum(p * np.log2(p + 1e-10))
    
    sub = sig[::4] if len(sig) > 2000 else sig
    N = len(sub)
    embed = np.array([sub[i:i + 3] for i in range(N - 2)])
    perms = np.argsort(embed, axis=1)
    _, p_counts = np.unique(perms, axis=0, return_counts=True)
    p_probs = p_counts / np.sum(p_counts)
    perm_ent = -np.sum(p_probs * np.log2(p_probs + 1e-10))
    
    r = 0.2 * np.std(sub)
    comp_proxy = np.std(np.diff(sub)) / (r + 1e-10)
    
    return {
        'shannon_ent': shannon,
        'perm_ent': perm_ent,
        'comp_proxy': comp_proxy
    }

# ==============================================================================
# 3. LOAD RAW 60s SIGNALS IF CSVS NOT FOUND
# ==============================================================================
def load_and_decompose_eeg():
    sz_files = sorted([f for f in (DATA_DIR / "schizophrenia").rglob("*") if f.is_file() and not f.name.startswith(".")])
    hc_files = sorted([f for f in (DATA_DIR / "healthy").rglob("*") if f.is_file() and not f.name.startswith(".")])
    
    subjects = []
    for f in sz_files:
        subjects.append((f.stem, 1, np.loadtxt(f).reshape(16, 7680)))
    for f in hc_files:
        subjects.append((f.stem, 0, np.loadtxt(f).reshape(16, 7680)))
        
    band_datasets = {b: [] for b in BANDS.keys()}
    
    for sub_id, label, raw_16ch in tqdm(subjects, desc="Decomposing & Extracting Bands"):
        for band_name, (low, high) in BANDS.items():
            filtered = butter_bandpass_filter(raw_16ch, low, high, SFREQ, order=4)
            row = {'subject_id': sub_id, 'label': label}
            
            for ch_idx, ch_name in enumerate(CHANNELS):
                sig = filtered[ch_idx]
                for k, v in extract_time_domain(sig).items():
                    row[f"{ch_name}_T_{k}"] = v
                for k, v in extract_frequency_domain(sig, SFREQ).items():
                    row[f"{ch_name}_F_{k}"] = v
                for k, v in extract_entropy_domain(sig).items():
                    row[f"{ch_name}_E_{k}"] = v
                    
            band_datasets[band_name].append(row)
            
    dfs = {}
    for band_name, rows in band_datasets.items():
        df = pd.DataFrame(rows)
        save_path = OUTPUT_DIR / f"schizophrenia_60s_{band_name.lower()}_band.csv"
        df.to_csv(save_path, index=False)
        dfs[band_name] = df
        
    return dfs

# ==============================================================================
# 4. HYPERPARAMETER TUNING CONFIGURATION FOR 5 MODELS
# ==============================================================================
def get_tuning_configs():
    return {
        "Naïve Bayes": {
            "model": GaussianNB(),
            "params": {'var_smoothing': [1e-9, 1e-6, 1e-4, 1e-2, 1e-1]}
        },
        "Tuned SVM (RBF)": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                'C': [1.0, 2.5, 3.5, 5.0, 10.0],
                'gamma': ['scale', 0.01, 0.05],
                'kernel': ['rbf']
            }
        },
        "Tuned Random Forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                'n_estimators': [100, 150],
                'max_depth': [4, 6, 8, 10],
                'min_samples_split': [2, 4]
            }
        },
        "Tuned XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
            "params": {
                'n_estimators': [100, 150],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.02, 0.04, 0.08],
                'subsample': [0.8, 1.0]
            }
        },
        "Tuned ANN (MLP)": {
            "model": MLPClassifier(early_stopping=True, max_iter=500, random_state=42),
            "params": {
                'hidden_layer_sizes': [(32,), (64, 32)],
                'activation': ['tanh', 'relu'],
                'alpha': [0.001, 0.01, 0.1]
            }
        }
    }

# ==============================================================================
# 5. DOMAIN-ISOLATED SELECTION + ACTIVE GRIDSEARCH
# ==============================================================================
def evaluate_band_isolated_with_tuning(df, band_name, k_per_domain=8):
    y = df['label'].values
    
    # 1. Isolate the 3 domains
    cols_T = [c for c in df.columns if "_T_" in c]
    cols_F = [c for c in df.columns if "_F_" in c]
    cols_E = [c for c in df.columns if "_E_" in c]
    
    df_T = df[cols_T]
    df_F = df[cols_F]
    df_E = df[cols_E]
    
    # Global top feature identification for display
    sel_T_glob = SelectKBest(score_func=mutual_info_classif, k=k_per_domain).fit(df_T, y)
    sel_F_glob = SelectKBest(score_func=mutual_info_classif, k=k_per_domain).fit(df_F, y)
    sel_E_glob = SelectKBest(score_func=mutual_info_classif, k=k_per_domain).fit(df_E, y)
    
    best_T_names = df_T.columns[sel_T_glob.get_support()].tolist()
    best_F_names = df_F.columns[sel_F_glob.get_support()].tolist()
    best_E_names = df_E.columns[sel_E_glob.get_support()].tolist()
    
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    tuning_configs = get_tuning_configs()
    results = []
    
    for model_name, cfg in tuning_configs.items():
        print(f"  • Tuning {model_name:<22} on 24 Domain-Isolated Features ... ", end="", flush=True)
        
        preds = np.zeros(len(y))
        probs = np.zeros(len(y))
        
        for train_idx, test_idx in outer_cv.split(df, y):
            # Scale inside outer fold
            scaler_T = RobustScaler()
            scaler_F = RobustScaler()
            scaler_E = RobustScaler()
            
            T_tr = scaler_T.fit_transform(df_T.iloc[train_idx])
            T_te = scaler_T.transform(df_T.iloc[test_idx])
            
            F_tr = scaler_F.fit_transform(df_F.iloc[train_idx])
            F_te = scaler_F.transform(df_F.iloc[test_idx])
            
            E_tr = scaler_E.fit_transform(df_E.iloc[train_idx])
            E_te = scaler_E.transform(df_E.iloc[test_idx])
            
            y_tr = y[train_idx]
            
            # Select Best Features INDEPENDENTLY from each domain
            sel_T = SelectKBest(score_func=f_classif, k=k_per_domain)
            T_tr_best = sel_T.fit_transform(T_tr, y_tr)
            T_te_best = sel_T.transform(T_te)
            
            sel_F = SelectKBest(score_func=f_classif, k=k_per_domain)
            F_tr_best = sel_F.fit_transform(F_tr, y_tr)
            F_te_best = sel_F.transform(F_te)
            
            sel_E = SelectKBest(score_func=f_classif, k=k_per_domain)
            E_tr_best = sel_E.fit_transform(E_tr, y_tr)
            E_te_best = sel_E.transform(E_te)
            
            # Concatenate into 24 balanced features (8T + 8F + 8E)
            X_tr_best = np.hstack([T_tr_best, F_tr_best, E_tr_best])
            X_te_best = np.hstack([T_te_best, F_te_best, E_te_best])
            
            # Inner GridSearchCV on the 24 balanced features
            searcher = GridSearchCV(
                estimator=cfg["model"],
                param_grid=cfg["params"],
                cv=inner_cv,
                scoring='f1',
                n_jobs=-1
            )
            searcher.fit(X_tr_best, y_tr)
            best_model = searcher.best_estimator_
            
            # Predict on outer test fold
            preds[test_idx] = best_model.predict(X_te_best)
            if hasattr(best_model, "predict_proba"):
                probs[test_idx] = best_model.predict_proba(X_te_best)[:, 1]
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
        
        print(f"Done! [Acc: {acc:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}]")
        
        results.append({
            "Frequency Band": band_name,
            "Classifier": model_name,
            "Accuracy (%)": round(acc, 2),
            "Precision": round(prec, 4),
            "Recall (Sensitivity)": round(rec, 4),
            "Specificity": round(spec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Features Used": f"{k_per_domain}T + {k_per_domain}F + {k_per_domain}E = {k_per_domain * 3}"
        })
        
    return pd.DataFrame(results), best_T_names, best_F_names, best_E_names

# ==============================================================================
# 6. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    print("=" * 95)
    print("DOMAIN-ISOLATED FEATURE SELECTION WITH ACTIVE HYPERPARAMETER TUNING")
    print("Strategy: 8 Time + 8 Frequency + 8 Entropy = 24 Balanced Features -> GridSearchCV")
    print("=" * 95)
    
    # Load existing band CSVs
    band_dfs = {}
    need_extract = False
    for b in BANDS.keys():
        f = OUTPUT_DIR / f"schizophrenia_60s_{b.lower()}_band.csv"
        if f.exists():
            band_dfs[b] = pd.read_csv(f)
        else:
            need_extract = True
            break
            
    if need_extract:
        band_dfs = load_and_decompose_eeg()
        
    all_band_results = []
    
    for band_name, df in band_dfs.items():
        print(f"\n{'='*35} {band_name.upper()} BAND PIPELINE {'='*35}")
        print(f"Selecting Top 8 from Time, Top 8 from Frequency, and Top 8 from Entropy...")
        
        res_df, best_T, best_F, best_E = evaluate_band_isolated_with_tuning(df, band_name, k_per_domain=8)
        all_band_results.append(res_df)
        
        print("\n" + res_df.to_string(index=False))
        print(f"\nTop 3 Features in Time (T):      {best_T[:3]}")
        print(f"Top 3 Features in Frequency (F): {best_F[:3]}")
        print(f"Top 3 Features in Entropy (E):   {best_E[:3]}")
        
    final_table = pd.concat(all_band_results, ignore_index=True)
    summary_csv = OUTPUT_DIR / "band_domain_isolated_tuned_results.csv"
    final_table.to_csv(summary_csv, index=False)
    
    print("\n" + "=" * 95)
    print("FINAL UNIFIED LEADERBOARD (DOMAIN-ISOLATED SELECTION + TUNING):")
    print("=" * 95)
    print(final_table.to_string(index=False))
    print("=" * 95)
    print(f"Saved results table to: {summary_csv.name}\n")
    
    # Plot Comparison Chart
    plt.figure(figsize=(11, 5))
    sns.barplot(data=final_table, x="Frequency Band", y="Accuracy (%)", hue="Classifier", palette="tab10")
    plt.title("Domain-Isolated Tuned Accuracy (8T + 8F + 8E Features)", fontsize=13, weight='bold')
    plt.ylim(50, 100)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xlabel("Frequency Band", fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plot_path = PLOTS_DIR / "band_domain_isolated_tuned_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved comparison plot to: {plot_path.name}")
    print("=" * 95)

if __name__ == "__main__":
    main()