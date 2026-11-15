import os
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 16 EEG Channels
CHANNELS = [
    'F7', 'F3', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4',
    'T4', 'T5', 'P3', 'Pz',
    'P4', 'T6', 'O1', 'O2'
]

SFREQ = 128             # 128 Hz sampling frequency
WINDOW_SEC = 1          # 1-second window duration
STEP_SEC = 0.5          # 0.5-second step (50% overlap)
WIN_POINTS = int(WINDOW_SEC * SFREQ)   # 128 points
STEP_POINTS = int(STEP_SEC * SFREQ)    # 64 points


# ==============================================================================
# CATEGORY 1: TIME DOMAIN (10 Features)
# ==============================================================================
def extract_time_features(x):
    N = len(x)
    d1 = np.diff(x)
    d2 = np.diff(d1)
    
    power = np.mean(x ** 2)
    rms = np.sqrt(power)
    line_length = np.sum(np.abs(d1))
    d1_mean = np.sum(np.abs(d1)) / (N - 1)
    d2_mean = np.sum(np.abs(d2)) / (N - 2) if N > 2 else 0.0
    
    var_0 = np.var(x)
    var_1 = np.var(d1)
    var_2 = np.var(d2)
    
    hjorth_act = var_0
    hjorth_mob = np.sqrt(var_1 / (var_0 + 1e-10))
    hjorth_comp = np.sqrt(var_2 / (var_1 + 1e-10)) / (hjorth_mob + 1e-10)
    
    return {
        'time_power': power,
        'time_rms': rms,
        'time_line_length': line_length,
        'time_d1': d1_mean,
        'time_d2': d2_mean,
        'time_hjorth_act': hjorth_act,
        'time_hjorth_mob': hjorth_mob,
        'time_hjorth_comp': hjorth_comp,
        'time_skew': skew(x),
        'time_kurtosis': kurtosis(x)
    }


# ==============================================================================
# CATEGORY 2: FREQUENCY & WAVELET DOMAIN (13 Features)
# ==============================================================================
def extract_frequency_features(x, sfreq=128):
    feats = {}
    N = len(x)
    
    # 1. FFT Dominant Peak Power & Dominant Frequency
    fft_vals = np.abs(np.fft.rfft(x)) ** 2
    freqs_fft = np.fft.rfftfreq(N, d=1.0 / sfreq)
    peak_idx = np.argmax(fft_vals)
    feats['freq_dominant_power'] = fft_vals[peak_idx]
    feats['freq_dominant_freq'] = freqs_fft[peak_idx]
    
    # 2. 5 Sub-Bands (Delta, Theta, Alpha, Beta, Gamma) via Welch PSD
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    freqs_w, psd = welch(x, fs=sfreq, nperseg=min(N, sfreq * 2))
    total_psd = np.sum(psd) + 1e-10
    
    for band_name, (low, high) in bands.items():
        idx = (freqs_w >= low) & (freqs_w <= high)
        feats[f'freq_band_{band_name}'] = np.sum(psd[idx]) / total_psd
        
    # 3. Clinical Biomarker Ratios
    th = feats['freq_band_theta']
    be = feats['freq_band_beta'] + 1e-10
    al = feats['freq_band_alpha']
    ga = feats['freq_band_gamma'] + 1e-10
    feats['freq_ratio_theta_beta'] = th / be
    feats['freq_ratio_theta_alpha'] = th / (al + 1e-10)
    feats['freq_ratio_slow_fast'] = (th + al) / (be + ga)
    
    # 4. Spectral Edge Frequency (SEF 95%)
    cum_power = np.cumsum(psd) / total_psd
    sef_idx = np.where(cum_power >= 0.95)[0]
    feats['freq_sef95'] = freqs_w[sef_idx[0]] if len(sef_idx) > 0 else freqs_w[-1]
    
    # 5. db8 Wavelet Energies (Approximation & Total Energy)
    max_level = min(pywt.dwt_max_level(N, 'db8'), 5)
    coeffs = pywt.wavedec(x, 'db8', level=max_level)
    approx = coeffs[0]
    details = coeffs[1:]
    feats['freq_wavelet_approx_energy'] = np.mean(approx ** 2)
    feats['freq_total_wavelet_energy'] = sum(np.mean(d ** 2) for d in details) + np.mean(approx ** 2)
    
    return feats


# ==============================================================================
# CATEGORY 3: ENTROPY & COMPLEXITY DOMAIN (5 Features)
# ==============================================================================
def sample_entropy_fast(x, m=2, r_scale=0.2):
    N = len(x)
    r = r_scale * (np.std(x) + 1e-10)
    
    def _count(m_len):
        patterns = np.array([x[i:i + m_len] for i in range(N - m_len + 1)])
        dists = np.max(np.abs(patterns[:, None, :] - patterns[None, :, :]), axis=2)
        return np.sum(dists < r) - len(patterns)
        
    B = _count(m)
    A = _count(m + 1)
    return -np.log(A / B) if (B > 0 and A > 0) else 0.0

def higuchi_fd(x, kmax=8):
    N = len(x)
    L = []
    for k in range(1, kmax + 1):
        Lk = 0.0
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            curve_len = np.sum(np.abs(np.diff(x[idx])))
            norm = (N - 1) / (len(idx) * k)
            Lk += curve_len * norm / k
        if Lk > 0:
            L.append(Lk / k)
    if len(L) < 2:
        return 0.0
    return np.polyfit(np.log(1.0 / np.arange(1, len(L) + 1)), np.log(L), 1)[0]

def extract_entropy_features(x, sfreq=128):
    feats = {}
    N = len(x)
    
    # 1. Spectral Entropy
    freqs, psd = welch(x, fs=sfreq, nperseg=min(N, sfreq * 2))
    psd_norm = psd / (np.sum(psd) + 1e-10)
    feats['entropy_spectral'] = entropy(psd_norm + 1e-10) / np.log(len(psd_norm))
    
    # 2. Shannon Entropy
    counts, _ = np.histogram(x, bins=20, density=True)
    counts = counts[counts > 0]
    feats['entropy_shannon'] = -np.sum(counts * np.log(counts + 1e-10))
    
    # 3. Permutation Entropy
    embedded = np.array([x[i:i + 3] for i in range(N - 2)])
    permutations = np.argsort(embedded, axis=1)
    _, p_counts = np.unique(permutations, axis=0, return_counts=True)
    probs = p_counts / np.sum(p_counts)
    feats['entropy_permutation'] = -np.sum(probs * np.log2(probs + 1e-10))
    
    # 4. Higuchi Fractal Dimension & 5. Sample Entropy
    feats['entropy_higuchi_fd'] = higuchi_fd(x, kmax=8)
    feats['entropy_sample'] = sample_entropy_fast(x, m=2, r_scale=0.2)
    
    return feats


# ==============================================================================
# FEATURE VECTOR BUILDER (28 features per channel * 16 = 448 features)
# ==============================================================================
def extract_window_vector(window_16ch):
    row = {}
    for ch_idx, ch_name in enumerate(CHANNELS):
        sig = window_16ch[ch_idx]
        
        # 10 Time features
        for k, v in extract_time_features(sig).items():
            row[f"{ch_name}_{k}"] = v
            
        # 13 Frequency features
        for k, v in extract_frequency_features(sig, SFREQ).items():
            row[f"{ch_name}_{k}"] = v
            
        # 5 Entropy features
        for k, v in extract_entropy_features(sig, SFREQ).items():
            row[f"{ch_name}_{k}"] = v
            
    return row


# ==============================================================================
# FILE LOADING HELPER
# ==============================================================================
def load_eeg_file(filepath):
    raw = np.loadtxt(filepath)
    if raw.size != 16 * 7680:
        raise ValueError(f"File {filepath} has {raw.size} numbers, expected 122,880.")
    return raw.reshape(16, 7680)


# ==============================================================================
# MAIN PROCESSING LOOP: EXTRACT AND AGGREGATE PER SUBJECT
# ==============================================================================
def process_all_files():
    sz_dir = DATA_DIR / "schizophrenia"
    hc_dir = DATA_DIR / "healthy"
    
    sz_files = sorted([f for f in sz_dir.rglob("*") if f.is_file() and not f.name.startswith(".")])
    hc_files = sorted([f for f in hc_dir.rglob("*") if f.is_file() and not f.name.startswith(".")])
    
    if len(sz_files) == 0 and len(hc_files) == 0:
        print("ERROR: No files found in data/schizophrenia or data/healthy.")
        return None
        
    print(f"Found {len(sz_files)} Schizophrenia subjects and {len(hc_files)} Healthy subjects.")
    
    subjects = []
    for f in sz_files:
        subjects.append((f.stem, 1, load_eeg_file(f)))
    for f in hc_files:
        subjects.append((f.stem, 0, load_eeg_file(f)))
        
    all_subject_rows = []
    print("\nExtracting 448 features (10 Time + 13 Frequency + 5 Entropy x 16 channels) per subject...")
    
    for subject_id, label, eeg_mat in tqdm(subjects, desc="Processing Subjects"):
        total_points = eeg_mat.shape[1]
        window_vectors = []
        
        # Slice into 59 windows
        for start in range(0, total_points - WIN_POINTS + 1, STEP_POINTS):
            window = eeg_mat[:, start:start + WIN_POINTS]
            feat_dict = extract_window_vector(window)
            window_vectors.append(feat_dict)
            
        # Average the 59 windows into ONE row for this subject
        df_windows = pd.DataFrame(window_vectors)
        mean_vector = df_windows.mean().to_dict()
        
        # Add metadata
        mean_vector['subject_id'] = subject_id
        mean_vector['label'] = label
        all_subject_rows.append(mean_vector)
        
    # Create final DataFrame
    df_final = pd.DataFrame(all_subject_rows)
    
    # Place subject_id and label first
    cols = ['subject_id', 'label'] + [c for c in df_final.columns if c not in ['subject_id', 'label']]
    df_final = df_final[cols]
    
    output_path = OUTPUT_DIR / "schizophrenia_features.csv"
    df_final.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print(f"SUCCESS! File saved to: {output_path}")
    print(f"Final Table Dimensions: {df_final.shape}")
    print(f"  • Rows (Subjects):       {df_final.shape[0]} (Expected: 84)")
    print(f"  • Extracted Features:    {df_final.shape[1] - 2} (Expected: 448)")
    print(f"  • Total Columns:         {df_final.shape[1]} (448 features + subject_id + label)")
    print("="*70)
    return df_final


if __name__ == "__main__":
    process_all_files()