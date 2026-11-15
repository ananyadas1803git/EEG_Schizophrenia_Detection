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

SFREQ = 128                          # 128 Hz sampling frequency
WINDOW_SEC = 6                       # 6-second window duration
WIN_POINTS = WINDOW_SEC * SFREQ      # 6 * 128 = 768 points per segment
STEP_POINTS = WIN_POINTS             # Non-overlapping (10 segments per 60s)
TOTAL_POINTS = 7680                  # 60s * 128 Hz


# ==============================================================================
# CATEGORY 1: TIME DOMAIN (10 Features per channel)
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
# CATEGORY 2: FREQUENCY & WAVELET DOMAIN (13 Features per channel)
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
    
    # 2. 5 Classical Sub-Bands (via Welch PSD)
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
    
    # 5. db8 Wavelet Energies
    max_level = min(pywt.dwt_max_level(N, 'db8'), 5)
    coeffs = pywt.wavedec(x, 'db8', level=max_level)
    approx = coeffs[0]
    details = coeffs[1:]
    feats['freq_wavelet_approx_energy'] = np.mean(approx ** 2)
    feats['freq_total_wavelet_energy'] = sum(np.mean(d ** 2) for d in details) + np.mean(approx ** 2)
    
    return feats


# ==============================================================================
# CATEGORY 3: ENTROPY & COMPLEXITY DOMAIN (5 Features per channel)
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
# SEGMENT FEATURE VECTOR (28 features per channel * 16 channels = 448 features)
# ==============================================================================
def extract_segment_features(segment_16ch):
    row = {}
    for ch_idx, ch_name in enumerate(CHANNELS):
        sig = segment_16ch[ch_idx]
        
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
    if raw.size != 16 * TOTAL_POINTS:
        raise ValueError(f"File {filepath} has {raw.size} values, expected {16 * TOTAL_POINTS}.")
    return raw.reshape(16, TOTAL_POINTS)


# ==============================================================================
# MAIN PROCESSING: 10 SEGMENTS OF 6 SECONDS PER SUBJECT
# ==============================================================================
def process_6s_segmentation():
    sz_dir = DATA_DIR / "schizophrenia"
    hc_dir = DATA_DIR / "healthy"
    
    sz_files = sorted([f for f in sz_dir.rglob("*") if f.is_file() and not f.name.startswith(".")])
    hc_files = sorted([f for f in hc_dir.rglob("*") if f.is_file() and not f.name.startswith(".")])
    
    if len(sz_files) == 0 and len(hc_files) == 0:
        raise FileNotFoundError("No files found in data/schizophrenia or data/healthy.")
        
    print(f"Found {len(sz_files)} Schizophrenia subjects and {len(hc_files)} Healthy subjects.")
    
    subjects = []
    for f in sz_files:
        subjects.append((f.stem, 1, load_eeg_file(f)))
    for f in hc_files:
        subjects.append((f.stem, 0, load_eeg_file(f)))
        
    all_segment_rows = []
    print(f"\nSegmenting into 10 segments of 6 seconds (768 points) per subject...")
    
    for subject_id, label, eeg_mat in tqdm(subjects, desc="Processing Subjects"):
        # Slice into exactly 10 non-overlapping segments of 6 seconds each
        segment_id = 0
        for start in range(0, TOTAL_POINTS - WIN_POINTS + 1, STEP_POINTS):
            segment = eeg_mat[:, start:start + WIN_POINTS]
            feat_dict = extract_segment_features(segment)
            
            # Metadata tags
            feat_dict['subject_id'] = subject_id
            feat_dict['segment_id'] = segment_id
            feat_dict['label'] = label
            
            all_segment_rows.append(feat_dict)
            segment_id += 1
            
    # 1. Create Master Segmented DataFrame
    df_master = pd.DataFrame(all_segment_rows)
    
    # Place metadata columns first
    meta_cols = ['subject_id', 'segment_id', 'label']
    feature_cols = [c for c in df_master.columns if c not in meta_cols]
    df_master = df_master[meta_cols + feature_cols]
    
    # Save Master File
    master_path = OUTPUT_DIR / "schizophrenia_6s_10segments.csv"
    df_master.to_csv(master_path, index=False)
    
    # ==========================================================================
    # 2. SPLIT INTO THE 3 CATEGORY DATASETS
    # ==========================================================================
    # Time Domain Dataset
    time_cols = [c for c in feature_cols if "_time_" in c]
    df_time = df_master[meta_cols + time_cols]
    time_path = OUTPUT_DIR / "schizophrenia_6s_time.csv"
    df_time.to_csv(time_path, index=False)
    
    # Frequency Domain Dataset
    freq_cols = [c for c in feature_cols if "_freq_" in c]
    df_freq = df_master[meta_cols + freq_cols]
    freq_path = OUTPUT_DIR / "schizophrenia_6s_frequency.csv"
    df_freq.to_csv(freq_path, index=False)
    
    # Entropy Based Dataset
    entropy_cols = [c for c in feature_cols if "_entropy_" in c]
    df_entropy = df_master[meta_cols + entropy_cols]
    entropy_path = OUTPUT_DIR / "schizophrenia_6s_entropy.csv"
    df_entropy.to_csv(entropy_path, index=False)
    
    # ==========================================================================
    # SUMMARY REPORT
    # ==========================================================================
    print("\n" + "="*75)
    print("SUCCESS! 6-SECOND SEGMENTATION COMPLETED (84 subjects × 10 segments = 840 rows)")
    print("="*75)
    print(f"1. Master Segmented File (All Features):")
    print(f"   • Path:    {master_path.name}")
    print(f"   • Shape:   {df_master.shape} (840 rows × 448 features + 3 metadata)")
    print("-" * 75)
    print(f"2. Time Domain Segments:")
    print(f"   • Path:    {time_path.name}")
    print(f"   • Shape:   {df_time.shape} (840 rows × 160 features + 3 metadata)")
    print("-" * 75)
    print(f"3. Frequency Domain Segments:")
    print(f"   • Path:    {freq_path.name}")
    print(f"   • Shape:   {df_freq.shape} (840 rows × 208 features + 3 metadata)")
    print("-" * 75)
    print(f"4. Entropy Based Segments:")
    print(f"   • Path:    {entropy_path.name}")
    print(f"   • Shape:   {df_entropy.shape} (840 rows × 80 features + 3 metadata)")
    print("="*75)


if __name__ == "__main__":
    process_6s_segmentation()