from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = [
    'F7', 'F3', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4',
    'T4', 'T5', 'P3', 'Pz',
    'P4', 'T6', 'O1', 'O2'
]
SFREQ = 128
WINDOW_SEC = 10                  # 10-second window
WIN_POINTS = WINDOW_SEC * SFREQ  # 1280 points
NUM_SEGMENTS = 6                 # 6 segments per subject (6 x 10s = 60s)

# ==============================================================================
# FEATURE EXTRACTION FUNCTIONS (10 Time, 13 Frequency, 5 Entropy = 28 per ch)
# ==============================================================================
def extract_time_features(x):
    d1 = np.diff(x)
    d2 = np.diff(d1)
    power = np.mean(x ** 2)
    var_0, var_1, var_2 = np.var(x), np.var(d1), np.var(d2)
    mob = np.sqrt(var_1 / (var_0 + 1e-10))
    comp = np.sqrt(var_2 / (var_1 + 1e-10)) / (mob + 1e-10)
    
    return {
        'time_power': power,
        'time_rms': np.sqrt(power),
        'time_line_length': np.sum(np.abs(d1)),
        'time_d1': np.mean(np.abs(d1)),
        'time_d2': np.mean(np.abs(d2)),
        'time_hjorth_act': var_0,
        'time_hjorth_mob': mob,
        'time_hjorth_comp': comp,
        'time_skew': skew(x),
        'time_kurtosis': kurtosis(x)
    }

def extract_frequency_features(x, sfreq=128):
    freqs, psd = welch(x, fs=sfreq, nperseg=min(len(x), 256))
    total_power = np.sum(psd) + 1e-10
    
    def band_pwr(low, high):
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.sum(psd[idx])
        
    d_p = band_pwr(0.5, 4.0)
    t_p = band_pwr(4.0, 8.0)
    a_p = band_pwr(8.0, 13.0)
    b_p = band_pwr(13.0, 30.0)
    g_p = band_pwr(30.0, 45.0)
    
    psd_norm = psd / total_power
    spec_ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    peak_idx = np.argmax(psd)
    
    return {
        'freq_peak_power': psd[peak_idx],
        'freq_peak_freq': freqs[peak_idx],
        'freq_delta_pwr': d_p,
        'freq_theta_pwr': t_p,
        'freq_alpha_pwr': a_p,
        'freq_beta_pwr': b_p,
        'freq_gamma_pwr': g_p,
        'freq_rel_delta': d_p / total_power,
        'freq_rel_theta': t_p / total_power,
        'freq_rel_alpha': a_p / total_power,
        'freq_rel_beta': b_p / total_power,
        'freq_rel_gamma': g_p / total_power,
        'freq_spectral_entropy': spec_ent
    }

def extract_entropy_features(x):
    counts, _ = np.histogram(x, bins=30, density=True)
    p = counts / (np.sum(counts) + 1e-10)
    p = p[p > 0]
    shannon = -np.sum(p * np.log2(p + 1e-10))
    
    sub = x[::2]  # Subsample for speed
    N = len(sub)
    embed = np.array([sub[i:i + 3] for i in range(N - 2)])
    perms = np.argsort(embed, axis=1)
    _, p_counts = np.unique(perms, axis=0, return_counts=True)
    p_probs = p_counts / np.sum(p_counts)
    perm_ent = -np.sum(p_probs * np.log2(p_probs + 1e-10))
    
    # Complexity and dynamic variance
    r = 0.2 * np.std(x)
    comp_proxy = np.std(np.diff(x)) / (r + 1e-10)
    
    # Approximate & Sample entropy proxies
    diffs = np.abs(np.diff(x))
    sa_proxy = -np.log(np.mean(diffs < r) + 1e-10)
    fuzzy_proxy = -np.log(np.mean(np.exp(-(diffs / (r + 1e-10))**2)) + 1e-10)
    
    return {
        'entropy_shannon': shannon,
        'entropy_permutation': perm_ent,
        'entropy_comp_proxy': comp_proxy,
        'entropy_sample_proxy': sa_proxy,
        'entropy_fuzzy_proxy': fuzzy_proxy
    }

# ==============================================================================
# MAIN SEGMENTATION LOOP
# ==============================================================================
def main():
    sz_files = sorted([f for f in (DATA_DIR / "schizophrenia").rglob("*") if f.is_file() and not f.name.startswith(".")])
    hc_files = sorted([f for f in (DATA_DIR / "healthy").rglob("*") if f.is_file() and not f.name.startswith(".")])
    
    print(f"Loading {len(sz_files)} Schizophrenia and {len(hc_files)} Healthy subjects...")
    subjects = [(f.stem, 1, np.loadtxt(f).reshape(16, 7680)) for f in sz_files] + \
               [(f.stem, 0, np.loadtxt(f).reshape(16, 7680)) for f in hc_files]
               
    all_rows = []
    print("\nExtracting 10s non-overlapping segments (6 segments per subject)...")
    
    for sub_id, label, eeg_mat in tqdm(subjects, desc="Processing 10s Segments"):
        for seg_idx in range(NUM_SEGMENTS):
            start = seg_idx * WIN_POINTS
            end = start + WIN_POINTS
            segment = eeg_mat[:, start:end]
            
            row = {'subject_id': sub_id, 'segment_id': seg_idx, 'label': label}
            
            for ch_idx, ch_name in enumerate(CHANNELS):
                sig = segment[ch_idx]
                for k, v in extract_time_features(sig).items():
                    row[f"{ch_name}_{k}"] = v
                for k, v in extract_frequency_features(sig, SFREQ).items():
                    row[f"{ch_name}_{k}"] = v
                for k, v in extract_entropy_features(sig).items():
                    row[f"{ch_name}_{k}"] = v
                    
            all_rows.append(row)
            
    df_10s = pd.DataFrame(all_rows)
    save_path = OUTPUT_DIR / "schizophrenia_10s_6segments.csv"
    df_10s.to_csv(save_path, index=False)
    
    print("\n" + "=" * 80)
    print(f"SUCCESS! 10-second segmented dataset created: {save_path.name}")
    print(f"Dimensions: {df_10s.shape} (Expected: 504 rows, 451 columns)")
    print(f"  • Total Segments: {df_10s.shape[0]} (84 subjects x 6 segments)")
    print(f"  • Features:       {df_10s.shape[1] - 3} (16 channels x 28 features = 448)")
    print("=" * 80)

if __name__ == "__main__":
    main()