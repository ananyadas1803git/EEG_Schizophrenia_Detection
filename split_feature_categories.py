import pandas as pd
from pathlib import Path

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Checks for both possible file names
input_file = OUTPUT_DIR / "schizophrenia_features.csv"
if not input_file.exists():
    input_file = OUTPUT_DIR / "schizophrenia.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"Could not find the master CSV file in {OUTPUT_DIR}. "
        "Please check if 'schizophrenia_features.csv' or 'schizophrenia.csv' exists."
    )

print(f"Loading master dataset from: {input_file.name}...")
df = pd.read_csv(input_file)
print(f"Loaded dataset with shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})\n")

# Metadata columns to retain in all 3 datasets
META_COLS = [c for c in ['subject_id', 'label'] if c in df.columns]

# ==============================================================================
# 1. TIME DOMAIN DATASET
# ==============================================================================
time_features = [c for c in df.columns if "_time_" in c]
time_df = df[META_COLS + time_features]
time_output = OUTPUT_DIR / "schizophrenia_time_features.csv"
time_df.to_csv(time_output, index=False)

# ==============================================================================
# 2. FREQUENCY DOMAIN DATASET
# ==============================================================================
freq_features = [c for c in df.columns if "_freq_" in c]
freq_df = df[META_COLS + freq_features]
freq_output = OUTPUT_DIR / "schizophrenia_frequency_features.csv"
freq_df.to_csv(freq_output, index=False)

# ==============================================================================
# 3. ENTROPY BASED DATASET
# ==============================================================================
entropy_features = [c for c in df.columns if "_entropy_" in c]
entropy_df = df[META_COLS + entropy_features]
entropy_output = OUTPUT_DIR / "schizophrenia_entropy_features.csv"
entropy_df.to_csv(entropy_output, index=False)

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================
print("=" * 70)
print("SUCCESSFULLY CREATED 3 CATEGORY DATASETS IN 'output/' FOLDER:")
print("=" * 70)
print(f"1. Time Domain Features:")
print(f"   • File:     {time_output.name}")
print(f"   • Shape:    {time_df.shape} (84 subjects × 160 features + metadata)")
print("-" * 70)
print(f"2. Frequency Domain Features:")
print(f"   • File:     {freq_output.name}")
print(f"   • Shape:    {freq_df.shape} (84 subjects × 208 features + metadata)")
print("-" * 70)
print(f"3. Entropy Based Features:")
print(f"   • File:     {entropy_output.name}")
print(f"   • Shape:    {entropy_df.shape} (84 subjects × 80 features + metadata)")
print("=" * 70)