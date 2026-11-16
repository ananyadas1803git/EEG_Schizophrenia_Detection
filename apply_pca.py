from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for clean saving in VS Code
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "pca_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

input_file = OUTPUT_DIR / "schizophrenia_6s_10segments.csv"
if not input_file.exists():
    raise FileNotFoundError(f"Cannot find {input_file}. Make sure your segmented file exists!")

print(f"Loading data from: {input_file.name}...")
df = pd.read_csv(input_file)
print(f"Loaded shape: {df.shape} (840 rows x {df.shape[1]} columns)\n")

# Separate Metadata & Features
meta_cols = ['subject_id', 'segment_id', 'label']
X = df.drop(columns=meta_cols)
y = df['label'].values
feature_names = X.columns.tolist()

# ==============================================================================
# 1. STANDARDIZATION (Mandatory before PCA)
# ==============================================================================
# PCA requires all features to have mean=0 and variance=1 so large-scale features
# (like total power) don't overpower small-scale features (like entropy)
print("Standardizing 448 features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 2. FIT PCA (Full Spectrum)
# ==============================================================================
print("Fitting Principal Component Analysis...")
pca_full = PCA()
pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100

# Find components needed for 80%, 90%, and 95% variance
n_80 = np.argmax(cum_var >= 80) + 1
n_90 = np.argmax(cum_var >= 90) + 1
n_95 = np.argmax(cum_var >= 95) + 1

print("-" * 65)
print(f"Components needed for  80% variance: {n_80} PCs (out of 448)")
print(f"Components needed for  90% variance: {n_90} PCs (out of 448)")
print(f"Components needed for  95% variance: {n_95} PCs (out of 448)")
print("-" * 65)

# ==============================================================================
# 3. GENERATE VISUALIZATIONS
# ==============================================================================
print("Generating PCA plots in 'output/pca_plots/'...")

# ------------------------------------------------------------------------------
# PLOT 1: Cumulative Explained Variance (Scree Plot)
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, color='#1f77b4', lw=2.5, label='Cumulative Variance')
plt.axhline(y=90, color='r', linestyle='--', label=f'90% Variance ({n_90} PCs)')
plt.axhline(y=95, color='g', linestyle='--', label=f'95% Variance ({n_95} PCs)')
plt.title("PCA Scree Plot: Cumulative Explained Variance", fontsize=13, weight='bold')
plt.xlabel("Number of Principal Components", fontsize=11)
plt.ylabel("Explained Variance (%)", fontsize=11)
plt.xlim(1, 150)  # Zoom in on first 150 components
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right', fontsize=10)
plt.savefig(PLOTS_DIR / "1_pca_scree_plot.png", bbox_inches='tight', dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# PLOT 2: 2D Projection (PC1 vs. PC2 Separation)
# ------------------------------------------------------------------------------
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

var_pc1 = pca_2d.explained_variance_ratio_[0] * 100
var_pc2 = pca_2d.explained_variance_ratio_[1] * 100

plt.figure(figsize=(9, 7))
sns.scatterplot(
    x=X_pca_2d[:, 0],
    y=X_pca_2d[:, 1],
    hue=['Schizophrenia' if label == 1 else 'Healthy' for label in y],
    palette={'Schizophrenia': '#e74c3c', 'Healthy': '#2ecc71'},
    alpha=0.7,
    s=45
)
plt.title("PCA 2D Projection: Schizophrenia vs. Healthy Brainwaves", fontsize=13, weight='bold')
plt.xlabel(f"Principal Component 1 ({var_pc1:.2f}% Variance)", fontsize=11)
plt.ylabel(f"Principal Component 2 ({var_pc2:.2f}% Variance)", fontsize=11)
plt.legend(title="Class", loc='upper right')
plt.grid(True, linestyle=':', alpha=0.5)
plt.savefig(PLOTS_DIR / "2_pca_2d_projection.png", bbox_inches='tight', dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# PLOT 3: Top Feature Loadings for PC1 and PC2 (Which features drive the PCs?)
# ------------------------------------------------------------------------------
loadings_df = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1_Loading', 'PC2_Loading'],
    index=feature_names
)

# Top 10 features driving PC1
top_pc1 = loadings_df['PC1_Loading'].abs().sort_values(ascending=False).head(10).index
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

loadings_df.loc[top_pc1, 'PC1_Loading'].plot(kind='barh', ax=ax1, color='#3498db')
ax1.set_title("Top 10 EEG Features Driving PC1", fontsize=12, weight='bold')
ax1.set_xlabel("Loading Weight")

# Top 10 features driving PC2
top_pc2 = loadings_df['PC2_Loading'].abs().sort_values(ascending=False).head(10).index
loadings_df.loc[top_pc2, 'PC2_Loading'].plot(kind='barh', ax=ax2, color='#9b59b6')
ax2.set_title("Top 10 EEG Features Driving PC2", fontsize=12, weight='bold')
ax2.set_xlabel("Loading Weight")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "3_pca_feature_loadings.png", bbox_inches='tight', dpi=300)
plt.close()

# ==============================================================================
# 4. SAVE PCA-REDUCED DATASET (Retaining 95% Variance)
# ==============================================================================
print(f"Creating reduced dataset with {n_95} components (95% variance)...")
pca_95 = PCA(n_components=n_95)
X_pca_95 = pca_95.fit_transform(X_scaled)

pc_columns = [f"PC_{i+1}" for i in range(n_95)]
df_pca_reduced = pd.DataFrame(X_pca_95, columns=pc_columns)

# Reattach metadata columns at the front
for col in reversed(meta_cols):
    df_pca_reduced.insert(0, col, df[col].values)

output_csv = OUTPUT_DIR / "schizophrenia_pca_features.csv"
df_pca_reduced.to_csv(output_csv, index=False)

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 65)
print("SUCCESS! PCA PIPELINE COMPLETE")
print("=" * 65)
print(f"1. Saved Reduced Dataset:  {output_csv.name}")
print(f"   • Reduced from 448 features down to: {n_95} Principal Components")
print(f"   • Final Shape: {df_pca_reduced.shape} (840 rows x {df_pca_reduced.shape[1]} columns)")
print("-" * 65)
print(f"2. Plots saved in: {PLOTS_DIR}/")
print("   • 1_pca_scree_plot.png       (Explained variance curve)")
print("   • 2_pca_2d_projection.png    (2D visual separation of patients vs healthy)")
print("   • 3_pca_feature_loadings.png (Which EEG features drive PC1 and PC2)")
print("=" * 65)