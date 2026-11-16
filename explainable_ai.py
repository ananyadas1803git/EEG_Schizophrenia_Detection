from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for clean saving in VS Code
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
import shap

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "xai_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

data_file = OUTPUT_DIR / "schizophrenia_6s_10segments.csv"
print(f"Loading data from: {data_file.name}...")
df = pd.read_csv(data_file)

# Separate Features (X) and Target (y)
meta_cols = ['subject_id', 'segment_id', 'label']
X = df.drop(columns=meta_cols)
y = df['label'].values
feature_names = X.columns.tolist()

# Train model
print("Fitting model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X, y)

# ==============================================================================
# 2. APPLY EXPLAINABLE AI (SHAP)
# ==============================================================================
print("Computing SHAP values (this may take a few seconds)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Handle 2-class binary output (select class 1 = Schizophrenia)
if len(shap_values.shape) == 3:
    shap_explanation = shap_values[:, :, 1]
    raw_shap_matrix = shap_values.values[:, :, 1]
else:
    shap_explanation = shap_values
    raw_shap_matrix = shap_values.values

# ==============================================================================
# 3. GENERATE CORE EXPLAINABILITY OUTPUTS
# ==============================================================================

# A. Global Beeswarm Plot (Shows feature ranking + directional risk)
print("1. Generating SHAP Beeswarm Summary Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(raw_shap_matrix, X, max_display=15, show=False)
plt.title("SHAP Summary: How Features Influence Schizophrenia Diagnosis", fontsize=12, weight='bold')
plt.savefig(PLOTS_DIR / "shap_beeswarm.png", bbox_inches='tight', dpi=300)
plt.close()

# B. Global Feature Importance Bar Plot
print("2. Generating SHAP Feature Importance Bar Plot...")
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_explanation, max_display=15, show=False)
plt.title("SHAP Feature Importance: Top 15 Discriminative Biomarkers", fontsize=12, weight='bold')
plt.savefig(PLOTS_DIR / "shap_importance_bar.png", bbox_inches='tight', dpi=300)
plt.close()

# C. Local Single-Patient Explanation (Waterfall Plot)
print("3. Generating Individual Patient Diagnostic Report (Waterfall Plot)...")
patient_idx = np.where(y == 1)[0][0]  # First schizophrenia patient in dataset
patient_id = df.loc[patient_idx, 'subject_id']

plt.figure(figsize=(9, 6))
shap.plots.waterfall(shap_explanation[patient_idx], max_display=10, show=False)
plt.title(f"Patient Explanation Report: Subject '{patient_id}'", fontsize=12, weight='bold')
plt.savefig(PLOTS_DIR / "shap_patient_waterfall.png", bbox_inches='tight', dpi=300)
plt.close()

# D. Save Ranked Features Table to CSV
mean_abs_shap = np.mean(np.abs(raw_shap_matrix), axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Absolute_SHAP': mean_abs_shap
}).sort_values('Mean_Absolute_SHAP', ascending=False)

csv_path = PLOTS_DIR / "ranked_features_shap.csv"
importance_df.to_csv(csv_path, index=False)

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 65)
print("EXPLAINABLE AI EXECUTION COMPLETE!")
print("=" * 65)
print(f"Outputs saved in: {PLOTS_DIR}/")
print("  • shap_beeswarm.png        (Global directional impact of features)")
print("  • shap_importance_bar.png  (Top 15 ranked features)")
print("  • shap_patient_waterfall.png (Diagnostic explanation for 1 patient)")
print("  • ranked_features_shap.csv   (All 448 features ranked by clinical weight)")
print("=" * 65)