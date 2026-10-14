
import os
import pandas as pd

REPORTS_DIR = "reports"
OUTPUT_FILE = os.path.join(
    REPORTS_DIR,
    "research_results_summary.txt"
)

os.makedirs(REPORTS_DIR, exist_ok=True)

lines = []

def add(text=""):
    lines.append(str(text))

def read_csv_if_exists(filename):
    path = os.path.join(REPORTS_DIR, filename)

    if not os.path.exists(path):
        add(f"NOT FOUND: {filename}")
        return None

    try:
        df = pd.read_csv(path)
        add(f"\nFILE: {filename}")
        add(f"Rows: {len(df)}")
        add(f"Columns: {', '.join(df.columns.astype(str))}")
        add(df.head(10).to_string(index=False))
        return df
    except Exception as error:
        add(f"Could not read {filename}: {error}")
        return None

add("DIABETES HEALTH INDICATOR ANALYSIS")
add("Research Results Summary")
add("=" * 60)
add("")

# 1. Overall model comparison
model_df = read_csv_if_exists(
    "publication_results_table.csv"
)

# 2. Per-class metrics
class_df = read_csv_if_exists(
    "per_class_performance.csv"
)

# 3. Target imbalance
imbalance_df = read_csv_if_exists(
    "class_imbalance_analysis.csv"
)

# 4. Feature importance
feature_df = read_csv_if_exists(
    "research_feature_importance.csv"
)

# 5. SHAP feature importance
shap_df = read_csv_if_exists(
    "shap_feature_importance.csv"
)

# 6. Five-fold robustness summary
robustness_df = read_csv_if_exists(
    "model_robustness_summary.csv"
)

add("")
add("=" * 60)
add("INTERPRETATION CHECKLIST")
add("=" * 60)

add(
    "1. Compare accuracy with macro F1; accuracy alone "
    "may hide weak minority-class performance."
)

add(
    "2. Examine prediabetes recall in per_class_performance.csv."
)

add(
    "3. Use the confusion matrix to describe correct and "
    "incorrect predictions for each class."
)

add(
    "4. Compare model-based feature importance with SHAP "
    "importance if both files are available."
)

add(
    "5. Do not interpret feature importance as proof of "
    "causation or as a clinical diagnosis."
)

add(
    "6. Keep fixed-test results separate from cross-validation "
    "results and identify each evaluation method."
)

add("")
add(
    "This file summarizes available results. Missing files "
    "must be generated before drawing conclusions."
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.write("\n".join(lines))

print("\n".join(lines))
print("\nSummary saved to:", OUTPUT_FILE)
print("\nRESEARCH RESULTS SUMMARY COMPLETED")
