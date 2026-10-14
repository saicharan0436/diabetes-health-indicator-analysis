import os
import pandas as pd


# ==========================================
# 1. Paths
# ==========================================

REPORTS_DIR = "reports"
NOTEBOOK_DIR = "Notebook"

os.makedirs(REPORTS_DIR, exist_ok=True)


# ==========================================
# 2. Load Original Model Comparison
# ==========================================

original_path = os.path.join(
    NOTEBOOK_DIR,
    "final_model_comparison.csv"
)

if not os.path.exists(original_path):
    print("ERROR:")
    print(f"Missing file: {original_path}")
    print("Make sure your original model comparison exists.")
    raise SystemExit


original = pd.read_csv(original_path)

print("\nOriginal Model Results:")
print(original)


# ==========================================
# 3. Standardize Column Names
# ==========================================

original.columns = [
    str(col).strip()
    for col in original.columns
]


# ==========================================
# 4. Rename Columns if Necessary
# ==========================================

rename_map = {
    "Model": "Model",
    "Accuracy": "Accuracy",
    "Macro Precision": "Macro Precision",
    "Macro Recall": "Macro Recall",
    "Macro F1": "Macro F1",
    "Precision": "Macro Precision",
    "Recall": "Macro Recall",
    "F1": "Macro F1"
}

original = original.rename(
    columns=rename_map
)


# ==========================================
# 5. Keep Important Columns
# ==========================================

required_columns = [
    "Model",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

missing = [
    col for col in required_columns
    if col not in original.columns
]

if missing:

    print("\nMissing columns:")
    print(missing)

    print("\nAvailable columns:")
    print(original.columns.tolist())

    raise SystemExit


original = original[required_columns].copy()


# ==========================================
# 6. Convert Metrics to Numeric
# ==========================================

metric_columns = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

for column in metric_columns:

    original[column] = pd.to_numeric(
        original[column],
        errors="coerce"
    )


# ==========================================
# 7. Load Robustness Results
# ==========================================

robustness_path = os.path.join(
    REPORTS_DIR,
    "model_robustness_summary.csv"
)

if os.path.exists(robustness_path):

    robustness = pd.read_csv(
        robustness_path
    )

    robustness.columns = [
        str(col).strip()
        for col in robustness.columns
    ]

    print("\nRobustness Results:")
    print(robustness)

else:

    print(
        "\nWARNING: model_robustness_summary.csv "
        "not found."
    )

    robustness = pd.DataFrame()


# ==========================================
# 8. Prepare Publication Table
# ==========================================

publication_rows = []


# Original results

for _, row in original.iterrows():

    publication_rows.append({
        "Experiment": "Original Evaluation",
        "Model": row["Model"],
        "Accuracy (%)": row["Accuracy"],
        "Macro Precision (%)": row["Macro Precision"],
        "Macro Recall (%)": row["Macro Recall"],
        "Macro F1 (%)": row["Macro F1"]
    })


# ==========================================
# 9. Add Cross-Validation Results
# ==========================================

if not robustness.empty:

    print(
        "\nRobustness columns:"
    )

    print(
        robustness.columns.tolist()
    )

    # Try to identify model column

    model_column = None

    for candidate in [
        "Model",
        "model",
        "Estimator"
    ]:

        if candidate in robustness.columns:
            model_column = candidate
            break


    # Try to identify metric columns

    accuracy_column = None
    f1_column = None

    for candidate in [
        "Accuracy Mean",
        "accuracy_mean",
        "Mean Accuracy",
        "Accuracy"
    ]:

        if candidate in robustness.columns:
            accuracy_column = candidate
            break


    for candidate in [
        "F1 Macro Mean",
        "f1_macro_mean",
        "Mean Macro F1",
        "Macro F1"
    ]:

        if candidate in robustness.columns:
            f1_column = candidate
            break


    if model_column and accuracy_column and f1_column:

        for _, row in robustness.iterrows():

            accuracy = pd.to_numeric(
                row[accuracy_column],
                errors="coerce"
            )

            f1 = pd.to_numeric(
                row[f1_column],
                errors="coerce"
            )

            publication_rows.append({
                "Experiment": "5-Fold Cross-Validation",
                "Model": row[model_column],
                "Accuracy (%)": accuracy * 100,
                "Macro Precision (%)": None,
                "Macro Recall (%)": None,
                "Macro F1 (%)": f1 * 100
            })

    else:

        print(
            "\nWARNING:"
        )

        print(
            "Could not automatically identify "
            "the robustness metric columns."
        )


# ==========================================
# 10. Create Final DataFrame
# ==========================================

publication_df = pd.DataFrame(
    publication_rows
)


# ==========================================
# 11. Convert Metrics
# ==========================================

for column in metric_columns:

    publication_df[column + " (%)"] = pd.to_numeric(
        publication_df[column + " (%)"],
        errors="coerce"
    ).round(2)


# ==========================================
# 12. Save Publication Table
# ==========================================

output_path = os.path.join(
    REPORTS_DIR,
    "publication_results_table.csv"
)

publication_df.to_csv(
    output_path,
    index=False
)


# ==========================================
# 13. Print Final Table
# ==========================================

print("\n")
print("=" * 70)
print("PUBLICATION RESULTS TABLE")
print("=" * 70)

print(
    publication_df.to_string(index=False)
)


# ==========================================
# 14. Best Models
# ==========================================

valid_accuracy = publication_df.dropna(
    subset=["Accuracy (%)"]
)

valid_f1 = publication_df.dropna(
    subset=["Macro F1 (%)"]
)


if not valid_accuracy.empty:

    best_accuracy = valid_accuracy.loc[
        valid_accuracy["Accuracy (%)"].idxmax()
    ]

    print("\nBest Accuracy:")
    print(
        f"{best_accuracy['Model']} - "
        f"{best_accuracy['Accuracy (%)']:.2f}%"
    )


if not valid_f1.empty:

    best_f1 = valid_f1.loc[
        valid_f1["Macro F1 (%)"].idxmax()
    ]

    print("\nBest Macro F1:")
    print(
        f"{best_f1['Model']} - "
        f"{best_f1['Macro F1 (%)']:.2f}%"
    )


print("\n")
print("=" * 70)
print("PUBLICATION TABLE CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nSaved to:\n{output_path}"
)