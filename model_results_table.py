import os
import pandas as pd

REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 75)
print("PUBLICATION-QUALITY MODEL RESULTS TABLE")
print("=" * 75)


# ============================================================
# 1. LOAD EXISTING MODEL COMPARISON
# ============================================================

existing_file = os.path.join(
    "Notebook",
    "final_model_comparison.csv"
)

if os.path.exists(existing_file):

    existing = pd.read_csv(existing_file)

    print("\nExisting model comparison:")
    print(existing.to_string(index=False))

else:

    print("\nERROR:")
    print("Notebook/final_model_comparison.csv was not found.")

    existing = pd.DataFrame()


# ============================================================
# 2. LOAD CROSS-VALIDATION RESULTS
# ============================================================

cv_file = os.path.join(
    REPORT_DIR,
    "model_robustness_summary.csv"
)

if os.path.exists(cv_file):

    cv_results = pd.read_csv(cv_file)

    print("\n5-fold cross-validation results:")
    print(cv_results.to_string(index=False))

else:

    print("\nWARNING:")
    print("model_robustness_summary.csv not found.")

    cv_results = pd.DataFrame()


# ============================================================
# 3. BUILD PUBLICATION TABLE
# ============================================================

publication_rows = []


# ------------------------------------------------------------
# Add original model comparison
# ------------------------------------------------------------

if not existing.empty:

    for _, row in existing.iterrows():

        publication_rows.append({

            "Model": row["Model"],

            "Original Accuracy (%)":
                float(row["Accuracy"]),

            "Original Macro Precision (%)":
                float(row["Macro Precision"]),

            "Original Macro Recall (%)":
                float(row["Macro Recall"]),

            "Original Macro F1 (%)":
                float(row["Macro F1"]),

            "5-Fold Accuracy (%)":
                float("nan"),

            "5-Fold Macro F1 (%)":
                float("nan")
        })


# ------------------------------------------------------------
# Add cross-validation results
# ------------------------------------------------------------

if not cv_results.empty:

    for _, row in cv_results.iterrows():

        model_name = row["Model"]

        matching_index = None

        for i, item in enumerate(publication_rows):

            if item["Model"] == model_name:

                matching_index = i
                break


        if matching_index is not None:

            publication_rows[
                matching_index
            ]["5-Fold Accuracy (%)"] = float(
                row["Accuracy"]
            )

            publication_rows[
                matching_index
            ]["5-Fold Macro F1 (%)"] = float(
                row["Macro F1"]
            )

        else:

            publication_rows.append({

                "Model": model_name,

                "Original Accuracy (%)":
                    float("nan"),

                "Original Macro Precision (%)":
                    float("nan"),

                "Original Macro Recall (%)":
                    float("nan"),

                "Original Macro F1 (%)":
                    float("nan"),

                "5-Fold Accuracy (%)":
                    float(row["Accuracy"]),

                "5-Fold Macro F1 (%)":
                    float(row["Macro F1"])
            })


# ============================================================
# 4. CREATE DATAFRAME
# ============================================================

publication_df = pd.DataFrame(
    publication_rows
)


# ============================================================
# 5. ROUND NUMERIC VALUES
# ============================================================

numeric_columns = [
    "Original Accuracy (%)",
    "Original Macro Precision (%)",
    "Original Macro Recall (%)",
    "Original Macro F1 (%)",
    "5-Fold Accuracy (%)",
    "5-Fold Macro F1 (%)"
]

for column in numeric_columns:

    if column in publication_df.columns:

        publication_df[column] = pd.to_numeric(
            publication_df[column],
            errors="coerce"
        ).round(2)


# ============================================================
# 6. SAVE PUBLICATION TABLE
# ============================================================

master_file = os.path.join(
    REPORT_DIR,
    "publication_model_results.csv"
)

publication_df.to_csv(
    master_file,
    index=False
)


print("\n")
print("=" * 75)
print("PUBLICATION MODEL RESULTS")
print("=" * 75)

print(
    publication_df.to_string(index=False)
)

print("\nSaved:")
print(master_file)


# ============================================================
# 7. BEST ORIGINAL ACCURACY
# ============================================================

print("\n")
print("=" * 75)
print("MODEL ANALYSIS")
print("=" * 75)


if "Original Accuracy (%)" in publication_df.columns:

    valid = publication_df.dropna(
        subset=["Original Accuracy (%)"]
    )

    if not valid.empty:

        best = valid.loc[
            valid["Original Accuracy (%)"].idxmax()
        ]

        print(
            f"\nHighest original accuracy:"
        )

        print(
            f"{best['Model']} "
            f"({best['Original Accuracy (%)']:.2f}%)"
        )


# ============================================================
# 8. BEST ORIGINAL MACRO F1
# ============================================================

if "Original Macro F1 (%)" in publication_df.columns:

    valid = publication_df.dropna(
        subset=["Original Macro F1 (%)"]
    )

    if not valid.empty:

        best = valid.loc[
            valid["Original Macro F1 (%)"].idxmax()
        ]

        print(
            f"\nHighest original Macro F1:"
        )

        print(
            f"{best['Model']} "
            f"({best['Original Macro F1 (%)']:.2f}%)"
        )


# ============================================================
# 9. BEST CROSS-VALIDATION ACCURACY
# ============================================================

if "5-Fold Accuracy (%)" in publication_df.columns:

    valid = publication_df.dropna(
        subset=["5-Fold Accuracy (%)"]
    )

    if not valid.empty:

        best = valid.loc[
            valid["5-Fold Accuracy (%)"].idxmax()
        ]

        print(
            f"\nHighest 5-fold accuracy:"
        )

        print(
            f"{best['Model']} "
            f"({best['5-Fold Accuracy (%)']:.2f}%)"
        )

    else:

        print(
            "\n5-fold accuracy:"
            "\nNot available yet."
        )


# ============================================================
# 10. BEST CROSS-VALIDATION MACRO F1
# ============================================================

if "5-Fold Macro F1 (%)" in publication_df.columns:

    valid = publication_df.dropna(
        subset=["5-Fold Macro F1 (%)"]
    )

    if not valid.empty:

        best = valid.loc[
            valid["5-Fold Macro F1 (%)"].idxmax()
        ]

        print(
            f"\nHighest 5-fold Macro F1:"
        )

        print(
            f"{best['Model']} "
            f"({best['5-Fold Macro F1 (%)']:.2f}%)"
        )

    else:

        print(
            "\n5-fold Macro F1:"
            "\nNot available yet."
        )


# ============================================================
# 11. RESEARCH INTERPRETATION
# ============================================================

print("\n")
print("=" * 75)
print("RESEARCH INTERPRETATION")
print("=" * 75)

print(
    """
Accuracy measures overall classification correctness.

Macro Precision, Macro Recall and Macro F1 give equal
importance to each target class.

This is particularly important because the diabetes dataset
contains substantial class imbalance.

Five-fold stratified cross-validation can be used to examine
whether model performance remains reasonably consistent across
different data partitions.

The final model selection should therefore consider both
overall accuracy and balanced class-level performance.
"""
)

print("=" * 75)
print("RESULTS TABLE GENERATION COMPLETED")
print("=" * 75)