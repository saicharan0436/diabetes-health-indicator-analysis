import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ============================================================
# MODEL ROBUSTNESS AND CLASS-IMBALANCE ANALYSIS
# ============================================================

DATA_PATH = "data/diabetes_012_health_indicators_BRFSS2015.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 75)
print("MODEL ROBUSTNESS AND CLASS-IMBALANCE ANALYSIS")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/6] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

FEATURES = [
    "HighBP",
    "HighChol",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]

TARGET = "Diabetes_012"

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 3. STRATIFIED CROSS VALIDATION
# ============================================================

print("\n[2/6] Creating stratified 5-fold cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("5 folds created.")


# ============================================================
# 4. DEFINE MODELS
# ============================================================

print("\n[3/6] Defining models...")

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Balanced Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Balanced Random Forest Subsample": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 5. CROSS-VALIDATE EACH MODEL
# ============================================================

print("\n[4/6] Running cross-validation...")
print("This may take several minutes.")

results = []

for model_name, model in models.items():

    print("\nRunning:", model_name)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro"
        ],
        n_jobs=1,
        return_train_score=False
    )

    results.append({
        "Model": model_name,

        "Accuracy_Mean": scores["test_accuracy"].mean(),
        "Accuracy_STD": scores["test_accuracy"].std(),

        "Macro_Precision_Mean":
            scores["test_precision_macro"].mean(),

        "Macro_Precision_STD":
            scores["test_precision_macro"].std(),

        "Macro_Recall_Mean":
            scores["test_recall_macro"].mean(),

        "Macro_Recall_STD":
            scores["test_recall_macro"].std(),

        "Macro_F1_Mean":
            scores["test_f1_macro"].mean(),

        "Macro_F1_STD":
            scores["test_f1_macro"].std()
    })


# ============================================================
# 6. CREATE RESULTS TABLE
# ============================================================

print("\n[5/6] Creating comparison table...")

results_df = pd.DataFrame(results)

percentage_columns = [
    "Accuracy_Mean",
    "Accuracy_STD",
    "Macro_Precision_Mean",
    "Macro_Precision_STD",
    "Macro_Recall_Mean",
    "Macro_Recall_STD",
    "Macro_F1_Mean",
    "Macro_F1_STD"
]

for column in percentage_columns:
    results_df[column] = results_df[column] * 100


print("\n")
print("=" * 75)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 75)

print(
    results_df.round(2).to_string(index=False)
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

csv_path = os.path.join(
    REPORT_DIR,
    "model_robustness_comparison.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print("\nSaved:")
print(csv_path)


# ============================================================
# 8. CREATE SUMMARY TABLE
# ============================================================

summary_df = results_df[
    [
        "Model",
        "Accuracy_Mean",
        "Macro_Precision_Mean",
        "Macro_Recall_Mean",
        "Macro_F1_Mean"
    ]
].copy()

summary_df.columns = [
    "Model",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

summary_path = os.path.join(
    REPORT_DIR,
    "model_robustness_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("Saved:")
print(summary_path)


# ============================================================
# 9. PLOT ACCURACY
# ============================================================

print("\n[6/6] Creating research charts...")

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy_Mean"]
)

plt.ylabel("5-Fold Cross-Validation Accuracy (%)")
plt.xlabel("Model")
plt.title(
    "Accuracy Comparison Under 5-Fold Cross-Validation"
)

plt.xticks(rotation=15)

plt.tight_layout()

accuracy_chart = os.path.join(
    REPORT_DIR,
    "robustness_accuracy_comparison.png"
)

plt.savefig(
    accuracy_chart,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(accuracy_chart)


# ============================================================
# 10. PLOT MACRO F1
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Macro_F1_Mean"]
)

plt.ylabel("5-Fold Cross-Validation Macro F1 (%)")
plt.xlabel("Model")
plt.title(
    "Macro F1 Comparison Under 5-Fold Cross-Validation"
)

plt.xticks(rotation=15)

plt.tight_layout()

f1_chart = os.path.join(
    REPORT_DIR,
    "robustness_macro_f1_comparison.png"
)

plt.savefig(
    f1_chart,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(f1_chart)


# ============================================================
# 11. DETERMINE BEST MODEL BY MACRO F1
# ============================================================

best_model = results_df.loc[
    results_df["Macro_F1_Mean"].idxmax(),
    "Model"
]

best_f1 = results_df.loc[
    results_df["Macro_F1_Mean"].idxmax(),
    "Macro_F1_Mean"
]

best_accuracy_model = results_df.loc[
    results_df["Accuracy_Mean"].idxmax(),
    "Model"
]

best_accuracy = results_df.loc[
    results_df["Accuracy_Mean"].idxmax(),
    "Accuracy_Mean"
]


print("\n")
print("=" * 75)
print("RESEARCH INTERPRETATION")
print("=" * 75)

print(
    f"\nHighest cross-validation accuracy: "
    f"{best_accuracy_model} ({best_accuracy:.2f}%)"
)

print(
    f"Highest cross-validation macro F1: "
    f"{best_model} ({best_f1:.2f}%)"
)

print(
    """
Macro F1 gives equal importance to each class.
Therefore, it is particularly useful when the target
classes are imbalanced.

A model with lower accuracy but higher macro F1 may
provide more balanced performance across the three
diabetes categories.
"""
)

print("=" * 75)
print("ROBUSTNESS ANALYSIS COMPLETED")
print("=" * 75)