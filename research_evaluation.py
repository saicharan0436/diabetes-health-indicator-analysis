import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/diabetes_012_health_indicators_BRFSS2015.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 70)
print("RESEARCH EVALUATION - DIABETES HEALTH INDICATOR PROJECT")
print("=" * 70)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[1/8] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

print("\n[2/8] Preparing features and target...")

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

print("Number of features:", len(FEATURES))
print("Target:", TARGET)

print("\nTarget distribution:")
print(y.value_counts().sort_index())


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n[3/8] Creating stratified train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 5. RANDOM FOREST RESEARCH MODEL
# ============================================================

print("\n[4/8] Training Random Forest research model...")

research_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

research_model.fit(X_train, y_train)

print("Random Forest training completed.")


# ============================================================
# 6. TEST SET EVALUATION
# ============================================================

print("\n[5/8] Evaluating model...")

y_pred = research_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)
macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\nResearch Model Results")
print("-" * 50)
print(f"Accuracy       : {accuracy * 100:.2f}%")
print(f"Macro Precision: {macro_precision * 100:.2f}%")
print(f"Macro Recall   : {macro_recall * 100:.2f}%")
print(f"Macro F1       : {macro_f1 * 100:.2f}%")


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\n[6/8] Generating per-class metrics...")

class_names = [
    "No Diabetes",
    "Prediabetes",
    "Diabetes"
]

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

classification_df = pd.DataFrame(report).transpose()

classification_path = os.path.join(
    REPORT_DIR,
    "research_classification_report.csv"
)

classification_df.to_csv(classification_path)

print("\nPer-class classification report:")
print(classification_df)

print("\nSaved:")
print(classification_path)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\n[7/8] Creating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

confusion_path = os.path.join(
    REPORT_DIR,
    "research_confusion_matrix.png"
)

plt.savefig(confusion_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print(confusion_path)


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

print("\n[8/8] Calculating feature importance...")

importance = research_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

feature_importance_path = os.path.join(
    REPORT_DIR,
    "research_feature_importance.csv"
)

feature_importance_df.to_csv(
    feature_importance_path,
    index=False
)

print("\nFeature Importance:")
print(feature_importance_df)

print("\nSaved:")
print(feature_importance_path)


# ============================================================
# 10. FEATURE IMPORTANCE CHART
# ============================================================

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Feature Importance")
plt.ylabel("Health Indicator")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

importance_chart_path = os.path.join(
    REPORT_DIR,
    "research_feature_importance.png"
)

plt.savefig(
    importance_chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(importance_chart_path)


# ============================================================
# 11. FIVE-FOLD CROSS VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    cv_model,
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

cv_summary = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],
    "Mean": [
        cv_results["test_accuracy"].mean(),
        cv_results["test_precision_macro"].mean(),
        cv_results["test_recall_macro"].mean(),
        cv_results["test_f1_macro"].mean()
    ],
    "Std": [
        cv_results["test_accuracy"].std(),
        cv_results["test_precision_macro"].std(),
        cv_results["test_recall_macro"].std(),
        cv_results["test_f1_macro"].std()
    ]
})

cv_summary["Mean_Percent"] = cv_summary["Mean"] * 100
cv_summary["Std_Percent"] = cv_summary["Std"] * 100

print("\nCross-validation results:")
print(cv_summary)

cv_path = os.path.join(
    REPORT_DIR,
    "research_cross_validation_results.csv"
)

cv_summary.to_csv(
    cv_path,
    index=False
)

print("\nSaved:")
print(cv_path)


# ============================================================
# 12. OVERALL RESEARCH SUMMARY
# ============================================================

summary_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],
    "Test_Set_Percent": [
        accuracy * 100,
        macro_precision * 100,
        macro_recall * 100,
        macro_f1 * 100
    ],
    "Cross_Validation_Mean_Percent": [
        cv_summary.loc[
            cv_summary["Metric"] == "Accuracy",
            "Mean_Percent"
        ].iloc[0],

        cv_summary.loc[
            cv_summary["Metric"] == "Macro Precision",
            "Mean_Percent"
        ].iloc[0],

        cv_summary.loc[
            cv_summary["Metric"] == "Macro Recall",
            "Mean_Percent"
        ].iloc[0],

        cv_summary.loc[
            cv_summary["Metric"] == "Macro F1",
            "Mean_Percent"
        ].iloc[0]
    ]
})

summary_path = os.path.join(
    REPORT_DIR,
    "research_model_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\n")
print("=" * 70)
print("RESEARCH EVALUATION COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print("1.", classification_path)
print("2.", confusion_path)
print("3.", feature_importance_path)
print("4.", importance_chart_path)
print("5.", cv_path)
print("6.", summary_path)

print("\nYour reports folder now contains research evaluation outputs.")
print("=" * 70)