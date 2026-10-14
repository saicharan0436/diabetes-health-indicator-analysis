import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================

DATA_PATH = "data/diabetes_012_health_indicators_BRFSS2015.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 70)
print("EXPLAINABLE AI - FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/5] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES
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
# 3. TRAIN RESEARCH RANDOM FOREST
# ============================================================

print("\n[2/5] Training Random Forest...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 4. CALCULATE FEATURE IMPORTANCE
# ============================================================

print("\n[3/5] Calculating feature importance...")

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

importance_df["Rank"] = range(
    1,
    len(importance_df) + 1
)

importance_df = importance_df[
    ["Rank", "Feature", "Importance"]
]

print("\nFeature Importance Ranking")
print("-" * 60)

print(importance_df.to_string(index=False))


# ============================================================
# 5. SAVE CSV
# ============================================================

csv_path = os.path.join(
    REPORT_DIR,
    "feature_importance_research.csv"
)

importance_df.to_csv(
    csv_path,
    index=False
)

print("\nSaved:")
print(csv_path)


# ============================================================
# 6. TOP FEATURES
# ============================================================

top_n = 10

top_features = importance_df.head(top_n)

print("\nTop 10 Features")
print("-" * 60)

for _, row in top_features.iterrows():

    print(
        f"{int(row['Rank']):2d}. "
        f"{row['Feature']:25s} "
        f"{row['Importance']:.4f}"
    )


# ============================================================
# 7. CREATE CHART
# ============================================================

print("\n[4/5] Creating feature importance chart...")

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Health Indicator")

plt.title(
    "Top 10 Health Indicators Based on Random Forest Feature Importance"
)

plt.tight_layout()

chart_path = os.path.join(
    REPORT_DIR,
    "top_10_feature_importance.png"
)

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(chart_path)


# ============================================================
# 8. SAVE TOP FEATURES SEPARATELY
# ============================================================

top_features_path = os.path.join(
    REPORT_DIR,
    "top_10_features.csv"
)

top_features.to_csv(
    top_features_path,
    index=False
)

print("Saved:")
print(top_features_path)


# ============================================================
# 9. FINAL INTERPRETATION
# ============================================================

print("\n[5/5] Analysis completed.")

print("\n")
print("=" * 70)
print("EXPLAINABILITY SUMMARY")
print("=" * 70)

print(
    """
Feature importance indicates how useful each input feature
was for the Random Forest's decision-making process.

The ranking should be interpreted as model-based importance,
not as proof that a feature causes diabetes.

The most important indicators can be discussed as potential
predictive signals within this dataset.
"""
)

print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)