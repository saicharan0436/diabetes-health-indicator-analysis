import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ==============================
# 1. Paths
# ==============================

DATA_PATH = "data/diabetes_012_health_indicators_BRFSS2015.csv"
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)


# ==============================
# 2. Load Dataset
# ==============================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==============================
# 3. Features and Target
# ==============================

features = [
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

target = "Diabetes_012"

X = df[features]
y = df[target]


# ==============================
# 4. Train/Test Split
# ==============================

print("Creating stratified train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 5. Train Random Forest
# ==============================

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Random Forest training completed.")


# ==============================
# 6. SHAP Sample
# ==============================

# We do NOT explain all 50,000+ test records.
# A smaller representative sample is sufficient
# for a research-level explanation.

sample_size = min(1000, len(X_test))

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

print("SHAP sample size:", len(X_sample))


# ==============================
# 7. Create SHAP Explainer
# ==============================

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)

print("SHAP calculation completed.")


# ==============================
# 8. Handle SHAP Output
# ==============================

# SHAP output differs between versions.
# This converts it into a format suitable
# for multiclass analysis.

if isinstance(shap_values, list):

    # Older SHAP versions:
    # list[class] -> samples x features

    mean_abs_values = []

    for class_values in shap_values:
        mean_abs_values.append(
            abs(class_values).mean(axis=0)
        )

    mean_abs_values = sum(mean_abs_values) / len(mean_abs_values)

else:

    # Newer SHAP versions may return:
    # samples x features x classes

    if len(shap_values.shape) == 3:

        mean_abs_values = abs(shap_values).mean(axis=(0, 2))

    else:

        mean_abs_values = abs(shap_values).mean(axis=0)


# ==============================
# 9. Create Feature Importance Table
# ==============================

shap_importance = pd.DataFrame({
    "Feature": features,
    "Mean_Absolute_SHAP": mean_abs_values
})

shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

shap_importance.to_csv(
    os.path.join(
        REPORTS_DIR,
        "shap_feature_importance.csv"
    ),
    index=False
)


# ==============================
# 10. Display Top Features
# ==============================

print("\nTop SHAP Features:")

print(
    shap_importance.head(10).to_string(index=False)
)


# ==============================
# 11. Plot Top 10 Features
# ==============================

top10 = shap_importance.head(10).sort_values(
    by="Mean_Absolute_SHAP"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top10["Feature"],
    top10["Mean_Absolute_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title(
    "Top 10 Features Based on SHAP Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORTS_DIR,
        "shap_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================
# 12. Completion Message
# ==============================

print("\n========================================")
print("SHAP ANALYSIS COMPLETED")
print("========================================")

print(
    "\nFiles created:"
)

print(
    "reports/shap_feature_importance.csv"
)

print(
    "reports/shap_feature_importance.png"
)