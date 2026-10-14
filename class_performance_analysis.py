
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Create output folder
os.makedirs("reports", exist_ok=True)

# 2. Load dataset
data_path = "data/diabetes_012_health_indicators_BRFSS2015.csv"

df = pd.read_csv(data_path)

# 3. Select features and target
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

X = df[features]
y = df["Diabetes_012"]

# 4. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Train a research Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# 6. Predict test records
y_pred = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# 7. Per-class report
class_names = ["No Diabetes", "Prediabetes", "Diabetes"]

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_path = "reports/per_class_performance.csv"
report_df.to_csv(report_path, index=True)

print("\nPer-class performance:")
print(report_df.round(4).to_string())

# 8. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv("reports/research_confusion_matrix.csv")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "reports/research_confusion_matrix_detail.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nCreated:")
print("reports/per_class_performance.csv")
print("reports/research_confusion_matrix.csv")
print("reports/research_confusion_matrix_detail.png")
print("\nCLASS PERFORMANCE ANALYSIS COMPLETED")
