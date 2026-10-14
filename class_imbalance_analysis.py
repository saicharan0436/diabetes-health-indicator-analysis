import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CLASS IMBALANCE ANALYSIS
# ============================================================

DATA_PATH = "data/diabetes_012_health_indicators_BRFSS2015.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 70)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded.")
print("Shape:", df.shape)

# ------------------------------------------------------------
# 2. Target distribution
# ------------------------------------------------------------

target_counts = df["Diabetes_012"].value_counts().sort_index()

class_names = {
    0: "No Diabetes",
    1: "Prediabetes",
    2: "Diabetes"
}

total = len(df)

analysis = pd.DataFrame({
    "Class_ID": target_counts.index,
    "Class_Name": [
        class_names[x] for x in target_counts.index
    ],
    "Count": target_counts.values
})

analysis["Percentage"] = (
    analysis["Count"] / total * 100
)

# ------------------------------------------------------------
# 3. Imbalance ratio
# ------------------------------------------------------------

largest_class = analysis["Count"].max()
smallest_class = analysis["Count"].min()

imbalance_ratio = largest_class / smallest_class

print("\nClass Distribution")
print("-" * 50)

print(analysis.to_string(index=False))

print("\nTotal records:", total)

print(
    f"\nLargest-to-smallest class ratio: "
    f"{imbalance_ratio:.2f}:1"
)

# ------------------------------------------------------------
# 4. Save analysis
# ------------------------------------------------------------

csv_path = os.path.join(
    REPORT_DIR,
    "class_imbalance_analysis.csv"
)

analysis.to_csv(csv_path, index=False)

print("\nSaved:")
print(csv_path)

# ------------------------------------------------------------
# 5. Create chart
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

bars = plt.bar(
    analysis["Class_Name"],
    analysis["Count"]
)

plt.title(
    "Class Distribution of Diabetes Health Indicators Dataset"
)

plt.xlabel("Diabetes Class")
plt.ylabel("Number of Records")

# Add values above bars
for bar, count in zip(bars, analysis["Count"]):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

chart_path = os.path.join(
    REPORT_DIR,
    "class_imbalance_chart.png"
)

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(chart_path)

# ------------------------------------------------------------
# 6. Final interpretation
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    f"""
The dataset contains {total:,} records.

The majority class is:
No Diabetes ({analysis.iloc[0]['Percentage']:.2f}%).

The minority class is:
Prediabetes ({analysis.iloc[1]['Percentage']:.2f}%).

The largest class is approximately
{imbalance_ratio:.2f} times larger than the smallest class.

Therefore, the dataset exhibits substantial class imbalance.

Because of this imbalance, model evaluation should not rely
only on overall accuracy. Macro Precision, Macro Recall,
Macro F1-score and the Confusion Matrix should also be considered.
"""
)

print("=" * 70)
print("CLASS IMBALANCE ANALYSIS COMPLETED")
print("=" * 70)