import pandas as pd
import os


# ============================================================
# DATASET LOCATION
# ============================================================

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "diabetes_012_health_indicators_BRFSS2015.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n============================================")
print("DATASET INFORMATION")
print("============================================")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumns:")
for column in df.columns:
    print("-", column)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

target_counts = (
    df["Diabetes_012"]
    .value_counts()
    .sort_index()
)

print("\n============================================")
print("DIABETES DISTRIBUTION")
print("============================================")

print(target_counts)


# ============================================================
# TARGET PERCENTAGES
# ============================================================

target_percentages = (
    df["Diabetes_012"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nTarget percentages:")

for category, percentage in target_percentages.items():

    if category == 0:
        name = "No Diabetes"

    elif category == 1:
        name = "Prediabetes"

    elif category == 2:
        name = "Diabetes"

    else:
        name = "Unknown"

    print(
        f"{name}: {percentage:.2f}%"
    )


# ============================================================
# HEALTH INDICATOR ANALYSIS
# ============================================================

print("\n============================================")
print("HEALTH INDICATOR ANALYSIS")
print("============================================")


# High Blood Pressure

high_bp = (
    df.groupby("HighBP")["Diabetes_012"]
    .mean()
)

print("\nAverage diabetes class by HighBP:")

print(high_bp)


# High Cholesterol

high_chol = (
    df.groupby("HighChol")["Diabetes_012"]
    .mean()
)

print("\nAverage diabetes class by HighChol:")

print(high_chol)


# BMI

bmi_statistics = df["BMI"].describe()

print("\nBMI Statistics:")

print(bmi_statistics)


# General Health

general_health = (
    df["GenHlth"]
    .value_counts()
    .sort_index()
)

print("\nGeneral Health Distribution:")

print(general_health)


# Age

age_distribution = (
    df["Age"]
    .value_counts()
    .sort_index()
)

print("\nAge Distribution:")

print(age_distribution)


# ============================================================
# FEATURE CORRELATION
# ============================================================

print("\n============================================")
print("FEATURE CORRELATION")
print("============================================")

correlation = (
    df.corr(numeric_only=True)["Diabetes_012"]
    .sort_values(ascending=False)
)

print("\nCorrelation with Diabetes_012:")

print(correlation)


# ============================================================
# SAVE ANALYSIS RESULTS
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# Target distribution

target_counts.to_csv(
    os.path.join(
        RESULTS_DIR,
        "diabetes_distribution.csv"
    )
)


# Correlation

correlation.to_csv(
    os.path.join(
        RESULTS_DIR,
        "feature_correlations.csv"
    )
)


# General health

general_health.to_csv(
    os.path.join(
        RESULTS_DIR,
        "general_health_distribution.csv"
    )
)


# Age

age_distribution.to_csv(
    os.path.join(
        RESULTS_DIR,
        "age_distribution.csv"
    )
)


print("\n============================================")
print("ANALYSIS COMPLETED")
print("============================================")

print("Results saved inside:")
print("reports/")