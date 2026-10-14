import os
import math
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ============================================================
# 1. APPLICATION AND PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model_deploy.joblib")
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "diabetes_012_health_indicators_BRFSS2015.csv",
)

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="",
)
CORS(app)

# Keep this feature order identical to the order used to train the compact model.
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
    "Income",
]

# ============================================================
# 2. LOAD THE MODEL AND DATASET
# ============================================================

model = None
df = None

try:
    if os.path.isfile(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Machine Learning model loaded successfully!")
        print("Model type:", type(model).__name__)
        print("Model path:", MODEL_PATH)
    else:
        print(f"ERROR: Model file not found: {MODEL_PATH}")
except Exception as exc:
    print("ERROR: Could not load ML model:", exc)

try:
    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print("Dataset loaded successfully!")
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])
    else:
        print(f"WARNING: Dataset file not found: {DATA_PATH}")
except Exception as exc:
    print("ERROR: Could not load dataset:", exc)

# ============================================================
# 3. PAGES
# ============================================================

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/dashboard")
def dashboard():
    return send_from_directory(FRONTEND_DIR, "dashboard.html")


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.route("/health")
def health():
    model_loaded = model is not None
    dataset_loaded = df is not None
    return jsonify({
        "status": "success" if model_loaded else "error",
        "message": "Diabetes Screening API is running",
        "model": "loaded" if model_loaded else "not loaded",
        "dataset": "loaded" if dataset_loaded else "not loaded",
    }), (200 if model_loaded else 503)


# ============================================================
# 5. ANALYTICS API
# ============================================================

@app.route("/api/analytics")
def analytics():
    try:
        if df is None:
            return jsonify({
                "success": False,
                "error": "Dataset is not loaded.",
            }), 500

        required_columns = {"Diabetes_012", "BMI", "HighBP", "HighChol", "GenHlth", "Age"}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            return jsonify({
                "success": False,
                "error": "Dataset is missing required columns.",
                "missing": missing_columns,
            }), 500

        target_counts = df["Diabetes_012"].value_counts().sort_index()
        bmi_by_class = df.groupby("Diabetes_012")["BMI"].mean().round(2)
        high_bp_counts = df["HighBP"].value_counts().sort_index()
        high_chol_counts = df["HighChol"].value_counts().sort_index()
        general_health_counts = df["GenHlth"].value_counts().sort_index()
        age_counts = df["Age"].value_counts().sort_index()

        return jsonify({
            "success": True,
            "total_records": int(len(df)),
            "total_features": int(len(df.columns) - 1),
            "target_classes": int(df["Diabetes_012"].nunique()),
            "target_distribution": {
                "no_diabetes": int(target_counts.get(0, 0)),
                "prediabetes": int(target_counts.get(1, 0)),
                "diabetes": int(target_counts.get(2, 0)),
            },
            "bmi_by_class": {
                "no_diabetes": float(bmi_by_class.get(0, 0)),
                "prediabetes": float(bmi_by_class.get(1, 0)),
                "diabetes": float(bmi_by_class.get(2, 0)),
            },
            "high_bp": {
                "no": int(high_bp_counts.get(0, 0)),
                "yes": int(high_bp_counts.get(1, 0)),
            },
            "high_cholesterol": {
                "no": int(high_chol_counts.get(0, 0)),
                "yes": int(high_chol_counts.get(1, 0)),
            },
            "general_health": {
                str(level): int(general_health_counts.get(level, 0))
                for level in range(1, 6)
            },
            "age_distribution": {
                str(int(age)): int(count)
                for age, count in age_counts.items()
            },
        })
    except Exception as exc:
        app.logger.exception("Analytics API error")
        return jsonify({"success": False, "error": "Could not calculate analytics."}), 500


# ============================================================
# 6. PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({
                "success": False,
                "error": "Machine Learning model is not loaded.",
            }), 503

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Send the input features as a JSON object.",
            }), 400

        missing_features = [feature for feature in FEATURES if feature not in data]
        if missing_features:
            return jsonify({
                "success": False,
                "error": "Missing input features.",
                "missing": missing_features,
            }), 400

        input_data = {}
        for feature in FEATURES:
            try:
                value = float(data[feature])
            except (TypeError, ValueError):
                return jsonify({
                    "success": False,
                    "error": f"Feature '{feature}' must be numeric.",
                }), 400

            if not math.isfinite(value):
                return jsonify({
                    "success": False,
                    "error": f"Feature '{feature}' must be a finite number.",
                }), 400

            input_data[feature] = value

        input_df = pd.DataFrame([input_data], columns=FEATURES)
        prediction = int(model.predict(input_df)[0])

        probabilities = None
        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba(input_df)[0]
            probabilities = [float(value) for value in raw_probabilities]

        result_map = {
            0: (
                "No Diabetes",
                "Lower Risk",
                "The model predicts a lower likelihood of diabetes based on the provided indicators.",
            ),
            1: (
                "Prediabetes",
                "Intermediate Risk",
                "The model predicts an intermediate diabetes-risk category based on the provided indicators.",
            ),
            2: (
                "Diabetes",
                "Higher Risk",
                "The model predicts a higher likelihood of diabetes based on the provided indicators.",
            ),
        }

        result, risk_level, message = result_map.get(
            prediction,
            ("Unknown", "Unknown", "The model returned an unexpected class."),
        )

        confidence = round(max(probabilities) * 100, 2) if probabilities else None

        return jsonify({
            "success": True,
            "prediction": prediction,
            "result": result,
            "risk_level": risk_level,
            "confidence": confidence,
            "message": message,
            "disclaimer": (
                "Educational screening prototype only. This result is not a diagnosis. "
                "Consult a qualified healthcare professional and appropriate laboratory tests."
            ),
        })
    except Exception:
        app.logger.exception("Prediction API error")
        return jsonify({
            "success": False,
            "error": "Prediction failed. Check the server logs for details.",
        }), 500


# ============================================================
# 7. START SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}

    print("=" * 48)
    print("Diabetes Health Indicator Analysis")
    print("Flask backend starting...")
    print(f"Home:      http://127.0.0.1:{port}/")
    print(f"Dashboard: http://127.0.0.1:{port}/dashboard")
    print(f"Health:    http://127.0.0.1:{port}/health")
    print(f"Analytics: http://127.0.0.1:{port}/api/analytics")
    print("=" * 48)

    # 0.0.0.0 is required for container/cloud hosting. Locally, open 127.0.0.1.
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
