from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os


# ============================================================
# 1. FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)

CORS(app)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Notebook",
    "diabetes_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "diabetes_012_health_indicators_BRFSS2015.csv"
)


# ============================================================
# 3. LOAD MACHINE LEARNING MODEL
# ============================================================

model = None

try:

    model = joblib.load(MODEL_PATH)

    print("--------------------------------------------")
    print("Machine Learning model loaded successfully!")
    print("Model:", type(model).__name__)
    print("--------------------------------------------")

except Exception as e:

    print("--------------------------------------------")
    print("ERROR: Could not load ML model")
    print(e)
    print("--------------------------------------------")


# ============================================================
# 4. LOAD DATASET
# ============================================================

df = None

try:

    df = pd.read_csv(DATA_PATH)

    print("--------------------------------------------")
    print("Dataset loaded successfully!")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("--------------------------------------------")

except Exception as e:

    print("--------------------------------------------")
    print("ERROR: Could not load dataset")
    print(e)
    print("--------------------------------------------")


# ============================================================
# 5. MODEL FEATURES
# ============================================================

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


# ============================================================
# 6. HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        "frontend",
        "index.html"
    )


# ============================================================
# 7. DASHBOARD PAGE
# ============================================================

@app.route("/dashboard")
def dashboard():

    return send_from_directory(
        "frontend",
        "dashboard.html"
    )


# ============================================================
# 8. HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "success" if model is not None else "error",

        "message": "Diabetes Screening API is running",

        "model": "loaded" if model is not None else "not loaded",

        "dataset": "loaded" if df is not None else "not loaded"

    })


# ============================================================
# 9. ANALYTICS API
# ============================================================

@app.route("/api/analytics")
def analytics():

    try:

        if df is None:

            return jsonify({
                "success": False,
                "error": "Dataset is not loaded."
            }), 500


        # ----------------------------------------------------
        # Target distribution
        # ----------------------------------------------------

        target_counts = (
            df["Diabetes_012"]
            .value_counts()
            .sort_index()
        )


        no_diabetes = int(
            target_counts.get(0, 0)
        )

        prediabetes = int(
            target_counts.get(1, 0)
        )

        diabetes = int(
            target_counts.get(2, 0)
        )


        # ----------------------------------------------------
        # Average BMI by diabetes class
        # ----------------------------------------------------

        bmi_by_class = (
            df.groupby("Diabetes_012")["BMI"]
            .mean()
            .round(2)
        )


        # ----------------------------------------------------
        # High Blood Pressure distribution
        # ----------------------------------------------------

        high_bp_counts = (
            df["HighBP"]
            .value_counts()
            .sort_index()
        )


        # ----------------------------------------------------
        # High Cholesterol distribution
        # ----------------------------------------------------

        high_chol_counts = (
            df["HighChol"]
            .value_counts()
            .sort_index()
        )


        # ----------------------------------------------------
        # General Health distribution
        # ----------------------------------------------------

        general_health_counts = (
            df["GenHlth"]
            .value_counts()
            .sort_index()
        )


        # ----------------------------------------------------
        # Age distribution
        # ----------------------------------------------------

        age_counts = (
            df["Age"]
            .value_counts()
            .sort_index()
        )


        # ----------------------------------------------------
        # Return analytics
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "total_records": int(len(df)),

            "total_features": int(
                len(df.columns) - 1
            ),

            "target_classes": int(
                df["Diabetes_012"].nunique()
            ),

            "target_distribution": {

                "no_diabetes": no_diabetes,

                "prediabetes": prediabetes,

                "diabetes": diabetes

            },

            "bmi_by_class": {

                "no_diabetes": float(
                    bmi_by_class.get(0, 0)
                ),

                "prediabetes": float(
                    bmi_by_class.get(1, 0)
                ),

                "diabetes": float(
                    bmi_by_class.get(2, 0)
                )

            },

            "high_bp": {

                "no": int(
                    high_bp_counts.get(0, 0)
                ),

                "yes": int(
                    high_bp_counts.get(1, 0)
                )

            },

            "high_cholesterol": {

                "no": int(
                    high_chol_counts.get(0, 0)
                ),

                "yes": int(
                    high_chol_counts.get(1, 0)
                )

            },

            "general_health": {

                "1": int(
                    general_health_counts.get(1, 0)
                ),

                "2": int(
                    general_health_counts.get(2, 0)
                ),

                "3": int(
                    general_health_counts.get(3, 0)
                ),

                "4": int(
                    general_health_counts.get(4, 0)
                ),

                "5": int(
                    general_health_counts.get(5, 0)
                )

            },

            "age_distribution": {

                str(int(age)): int(count)

                for age, count
                in age_counts.items()

            }

        })


    except Exception as e:

        print("--------------------------------------------")
        print("Analytics Error:")
        print(e)
        print("--------------------------------------------")

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# 10. PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None:

            return jsonify({

                "success": False,

                "error":
                "Machine Learning model is not loaded."

            }), 500


        data = request.get_json()


        if data is None:

            return jsonify({

                "success": False,

                "error":
                "No input data received."

            }), 400


        # ----------------------------------------------------
        # Check missing features
        # ----------------------------------------------------

        missing_features = [

            feature

            for feature in features

            if feature not in data

        ]


        if missing_features:

            return jsonify({

                "success": False,

                "error":
                "Missing input features.",

                "missing":
                missing_features

            }), 400


        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        input_data = {}

        for feature in features:

            input_data[feature] = float(
                data[feature]
            )


        input_df = pd.DataFrame(

            [input_data],

            columns=features

        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )[0]

        prediction = int(prediction)


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_df
            )[0]


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if prediction == 0:

            result = "No Diabetes"

            risk_level = "Lower Risk"

            message = (
                "The model predicts a lower likelihood "
                "of diabetes based on the provided indicators."
            )


        elif prediction == 1:

            result = "Prediabetes"

            risk_level = "Intermediate Risk"

            message = (
                "The model predicts an intermediate "
                "diabetes-risk category based on the "
                "provided indicators."
            )


        elif prediction == 2:

            result = "Diabetes"

            risk_level = "Higher Risk"

            message = (
                "The model predicts a higher likelihood "
                "of diabetes based on the provided indicators."
            )


        else:

            result = "Unknown"

            risk_level = "Unknown"

            message = (
                "The model returned an unexpected class."
            )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = None

        if probabilities is not None:

            confidence = round(

                float(
                    max(probabilities)
                ) * 100,

                2

            )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "prediction": prediction,

            "result": result,

            "risk_level": risk_level,

            "confidence": confidence,

            "message": message

        })


    except Exception as e:

        print("--------------------------------------------")
        print("Prediction Error:")
        print(e)
        print("--------------------------------------------")

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# 11. START SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("============================================")
    print(" Diabetes Health Indicator Analysis")
    print(" Flask Backend Starting...")
    print("============================================")

    print("")
    print("Prediction:")
    print("http://127.0.0.1:5000/")

    print("")
    print("Dashboard:")
    print("http://127.0.0.1:5000/dashboard")

    print("")
    print("Health:")
    print("http://127.0.0.1:5000/health")

    print("")
    print("Analytics API:")
    print("http://127.0.0.1:5000/api/analytics")

    print("")
    print("============================================")


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )