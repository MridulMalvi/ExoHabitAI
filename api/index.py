import os
import sys

# Get the directory where index.py is located (the api folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# The root directory is one level up from the api folder
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# Add root directory to path so we can import modules if needed
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

try:
    import imblearn  # REQUIRED for model loading
except ImportError:
    # This might happen if requirements installation failed silently
    imblearn = None

app = Flask(__name__, 
            template_folder=os.path.join(ROOT_DIR, 'templates'),
            static_folder=os.path.join(ROOT_DIR, 'static'))


def _patch_simple_imputer_fill_dtype(pipeline):
    """Work around scikit-learn pickle breakage."""
    try:
        steps = getattr(pipeline, "named_steps", {})
        imputer = steps.get("simpleimputer") if isinstance(steps, dict) else None
        if imputer and not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "_fit_dtype"):
            imputer._fill_dtype = imputer._fit_dtype  # type: ignore[attr-defined]
    except Exception as exc:
        app.logger.warning("Imputer patch skipped: %s", exc)


# Load model at startup
model_path = os.path.join(ROOT_DIR, "exo_model.pkl")

# Error handling for model loading to help debug Vercel logs
model = None
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        _patch_simple_imputer_fill_dtype(model)
    else:
        print(f"Error: Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.get_json()

    # Feature order MUST match training
    features = [
        data["P_RADIUS"],
        data["P_MASS"],
        data["P_GRAVITY"],
        data["P_PERIOD"],
        data["P_TEMP_EQUIL"],
        data["S_MASS"],
        data["S_RADIUS"],
        data["S_TEMPERATURE"],
        data["S_LUMINOSITY"]
    ]

    input_df = pd.DataFrame([features], columns=[
        "P_RADIUS",
        "P_MASS",
        "P_GRAVITY",
        "P_PERIOD",
        "P_TEMP_EQUIL",
        "S_MASS",
        "S_RADIUS",
        "S_TEMPERATURE",
        "S_LUMINOSITY"
    ])

    prediction = int(model.predict(input_df)[0])
    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "prediction": prediction,
        "label": "Potentially Habitable" if prediction == 1 else "Non-Habitable",
        "probability": round(probability * 100, 2)
    })


# Vercel requires the app to be exposed
app = app
