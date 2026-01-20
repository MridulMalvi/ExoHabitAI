import os
import sys

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
import imblearn  # REQUIRED for model loading

# Get the root directory path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(ROOT_DIR, 'templates'),
            static_folder=os.path.join(ROOT_DIR, 'static'))


def _patch_simple_imputer_fill_dtype(pipeline):
    """Work around scikit-learn 1.6 to 1.8 pickle breakage.

    The saved model was trained with scikit-learn 1.6.1 and the current
    environment has 1.8.0. When the `SimpleImputer` step is unpickled under
    1.8.0 it is missing the private `_fill_dtype` attribute, which causes an
    AttributeError during `transform`. We set it to the recorded `_fit_dtype`
    if needed so predictions keep working.
    """
    try:
        steps = getattr(pipeline, "named_steps", {})
        imputer = steps.get("simpleimputer") if isinstance(steps, dict) else None
        if imputer and not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "_fit_dtype"):
            imputer._fill_dtype = imputer._fit_dtype  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover — defensive logging only
        app.logger.warning("Imputer patch skipped: %s", exc)


# Load model once at startup and apply compatibility patch
model_path = os.path.join(ROOT_DIR, "exo_model.pkl")
model = joblib.load(model_path)
_patch_simple_imputer_fill_dtype(model)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
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
