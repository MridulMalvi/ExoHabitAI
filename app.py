import joblib
import pandas as pd
from flask import Flask, render_template, request
import imblearn  # Keep this: Essential for the model to load

app = Flask(__name__)

# Load model once
model = joblib.load('exo_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract values in the exact order the model was trained on
    features = [
        data['P_RADIUS'], data['P_MASS'], data['P_GRAVITY'], 
        data['P_PERIOD'], data['P_TEMP_EQUIL'], data['S_MASS'], 
        data['S_RADIUS'], data['S_TEMPERATURE'], data['S_LUMINOSITY']
    ]

    # Predict
    input_df = pd.DataFrame([features], columns=[
        'P_RADIUS', 'P_MASS', 'P_GRAVITY', 'P_PERIOD', 'P_TEMP_EQUIL', 
        'S_MASS', 'S_RADIUS', 'S_TEMPERATURE', 'S_LUMINOSITY'
    ])
    
    prediction = int(model.predict(input_df)[0])
    probability = model.predict_proba(input_df)[0][1]

    # Return simple JSON
    return {
        'prediction': prediction,
        'label': "Potentially Habitable" if prediction == 1 else "Non-Habitable",
        'probability': round(probability * 100, 2)
    }

if __name__ == '__main__':
    app.run(debug=True)