# 🪐 ExoHabitAI

**ExoHabitAI** is a machine learning-powered web application designed to predict the potential habitability of exoplanets. By analyzing critical planetary and stellar features, the system classifies exoplanets as "Potentially Habitable" or "Non-Habitable" using a trained Random Forest model.

## 🚀 Features
* **ML-Powered Predictions:** Uses a pre-trained Scikit-learn model to analyze planetary data.
* **Interactive Dashboard:** Visualizes results using Plotly.js charts, including:
    * Habitability Probability Gauge
    * Planetary Feature Radar Chart (comparing against Earth)
    * Decision Boundary visualization
    * Feature Deviation Bar Chart
   .
* **User-Friendly Interface:** Space-themed, responsive design built with HTML/CSS.
* **Real-time Analysis:** Instantly calculates habitability probability based on physical parameters.

## 🛠️ Tech Stack
* **Backend:** Python 3.x, Flask, Gunicorn
* **Machine Learning:** Scikit-learn, Pandas, NumPy, Imbalanced-learn, Joblib
* **Frontend:** HTML5, CSS3, JavaScript, Plotly.js
* **Deployment:** Render 

## 📋 Input Parameters
To generate a prediction, the model requires the following 9 attributes:
1.  **P_RADIUS:** Planetary Radius (Earth Radii)
2.  **P_MASS:** Planetary Mass (Earth Masses)
3.  **P_GRAVITY:** Planetary Gravity (Earth g)
4.  **P_PERIOD:** Orbital Period (Days)
5.  **P_TEMP_EQUIL:** Equilibrium Temperature (Kelvin)
6.  **S_MASS:** Stellar Mass (Solar Mass)
7.  **S_RADIUS:** Stellar Radius (Solar Radii)
8.  **S_TEMPERATURE:** Stellar Temperature (Kelvin)
9.  **S_LUMINOSITY:** Stellar Luminosity (Solar Luminosity)


## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/ExoHabitAI.git](https://github.com/yourusername/ExoHabitAI.git)
    cd ExoHabitAI
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```


## 📂 Project Structure
```text
ExoHabitAI/
├── app.py                 # Main Flask application
├── exo_model.pkl          # Trained ML Model
├── requirements.txt       # Python dependencies
├── Procfile               # Deployment command
├── static/
│   ├── style.css          # Stylesheet
│   ├── script.js          # Charting & logic
│   └── bg.png             # Background image
└── templates/
    └── index.html         # Frontend HTML
```

## ☁️ Deployment
This project is deployed on render .
