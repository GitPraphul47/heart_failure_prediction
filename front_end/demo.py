from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load("final_frontend_model.pkl")

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = [float(x) for x in request.form.values()]
    features = np.array(data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    result = "High Risk of Heart Failure" if prediction[0] == 1 else "Low Risk of Heart Failure"

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True, port=8000)
