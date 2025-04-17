from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("final_frontend_model.pkl")
app = Flask(__name__)
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

    return render_template('predict_Show.html', prediction_text=f'Prediction: {result}')

if __name__=='__main__':
    app.run(debug=True, port=8000)
