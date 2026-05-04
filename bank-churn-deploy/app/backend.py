import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'churn_model.pkl')

import sys
sys.path.append(os.path.join(BASE_DIR, 'src'))

# Load the model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        # Get data from request
        data = request.json
        
        # Ensure input features EXACTLY match model training features
        input_data = {
            'CreditScore': [float(data.get('CreditScore', 0))],
            'Geography': [data.get('Geography', '')],
            'Gender': [data.get('Gender', '')],
            'Age': [float(data.get('Age', 0))],
            'Tenure': [float(data.get('Tenure', 0))],
            'Balance': [float(data.get('Balance', 0.0))],
            'NumOfProducts': [float(data.get('NumOfProducts', 0))],
            'HasCrCard': [int(data.get('HasCrCard', 0))],
            'IsActiveMember': [int(data.get('IsActiveMember', 0))],
            'EstimatedSalary': [float(data.get('EstimatedSalary', 0.0))],
            'Satisfaction Score': [float(data.get('Satisfaction Score', 3))],
            'Card Type': [data.get('Card Type', 'DIAMOND')],
            'Point Earned': [float(data.get('Point Earned', 0))]
        }
        
        # Create DataFrame
        df = pd.DataFrame(input_data)
        
        # Predict probability
        probability = model.predict_proba(df)[0][1]
        
        # Risk Logic
        if probability > 0.7:
            risk_category = "High Risk"
            recommendation = "Offer retention incentive immediately. Assign dedicated account manager."
        elif probability >= 0.4:
            risk_category = "Medium Risk"
            recommendation = "Engage with personalized offers. Send customer satisfaction survey."
        else:
            risk_category = "Low Risk"
            recommendation = "Maintain regular communication. Potential for cross-selling."
            
        return jsonify({
            'probability': round(probability, 4),
            'risk_category': risk_category,
            'recommendation': recommendation
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Add src to python path to ensure preprocess.py can be found during joblib.load
    import sys
    sys.path.append(os.path.join(BASE_DIR, 'src'))
    
    app.run(debug=True, host='127.0.0.1', port=5000)
