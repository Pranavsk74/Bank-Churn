import os
import joblib
import pandas as pd

def load_model(model_path=None):
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'model', 'churn_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
    return joblib.load(model_path)

def predict_churn(model, input_data):
    """
    input_data: dictionary with feature names as keys and single values
    Returns: probability of churn
    """
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Predict probability of class 1 (churn)
    proba = model.predict_proba(df)[0, 1]
    
    return proba

def get_business_insights(proba, input_data):
    """
    Determine risk category, revenue at risk and recommendations based on probability and inputs.
    """
    # Segment risk
    if proba > 0.7:
        risk_category = "High Risk"
    elif proba > 0.4:
        risk_category = "Medium Risk"
    else:
        risk_category = "Low Risk"
        
    # Revenue at risk estimation
    # Example logic: Assume 5% of balance + 10% of estimated salary is core revenue value
    balance = input_data.get('Balance', 0)
    salary = input_data.get('EstimatedSalary', 0)
    
    revenue_at_risk = (balance * 0.05) + (salary * 0.1)
    
    # Value Segmentation
    is_high_value = revenue_at_risk > 15000
    
    # Recommendation
    if risk_category == "High Risk" and is_high_value:
        recommendation = "High Priority: Offer a premium retention package or personalized account management."
    elif risk_category == "High Risk" and not is_high_value:
        recommendation = "Low Priority: Send automated re-engagement email; limit manual outreach costs."
    elif risk_category == "Medium Risk" and is_high_value:
        recommendation = "Monitor Closely: Assign account manager to check in or offer a targeted product."
    elif risk_category == "Medium Risk" and not is_high_value:
        recommendation = "Standard Retention: Send educational material on underutilized bank products."
    else:
        recommendation = "Maintain Engagement: Customer is stable. Standard marketing flow applies."
        
    return {
        "probability": float(proba),
        "risk_category": risk_category,
        "revenue_at_risk": float(revenue_at_risk),
        "recommendation": recommendation
    }
