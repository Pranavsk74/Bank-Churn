import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.predict import load_model, predict_churn, get_business_insights

# Set page config
st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa421; font-weight: bold; }
    .risk-low { color: #21c354; font-weight: bold; }
    .metric-card {
        background-color: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()

# Header
st.title("🏦 Bank Customer Churn Intelligence")
st.markdown("Predict customer churn probability and generate actionable business insights.")

# Sidebar Inputs
st.sidebar.header("Customer Profile")

col1, col2 = st.sidebar.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=20, value=5)
with col2:
    balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0)
    point_earned = st.number_input("Points Earned", min_value=0, value=500)

geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
card_type = st.sidebar.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])

num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
is_active = st.sidebar.checkbox("Active Member", value=True)
has_crcard = st.sidebar.checkbox("Has Credit Card", value=True)

input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': int(has_crcard),
    'IsActiveMember': int(is_active),
    'EstimatedSalary': salary,
    'Satisfaction Score': 3, # Default mid value
    'Card Type': card_type,
    'Point Earned': point_earned
}

if model is not None:
    # Get Prediction
    proba = predict_churn(model, input_data)
    insights = get_business_insights(proba, input_data)
    
    # Main Dashboard
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    # 1. Churn Probability
    with col1:
        st.subheader("Churn Probability")
        
        # Gauge chart for probability
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = proba * 100,
            number = {'suffix': "%", 'valueformat': ".1f"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "rgba(33, 195, 84, 0.6)"},
                    {'range': [40, 70], 'color': "rgba(255, 164, 33, 0.6)"},
                    {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.6)"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': proba * 100}
            }
        ))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # 2. Risk Category & Revenue at Risk
    with col2:
        st.subheader("Risk Category")
        risk_class = ""
        if insights['risk_category'] == "High Risk":
            risk_class = "risk-high"
        elif insights['risk_category'] == "Medium Risk":
            risk_class = "risk-medium"
        else:
            risk_class = "risk-low"
            
        st.markdown(f"<h2 class='{risk_class}'>{insights['risk_category']}</h2>", unsafe_allow_html=True)
        
        st.metric(label="Estimated Revenue at Risk", value=f"${insights['revenue_at_risk']:,.2f}")
        
    # 3. Recommendation
    with col3:
        st.subheader("Business Action")
        st.info(insights['recommendation'])

    st.markdown("---")
    
    # Feature Importance and Distribution Section
    st.subheader("Model Insights")
    
    tab1, tab2 = st.tabs(["Feature Importance", "Customer Comparison"])
    
    with tab1:
        st.write("Top features driving the churn prediction for the Random Forest Model:")
        try:
            rf = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']
            
            num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                        'EstimatedSalary', 'Satisfaction Score', 'Point Earned',
                        'BalancePerProduct', 'AgeGroup', 'TenureBucket']
            cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
            pass_cols = ['HasCrCard', 'IsActiveMember', 'Gender']
            
            all_features = num_cols + cat_cols + pass_cols
            importances = rf.feature_importances_
            
            feat_df = pd.DataFrame({
                'Feature': all_features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(10)
            
            fig_bar = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Blues')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.warning("Could not extract feature importances. Please ensure model is fully trained.")
            
    with tab2:
        st.write("Comparing current customer's metrics against typical distributions.")
        
        colA, colB = st.columns(2)
        with colA:
            # Dummy data for visualization context
            fig_age = go.Figure()
            fig_age.add_trace(go.Histogram(x=np.random.normal(45, 10, 500), name="Typical Churned", opacity=0.7, marker_color='red'))
            fig_age.add_trace(go.Histogram(x=np.random.normal(35, 10, 500), name="Typical Retained", opacity=0.7, marker_color='green'))
            fig_age.add_vline(x=age, line_dash="dash", line_color="black", annotation_text="This Customer")
            fig_age.update_layout(barmode='overlay', title="Age Distribution Context", height=350)
            st.plotly_chart(fig_age, use_container_width=True)
            
        with colB:
            fig_bal = go.Figure()
            fig_bal.add_trace(go.Histogram(x=np.random.normal(120000, 30000, 500), name="Typical Churned", opacity=0.7, marker_color='red'))
            fig_bal.add_trace(go.Histogram(x=np.random.normal(80000, 40000, 500), name="Typical Retained", opacity=0.7, marker_color='green'))
            fig_bal.add_vline(x=balance, line_dash="dash", line_color="black", annotation_text="This Customer")
            fig_bal.update_layout(barmode='overlay', title="Balance Distribution Context", height=350)
            st.plotly_chart(fig_bal, use_container_width=True)

else:
    st.warning("Model not found. Please run the training script first.")
