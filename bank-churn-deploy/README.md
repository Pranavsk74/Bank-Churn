# Bank Customer Churn Intelligence

An end-to-end Machine Learning web application designed to predict bank customer churn and provide actionable business insights to retention teams.

##  Project Overview

Customer churn is a critical metric for banks. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project provides a predictive model and an interactive dashboard to:
1. Predict the probability of a customer leaving the bank (churning).
2. Segment customers based on their risk profile.
3. Estimate the potential revenue at risk.
4. Recommend tailored business actions based on risk and value.

##  Dataset

The model uses the **Bank Customer Churn Dataset**. It contains demographic and financial information of customers, including:
- **Demographics:** Age, Gender, Geography
- **Financials:** Balance, Estimated Salary, Credit Score
- **Engagement:** Tenure, Number of Products, Activity Status, Credit Card Status
- **Target:** `Exited` (1 = Churned, 0 = Retained)

## Approach

1. **Preprocessing & Feature Engineering:**
   - Engineered new features: `BalancePerProduct`, `AgeGroup`, `TenureBucket`
   - Removed irrelevant/leaky columns (`RowNumber`, `CustomerId`, `Surname`, `Complain`).
   - Categorical encoding: One-Hot Encoding for Geography/Card Type, Label Encoding for Gender.
   - Scaled numerical features using `StandardScaler`.
2. **Modeling:**
   - Employed an `sklearn.pipeline.Pipeline` integrating the custom `FeatureEngineer` transformer, `ColumnTransformer`, and a `RandomForestClassifier`.
   - Used class weight balancing to address data skewness.
3. **Evaluation:**
   - Evaluated model on ROC-AUC, Precision, Recall, and F1-Score.
   - High Recall was prioritized to ensure most churning customers are caught.
4. **Deployment:**
   - Built an interactive Streamlit UI (`app.py`).

##  How to Run Locally

### Prerequisites
- Python 3.8+
- `pip`

### Setup

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd bank-churn-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   This will train the Random Forest model and save the artifact to `model/churn_model.pkl`.
   ```bash
   python src/train.py
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app/app.py
   ```

##  Deployment to Streamlit Cloud / Render

This project is structured to be immediately deployable.
1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app and link it to this repository.
3. Set the Main file path to `app/app.py`.
4. Deploy! All dependencies are present in `requirements.txt`.

##  Business Insights

The application goes beyond simple classification by offering a **Business Logic Layer**:
- **High Risk + High Value:** Triggers recommendations for premium retention packages and personalized account management.
- **High Risk + Low Value:** Suggests automated re-engagement to minimize manual effort while attempting retention.
- **Medium Risk:** Recommends standard retention educational content or monitoring.
