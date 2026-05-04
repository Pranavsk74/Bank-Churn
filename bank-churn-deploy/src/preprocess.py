import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Feature Engineering: Balance per product
        if 'Balance' in X.columns and 'NumOfProducts' in X.columns:
            X['BalancePerProduct'] = X['Balance'] / X['NumOfProducts'].replace(0, 1)
        
        # Feature Engineering: Age groups
        if 'Age' in X.columns:
            X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 30, 40, 50, 60, 120], labels=[0, 1, 2, 3, 4]).astype(float)
            
        # Feature Engineering: Tenure buckets
        if 'Tenure' in X.columns:
            X['TenureBucket'] = pd.cut(X['Tenure'], bins=[-1, 2, 5, 8, 12], labels=[0, 1, 2, 3]).astype(float)
            
        # Removed manual Gender encoding, using OneHotEncoder instead
                
        # Drop irrelevant columns
        # Dropping Complain to avoid data leakage since it perfectly correlates with churning in some versions of this dataset.
        cols_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Complain']
        cols_to_drop = [c for c in cols_to_drop if c in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        return X

def get_preprocessor():
    """Returns the ColumnTransformer for preprocessing."""
    # Define columns to be processed
    categorical_cols = ['Geography', 'Card Type', 'Gender'] # Geography, Card Type, Gender get OneHotEncoded
    
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                      'EstimatedSalary', 'Satisfaction Score', 'Point Earned',
                      'BalancePerProduct', 'AgeGroup', 'TenureBucket']
    
    # Binary/Already encoded
    passthrough_cols = ['HasCrCard', 'IsActiveMember']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('pass', 'passthrough', passthrough_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor
