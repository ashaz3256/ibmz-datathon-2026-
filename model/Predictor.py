# predictor.py
import pandas as pd
import numpy as np
import xgboost as xgb

# 1. Load the model once when the module is imported
model = xgb.XGBClassifier()
model.load_model('model\\crisis_prediction_model.json')

def get_crisis_predictions(weekly_df, people_df, transactions_df):
    """
    Takes raw dataframes, engineers features, and returns a dataframe with risk probabilities.
    """
    # Aggregate transactions
    tx_agg = transactions_df.groupby(['person_id', 'week']).agg(
        total_spent=('amount', 'sum'),
        tx_count=('amount', 'count')
    ).reset_index()

    tx_cat = transactions_df.pivot_table(
        index=['person_id', 'week'], columns='category', 
        values='amount', aggfunc='sum', fill_value=0
    ).reset_index()
    tx_cat.columns = [f"spend_{c}" if c not in ['person_id', 'week'] else c for c in tx_cat.columns]

    # Merge data
    df = pd.merge(weekly_df, people_df, on='person_id', how='left')
    df = pd.merge(df, tx_agg, on=['person_id', 'week'], how='left')
    df = pd.merge(df, tx_cat, on=['person_id', 'week'], how='left')

    # Handle missing transaction columns the model expects
    expected_features = model.feature_names_in_ 
    for col in expected_features:
        if col not in df.columns and col.startswith('spend_'):
            df[col] = 0.0
            
    tx_cols = ['total_spent', 'tx_count'] + [c for c in df.columns if str(c).startswith('spend_')]
    existing_tx_cols = [c for c in tx_cols if c in df.columns]
    df[existing_tx_cols] = df[existing_tx_cols].fillna(0)

    # Calculate financial ratios
    df['rent_to_income_ratio'] = np.where(df['monthly_income'] > 0, df['monthly_rent'] / df['monthly_income'], np.nan)
    df['credit_utilization'] = np.where(df['overdraft_limit'] > 0, df['balance'] / df['overdraft_limit'], np.nan)
    df['available_liquidity'] = df['balance'] + df['overdraft_limit']

    # Ensure column order matches the trained model
    X_predict = df[expected_features]

    # Predict probabilities
    probabilities = model.predict_proba(X_predict)[:, 1]

    # Create the results dataframe to return
    results = df[['person_id', 'week', 'balance']].copy()
    results['crisis_probability'] = probabilities
    results['high_risk_flag'] = (probabilities > 0.70).astype(int) # Custom threshold
    
    return results