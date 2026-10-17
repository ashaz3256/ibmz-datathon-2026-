# predictor.py
import pandas as pd
import numpy as np
import xgboost as xgb

# 1. Load the model once when the module is imported
model = xgb.XGBClassifier()
model.load_model('model\\crisis_prediction_model.json')

def get_crisis_predictions(weekly, people, transactions):
    xgb_model = model
    # Safety check for memory blowout
    if len(people) > 1:
        raise ValueError("people_df has more than 1 row! Use only single-person data.")

    # 1. Aggregate transactions purely by week
    if not transactions.empty:
        tx_agg = transactions.groupby('week').agg(
            total_spent=('amount', 'sum'), 
            tx_count=('amount', 'count')
        ).reset_index()

        tx_cat = transactions.pivot_table(
            index='week', columns='category', values='amount', aggfunc='sum', fill_value=0
        ).reset_index()
        tx_cat.columns = [f'spend_{c}' if c != 'week' else c for c in tx_cat.columns]
    else:
        tx_agg = pd.DataFrame(columns=['week', 'total_spent', 'tx_count'])
        tx_cat = pd.DataFrame(columns=['week'])

    # 2. Assign demographic info safely directly to weekly data
    df = weekly.copy()
    person_data = people.iloc[0].to_dict()
    for col, val in person_data.items():
        df[col] = val

    # 3. Merge transaction summaries
    df = pd.merge(df, tx_agg, on='week', how='left')
    df = pd.merge(df, tx_cat, on='week', how='left')

    # 4. Align features with the model's expectations (Fixed to initialize all missing expected features)
    expected_features = list(xgb_model.feature_names_in_)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0

    tx_cols = ['total_spent', 'tx_count'] + [c for c in df.columns if str(c).startswith('spend_')]
    df[tx_cols] = df[tx_cols].fillna(0.0)

    # 5. Financial ratios
    df['rent_to_income_ratio'] = np.where(df['monthly_income'] > 0, df['monthly_rent'] / df['monthly_income'], np.nan)
    df['credit_utilization'] = np.where(df['overdraft_limit'] > 0, df['balance'] / df['overdraft_limit'], np.nan)
    df['available_liquidity'] = df['balance'] + df['overdraft_limit']

    # 6. Extract exact feature matrix
    X_predict = df[expected_features]

    # 7. Predict
    probabilities = xgb_model.predict_proba(X_predict)[:, 1]

    # 8. Output results
    results = weekly[['week', 'balance']].copy()
    if 'missed_rent' in weekly.columns:
        results['missed_rent'] = weekly['missed_rent']
    results['crisis_probability'] = probabilities
    results['high_risk_flag'] = (probabilities > 0.70).astype(int)

    return results

if __name__ == "__main__":
    weekly = pd.read_csv('model\\weekly.csv')
    transactions = pd.read_csv('model\\transactions.csv')
    people = pd.read_csv('model\\people.csv')
    print(get_crisis_predictions(weekly, people, transactions))