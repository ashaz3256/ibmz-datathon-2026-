import io
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    classification_report
)
# ==========================================
# 1. Mock Data Setup (Updated Schema)
# ==========================================
weekly = pd.read_csv('data\\weekly.csv')
transactions = pd.read_csv('data\\transactions.csv')
people = pd.read_csv('data\\people.csv')

# ==========================================
# 2. Transaction Aggregation
# ==========================================
# Aggregate basic counts and sums
tx_agg = transactions.groupby(['person_id', 'week']).agg(
    total_spent=('amount', 'sum'),
    tx_count=('amount', 'count')
).reset_index()

# Pivot categories into separate columns
tx_cat = transactions.pivot_table(
    index=['person_id', 'week'],
    columns='category',
    values='amount',
    aggfunc='sum',
    fill_value=0
).reset_index()
tx_cat.columns = [f"spend_{c}" if c not in ['person_id', 'week'] else c for c in tx_cat.columns]

# ==========================================
# 3. Merging
# ==========================================
df = pd.merge(weekly, people, on='person_id', how='left')
df = pd.merge(df, tx_agg, on=['person_id', 'week'], how='left')
df = pd.merge(df, tx_cat, on=['person_id', 'week'], how='left')

# ==========================================
# 4. Safe Missing Value Handling
# ==========================================
# ONLY fill transaction columns with 0 (missing week in transactions means £0 spent)
tx_cols = ['total_spent', 'tx_count'] + [c for c in df.columns if str(c).startswith('spend_')]
df[tx_cols] = df[tx_cols].fillna(0)

# Calculate financial ratios safely (leaving NaNs intact for XGBoost to handle)
df['rent_to_income_ratio'] = np.where(
    df['monthly_income'] > 0, 
    df['monthly_rent'] / df['monthly_income'], 
    np.nan
)

df['credit_utilization'] = np.where(
    df['overdraft_limit'] > 0, 
    df['balance'] / df['overdraft_limit'], 
    np.nan
)
df['available_liquidity'] = df['balance'] + df['overdraft_limit']

# ==========================================
# 5. Model Setup & Training
# ==========================================
target = 'crisis_within_6w'
cols_to_drop = ['person_id', 'week', target] 
features = [c for c in df.columns if c not in cols_to_drop]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Handle class imbalance natively
pos_count = sum(y_train)
neg_count = len(y_train) - pos_count
scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc',
    missing=np.nan # Explicitly tell XGBoost to look for NaNs
)

model.fit(X_train, y_train)

# ==========================================
# 6. Evaluation & Probabilities
# ==========================================
# Predict probabilities (the likelihood of a 1 / crisis)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

# 3. Print the summary
print("\n=== MODEL PERFORMANCE SUMMARY ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives (Correctly predicted safe):     {cm[0][0]}")
print(f"False Positives (False alarms):                {cm[0][1]}")
print(f"False Negatives (Missed crises):               {cm[1][0]}")
print(f"True Positives (Correctly predicted crisis):   {cm[1][1]}")


model.save_model('model\\crisis_prediction_model.json')
print("Model successfully saved to crisis_prediction_model.json")