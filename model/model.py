import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. Load the Data
# ==========================================
weekly = pd.read_csv('data\\weekly.csv')
transactions = pd.read_csv('data\\transactions.csv')
people = pd.read_csv('data\\people.csv')

# ==========================================
# 2. Feature Engineering: Transactions
# ==========================================
# Calculate total spending and number of transactions per person per week
tx_agg = transactions.groupby(['person_id', 'week']).agg(
    total_spent=('amount', 'sum'),
    tx_count=('amount', 'count')
).reset_index()

# Pivot spending by category to create features like 'spend_groceries', 'spend_transport'
tx_cat = transactions.pivot_table(
    index=['person_id', 'week'], 
    columns='category', 
    values='amount', 
    aggfunc='sum', 
    fill_value=0
).reset_index()

# Rename categorical columns to avoid spaces/issues
tx_cat.columns = [f"spend_{col}" if col not in ['person_id', 'week'] else col for col in tx_cat.columns]

# ==========================================
# 3. Merge Datasets
# ==========================================
# Merge static person data into weekly tracker
df = pd.merge(weekly, people, on='person_id', how='left')

# Merge engineered transaction features
df = pd.merge(df, tx_agg, on=['person_id', 'week'], how='left')
df = pd.merge(df, tx_cat, on=['person_id', 'week'], how='left')

# Weeks with no transactions will have NaNs after merging; fill with 0
df = df.fillna(0) 

# ==========================================
# 4. Feature Engineering: Financial Ratios
# ==========================================
# Only train on rows where the 6-week window actually completed
df = df[df['label_complete'] == 1]

# Create standard risk indicators
df['rent_to_income_ratio'] = df['monthly_rent'] / df['monthly_income']
# Adding a tiny number (1e-5) to avoid dividing by zero if overdraft limit is 0
df['credit_utilization'] = df['balance'] / (df['overdraft_limit'] + 1e-5) 
df['available_liquidity'] = df['balance'] + df['overdraft_limit']

# Convert categorical 'persona' to numeric values for XGBoost
le = LabelEncoder()
if 'persona' in df.columns:
    df['persona_encoded'] = le.fit_transform(df['persona'].astype(str))

# ==========================================
# 5. Model Setup & Train-Test Split
# ==========================================
target = 'crisis_within_6w'

# Drop columns that are identifiers, dates, the target, or leak future info
cols_to_drop = [
    'person_id', 'week', 'week_start', target, 'label_complete', 'persona', 'shock_week'
]
features = [col for col in df.columns if col not in cols_to_drop]

X = df[features]
y = df[target]

# Stratify ensures the same proportion of bankruptcies in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 6. Train the XGBoost Model
# ==========================================
# Calculate weights to handle rare events (imbalanced data)
# Ratio of negative class to positive class
scale_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1

model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_weight, # Forces model to pay more attention to the rare '1's
    random_state=42,
    eval_metric='auc'
)

print("Training model...")
model.fit(X_train, y_train)

# ==========================================
# 7. Evaluate Performance
# ==========================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ROC-AUC is often a better metric than accuracy for imbalanced datasets
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")