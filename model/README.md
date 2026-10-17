# Risk model

Predicts whether a customer will **miss rent in the next 6 weeks**, from their recent transactions.

```bash
python3 -m model.train --data data            # trains, prints metrics, saves model/risk_model.pkl
python3 -m model.train --backend snapml       # force IBM Snap ML (pip install snapml)
```

Backend is picked automatically: **IBM Snap ML** if installed, else XGBoost, else scikit-learn.
Retrain on the LinuxONE VM: the saved model is not committed.

## Files

| File | What it does |
| --- | --- |
| `features.py` | Transactions → 20 weekly features (cash flow, overdraft, payday loans, BNPL, bounced payments, gambling share, months of money left, …). Each week only uses past data. Works on CSV or parsed statement PDFs. |
| `train.py` | Trains and evaluates on customers the model has never seen (split by person) |
| `risk.py` | `RiskModel.score_customer()` → weekly risk, level, trend, plain-English reasons; `best_actions()` → what-if scenarios replayed on the customer's real transactions |

## Results (2,000 synthetic customers, 500 held out)

| Metric | Value |
| --- | --- |
| ROC AUC | 0.96 |
| Crises flagged before rent was missed | 80% |
| Median warning | **5 weeks before missed rent** |
| False alarms (people who never missed rent) | 12% |
| Scoring speed | ~0.002 ms per customer-week |

## Risk levels (what the customer sees)

| Level | Model probability |
| --- | --- |
| Steady | < 0.10 |
| Stretched | 0.10 – 0.25 |
| Under pressure | 0.25 – 0.50 (early-warning alert) |
| At risk | ≥ 0.50 |

## Example

```python
import pandas as pd
from model.risk import RiskModel, best_actions

rm = RiskModel.load("model/risk_model.pkl")
tx = pd.read_csv("data/transactions.csv")
customer = tx[tx.person_id == 31]

timeline = rm.score_customer(customer)          # one row per week
now = timeline[timeline.week == 28].iloc[0]
print(now.level, now.trend, now.reasons)
print(best_actions(rm, customer, as_of_week=28))
# At risk, Getting harder, ['Gambling is taking about 52% of your income', ...]
# [{'scenario': 'Stop gambling', 'before': 'At risk', 'after': 'Steady', ...}]
```
