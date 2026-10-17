# Risk model

XGBoost model that predicts `crisis_within_6w`: will this customer miss rent in the next 6 weeks?
It looks at 25 things about each week: balance, income, rent, spending by category (groceries,
gambling, payday loans, bounced payments, overdraft fees, ...) and ratios such as rent-to-income.

| File | What it does |
| --- | --- |
| `model.py` / `model.ipynb` | Trains the model on `data/` and saves it |
| `crisis_prediction_model.json` | The trained model |
| `Predictor.py` | `get_crisis_predictions(weekly, people, transactions)` returns a crisis probability for each week of one customer |

The app doesn't call `Predictor.py` directly. It goes through `agent/risk_bridge.py`, which runs it
unchanged and adds the risk level, top reasons, what-ifs and first alert week.

**How well it works:** see the Results section of the main README (`python3 agent/evaluate_model.py`).

Note: the file paths in `model.py` and `Predictor.py` are Windows-style (`data\\weekly.csv`). To retrain
on Linux, change `\\` to `/`. The app works either way.
