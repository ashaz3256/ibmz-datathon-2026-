"""
Bridge between Sourov's risk model (model/Predictor.py) and the Granite agent.

Uses Sourov's get_crisis_predictions() and saved XGBoost model unchanged, and adds
what the agent and front end need on top:
  - a risk level (Steady / Stretched / Under pressure / At risk)
  - the top reasons, from XGBoost's per-feature contributions for that week
  - what-ifs (e.g. remove gambling spend and re-score)
  - the person's bills, priority ones first

Usage:
    python3 agent/risk_bridge.py 42         # score customer 42 and print the result
    python3 agent/risk_bridge.py 42 --agent # ...and ask Granite to write the message
"""

import inspect
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))

# Risk bands agreed for the project (model/README.md)
LEVELS = [(0.10, "Steady"), (0.25, "Stretched"), (0.50, "Under pressure"), (1.01, "At risk")]
ALERT_AT = 0.25  # "Under pressure" and above triggers the early-warning alert
WARMUP_WEEKS = 4  # the model needs a few weeks of history before alerts count

FIRST_NAMES = ["Jordan", "Priya", "Sam", "Aisha", "Callum", "Mei", "Tom", "Zara",
               "Liam", "Hannah", "Omar", "Grace", "Ryan", "Fatima", "Ben", "Chloe"]


def level_for(p):
    for cut, name in LEVELS:
        if p < cut:
            return name
    return "At risk"


# ---------- load Sourov's Predictor without editing it ----------

def _load_predictor():
    """Imports model/Predictor.py, fixing its Windows-style model path so it runs on Linux."""
    src = (MODEL_DIR / "Predictor.py").read_text()
    model_path = str(MODEL_DIR / "crisis_prediction_model.json")
    # Works whether Predictor.py says 'model\\crisis_prediction_model.json', 'model/...' or just the file name
    src = re.sub(r"""['"](?:model[\\/]+)?crisis_prediction_model\.json['"]""", repr(model_path), src)
    mod = types.ModuleType("Predictor")
    mod.__file__ = str(MODEL_DIR / "Predictor.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


_predictor = None


def predictor():
    global _predictor
    if _predictor is None:
        _predictor = _load_predictor()
    return _predictor


class _CaptureFeatures:
    """Wraps the XGBoost model so we can see the exact feature rows Sourov's code builds."""
    def __init__(self, model):
        self.model = model
        self.feature_names_in_ = model.feature_names_in_
        self.X = None

    def predict_proba(self, X):
        self.X = X.copy()
        return self.model.predict_proba(X)


# ---------- data ----------

_data_cache = {}


def load_data(data_dir=DATA_DIR):
    key = str(data_dir)
    if key not in _data_cache:
        d = Path(data_dir)
        missing = [f for f in ("weekly.csv", "people.csv", "transactions.csv") if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {missing} in {d}. Generate them with: python3 data_gen/gen_personas.py --out data")
        _data_cache[key] = (pd.read_csv(d / "weekly.csv"),
                            pd.read_csv(d / "people.csv"),
                            pd.read_csv(d / "transactions.csv"))
    return _data_cache[key]


def person_frames(person_id, data_dir=DATA_DIR, transactions=None):
    weekly, people, tx = load_data(data_dir)
    w = weekly[weekly.person_id == person_id].sort_values("week").reset_index(drop=True)
    p = people[people.person_id == person_id].reset_index(drop=True)
    t = tx[tx.person_id == person_id] if transactions is None else transactions
    if w.empty or p.empty:
        raise ValueError(f"No customer with person_id {person_id}")
    return w, p, t.reset_index(drop=True)


def _score(w, p, t):
    pred = predictor()
    real_model = pred.model
    cap = _CaptureFeatures(real_model)
    if len(inspect.signature(pred.get_crisis_predictions).parameters) >= 4:
        res = pred.get_crisis_predictions(w, p, t, cap)    # older version: model passed in
    else:
        pred.model = cap                                     # newer version: uses its own global model
        try:
            res = pred.get_crisis_predictions(w, p, t)
        finally:
            pred.model = real_model
    return res.reset_index(drop=True), cap.X.reset_index(drop=True)


# ---------- reasons ----------

def _money(x):
    return f"£{abs(x):,.0f}"


def _reason_text(feat, row, usual):
    v, u = row[feat], usual.get(feat, np.nan)
    usual_txt = f" (usually {_money(u)})" if pd.notna(u) and abs(u) >= 1 else ""
    cat = feat.replace("spend_", "")
    if feat == "missed_rent":
        return "rent was missed this week" if v > 0 else None
    if feat == "balance":
        return f"balance is down to {'-' if v < 0 else ''}{_money(v)}" if v < (u if pd.notna(u) else 1e9) else None
    if feat == "available_liquidity":
        return f"only {_money(v)} left including the overdraft"
    if feat == "credit_utilization":
        return f"using {abs(v) * 100:.0f}% of the overdraft" if row.get("balance", 0) < 0 else None
    if feat == "rent_to_income_ratio":
        return f"rent takes {v * 100:.0f}% of monthly income"
    if feat == "spend_failed_payment":
        return "a payment bounced this week" if v != 0 else None
    if feat == "spend_overdraft_fee":
        return f"overdraft fees of {_money(v)} this week" if v != 0 else None
    if feat == "spend_payday_loan":
        return f"took out a payday loan of {_money(v)}" if v != 0 else None
    if feat in ("spend_payday_loan_repayment", "spend_bnpl_repayment"):
        name = "payday loan" if "payday" in feat else "buy now pay later"
        if v == 0 or (pd.notna(u) and abs(v) <= 1.2 * abs(u)):
            return None
        return f"{name} repayments of {_money(v)} this week{usual_txt}"
    if feat == "spend_salary":
        if pd.notna(u) and abs(v) < 0.8 * abs(u):
            return f"income this week {_money(v)}{usual_txt}"
        return None
    if feat == "spend_universal_credit":
        return "now receiving Universal Credit" if v != 0 else None
    if feat.startswith("spend_"):
        if v == 0 or (pd.notna(u) and abs(v) <= 1.2 * abs(u)):
            return None  # not unusual for them, so not a reason to worry
        return f"{cat.replace('_', ' ')} spend {_money(v)} this week{usual_txt}"
    if feat == "total_spent":
        if v >= 0 or (pd.notna(u) and abs(v) <= 1.2 * abs(u)):
            return None
        return f"net spending of {_money(v)} this week{usual_txt}"
    return None


def top_reasons(X, week_idx, n=3):
    booster = pred_model().get_booster()
    contribs = booster.predict(xgb.DMatrix(X.iloc[[week_idx]]), pred_contribs=True)[0][:-1]
    row = X.iloc[week_idx]
    past = X.iloc[max(0, week_idx - 12):week_idx]
    usual = past.median(numeric_only=True) if len(past) else pd.Series(dtype=float)
    reasons = []
    for i in np.argsort(contribs)[::-1]:
        if contribs[i] <= 0:
            break
        text = _reason_text(X.columns[i], row, usual)
        if text and text not in reasons:
            reasons.append(text)
        if len(reasons) == n:
            break
    return reasons


def pred_model():
    return predictor().model


# ---------- what-ifs and bills ----------

CUTTABLE = {"gambling": "stop gambling spend", "discretionary": "cut back on non-essential spending"}


def what_ifs(w, p, t, week_idx, base_p, weeks_back=8):
    """Re-scores as if they had stopped that spending for the last 8 weeks (money stays in the account)."""
    out = []
    base_level = level_for(base_p)
    week = int(w.week.iloc[week_idx])
    for cat, label in CUTTABLE.items():
        window = (t.category == cat) & (t.week > week - weeks_back) & (t.week <= week)
        if not window.any():
            continue
        saved = t[window].groupby("week").amount.sum().abs()
        w2 = w.copy()
        w2["balance"] = w2.balance + w2.week.map(saved.cumsum().reindex(w2.week, method="ffill")).fillna(0)
        res, _ = _score(w2, p, t[~window])
        new_p = float(res.crisis_probability.iloc[week_idx])
        if new_p < base_p * 0.8:
            out.append({"change": label, "from": base_level, "to": level_for(new_p),
                        "from_p": base_p, "to_p": new_p, "saved": float(saved.sum())})
    return sorted(out, key=lambda x: x["to_p"])  # biggest improvement first


def bills(w, p, t, week):
    recent = t[(t.week > week - 8) & (t.week <= week)]

    def spent(cat, weeks=4):
        r = t[(t.week > week - weeks) & (t.week <= week) & (t.category == cat)]
        return abs(r.amount.sum())

    out = [{"type": "rent", "amount": float(p.monthly_rent.iloc[0])}]
    if spent("council_tax"):
        out.append({"type": "council tax", "amount": spent("council_tax")})
    if spent("energy"):
        out.append({"type": "energy", "amount": spent("energy")})
    bal = float(w.loc[w.week == week, "balance"].iloc[0])
    if bal < 0:
        out.append({"type": "overdraft", "amount": -bal})
    if (recent.category == "payday_loan").any():
        out.append({"type": "payday loan", "amount": abs(recent[recent.category == "payday_loan"].amount.sum())})
    if spent("bnpl_repayment"):
        out.append({"type": "buy now pay later", "amount": spent("bnpl_repayment")})
    return [{"type": b["type"], "amount": round(float(b["amount"]), 2)} for b in out]


# ---------- main entry point ----------

def score_person(person_id, week=None, data_dir=DATA_DIR, transactions=None):
    """
    Scores one customer with Sourov's model and returns everything the agent and UI need.
    week: which week counts as "now" (default: their latest week).
    transactions: optional dataframe of this person's transactions (e.g. loaded in the UI).
    """
    w, p, t = person_frames(person_id, data_dir, transactions)
    res, X = _score(w, p, t)
    if week is None:
        week = int(res.week.iloc[-1])
    idx = int(res.index[res.week == week][0])
    prob = float(res.crisis_probability.iloc[idx])

    hi = (res.crisis_probability >= ALERT_AT) & (res.week >= WARMUP_WEEKS)
    alerts = res[hi & hi.shift(-1, fill_value=False)]
    missed = w[w.get("missed_rent", 0) == 1] if "missed_rent" in w else w.iloc[0:0]
    first_alert = int(alerts.week.iloc[0]) if len(alerts) else None
    first_missed = int(missed.week.iloc[0]) if len(missed) else None

    case = {
        "person_id": int(person_id),
        "name": FIRST_NAMES[int(person_id) % len(FIRST_NAMES)],
        "week": week,
        "risk_score": prob,
        "risk_level": level_for(prob),
        "top_reasons": top_reasons(X, idx),
        "debts": bills(w, p, t, week),
        "what_ifs": what_ifs(w, p, t, idx, prob) if prob >= 0.10 else [],
        "first_alert_week": first_alert,
        "first_missed_rent_week": first_missed,
    }
    # every week's result from Sourov's Predictor (balance, probability, his high-risk flag)
    keep = [c for c in ("week", "balance", "crisis_probability", "high_risk_flag", "missed_rent") if c in res]
    weekly_risk = res[keep].copy()
    return case, weekly_risk


def estimate_any_csv(rows):
    """Ismail's method for scoring ANY transactions CSV (one person, no income/rent details):
    estimate the model's inputs from the transactions alone, then run Sourov's model.
    rows: list of dicts with date, amount, category, balance_after and optionally week.
    Scores are estimates: the model was trained on real income, rent and overdraft figures."""
    tx = pd.DataFrame(rows)
    tx["person_id"] = 1
    if "week" not in tx or tx["week"].isna().all() or (tx["week"] == 0).all():
        dates = pd.to_datetime(tx["date"], errors="coerce", format="ISO8601")  # 2026-01-05
        if dates.isna().mean() > 0.5:
            dates = pd.to_datetime(tx["date"], errors="coerce", dayfirst=True)  # UK style 05/01/2026
        tx["week"] = ((dates - dates.min()).dt.days // 7).fillna(0).astype(int) if dates.notna().any() else 0
    tx["week"] = tx["week"].astype(int)
    tx["category"] = tx["category"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    weeks = max(tx.week.nunique(), 1)
    per_month = 52 / 12 / weeks
    income = tx.amount[tx.amount > 0].sum() * per_month
    rent_rows = tx[tx.category.str.contains("rent")]
    # rent from rent payments if there are any; otherwise Ismail's fallback (all spending)
    rent = (abs(rent_rows.amount.sum()) if len(rent_rows) else abs(tx.amount[tx.amount < 0].sum())) * per_month
    weekly = (tx.sort_values("week").groupby("week").agg(balance=("balance_after", "last")).reset_index())
    weekly["person_id"] = 1
    people = pd.DataFrame([{"person_id": 1, "monthly_income": float(income),
                            "monthly_rent": float(rent), "overdraft_limit": 500.0}])  # Ismail's default limit
    res, X = _score(weekly, people, tx)
    last = len(res) - 1
    top = res.loc[res.crisis_probability.idxmax()]
    return {
        "risk_score": float(res.crisis_probability.iloc[last]), "week": int(res.week.iloc[last]),
        "risk_level": level_for(float(res.crisis_probability.iloc[last])),
        "max_score": float(top.crisis_probability), "max_week": int(top.week),
        "top_reasons": top_reasons(X, last),
        "estimated": {"monthly_income": round(float(income), 2), "monthly_rent": round(float(rent), 2),
                      "rent_from": "rent payments" if len(rent_rows) else "all spending (no rent found)",
                      "overdraft_limit": 500.0},
        "weekly": res[[c for c in ("week", "balance", "crisis_probability", "high_risk_flag") if c in res]],
    }


def demo_people(data_dir=DATA_DIR, n=3):
    """A few interesting customers to demo: ones who were flagged before they missed rent."""
    _, people, _ = load_data(data_dir)
    picks = []
    for persona in ("gambling", "job_loss", "rent_shock", "near_miss", "steady"):
        ids = people[people.persona == persona].person_id.head(n).tolist()
        picks += [(int(i), persona) for i in ids]
    return picks


if __name__ == "__main__":
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else int(demo_people()[0][0])
    case, wr = score_person(pid)
    print(f"Customer {pid} ({case['name']}), week {case['week']}")
    print(f"  Risk: {case['risk_level']} ({case['risk_score']:.2f})")
    print(f"  First alert week: {case['first_alert_week']}   First missed rent week: {case['first_missed_rent_week']}")
    print("  Reasons:", *[f"\n    - {r}" for r in case["top_reasons"]])
    print("  Bills:", case["debts"])
    print("  What-ifs:", case["what_ifs"])
    if "--agent" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent))
        from agent import run_agent
        msg, src = run_agent(case)
        print(f"\n--- message (via {src}) ---\n{msg}")
