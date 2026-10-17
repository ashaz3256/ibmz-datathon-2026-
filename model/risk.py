"""
Scoring helpers used by the agent and dashboard.

    from model.risk import RiskModel
    rm = RiskModel.load("model/risk_model.pkl")
    timeline = rm.score_customer(transactions_for_one_person)   # one row per week
    latest = timeline.iloc[-1]
    latest.level, latest.trend, latest.reasons
    best_actions(rm, transactions_for_one_person, as_of_week=latest.week)

The customer only ever sees a level word, a trend and plain-English reasons.
The raw probability is for the bank's support team.
"""
import pickle

import numpy as np
import pandas as pd

from model.features import FEATURES, build_features

# probability of missing rent in the next 6 weeks -> level shown to the customer
LEVELS = [
    (0.10, "Steady", "You're managing well. Your income covers your essentials."),
    (0.25, "Stretched", "Money's getting tighter. A few things are worth keeping an eye on."),
    (0.50, "Under pressure", "You're spending more than you earn and starting to borrow. Let's make a plan."),
    (1.01, "At risk", "There's a real risk of falling behind on rent soon. Here's what to do first."),
]
ALERT_THRESHOLD = 0.25  # "Under pressure" or worse -> early warning

# what-if scenarios, replayed on the customer's real transactions over the last 8 weeks
SCENARIOS = {
    "stop_gambling": "Stop gambling",
    "pause_bnpl": "Pause buy-now-pay-later repayments",
    "claim_support": "Claim support you may be entitled to (about £150 a month, e.g. council tax reduction)",
}


def level_for(p):
    for cut, name, blurb in LEVELS:
        if p < cut:
            return name, blurb
    return LEVELS[-1][1], LEVELS[-1][2]


def reasons_for(row) -> list:
    """Plain-English reasons, most serious first. Rule-based so they're always explainable."""
    r = []
    if row["failed_payments_4w"] > 0:
        n = int(row["failed_payments_4w"])
        r.append(f"{n} payment{'s' if n > 1 else ''} bounced in the last 4 weeks")
    if row["payday_loans_8w"] > 0:
        r.append("You've taken a payday loan in the last 2 months")
    if row["weeks_net_negative_8w"] >= 4 and row["net_8w"] < 0:
        r.append(f"You've spent about £{abs(row['net_8w']) / 2:,.0f} a month more than you earned recently")
    if row["income_ratio"] < 0.8:
        r.append(f"Your income has dropped by about {100 * (1 - row['income_ratio']):.0f}%")
    if row["weeks_overdrawn_8w"] >= 3:
        r.append(f"You've been overdrawn for {int(row['weeks_overdrawn_8w'])} of the last 8 weeks")
    if row["gambling_share_8w"] > 0.10:
        r.append(f"Gambling is taking about {100 * row['gambling_share_8w']:.0f}% of your income")
    if row["bnpl_8w"] > 0:
        r.append(f"£{row['bnpl_8w']:,.0f} of buy-now-pay-later repayments in the last 2 months")
    if row["essentials_share_8w"] > 0.65:
        r.append(f"Rent and bills take {100 * row['essentials_share_8w']:.0f}% of your income")
    if row["overdraft_fees_4w"] > 10:
        r.append(f"£{row['overdraft_fees_4w']:,.0f} in overdraft charges this month")
    return r


class RiskModel:
    def __init__(self, backend, model, features=FEATURES):
        self.backend, self.model, self.features = backend, model, features

    @classmethod
    def load(cls, path="model/risk_model.pkl"):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["backend"], d["model"], d["features"])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"backend": self.backend, "model": self.model, "features": self.features}, f)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features].to_numpy(dtype=np.float32)
        p = self.model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p

    def score_customer(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Weekly timeline for one customer: probability, level, trend, reasons."""
        f = build_features(tx)
        f["risk"] = self.predict(f)
        f = f[f["week"] >= 4].reset_index(drop=True)   # need a month of history first
        f["level"] = [level_for(p)[0] for p in f["risk"]]
        prev = f.groupby("person_id")["risk"].shift(4)
        delta = (f["risk"] - prev).fillna(0)
        f["trend"] = np.select([delta > 0.10, delta < -0.10],
                               ["Getting harder", "Improving"], "Holding steady")
        f["reasons"] = [reasons_for(row) for _, row in f.iterrows()]
        return f


SCENARIO_APPLIES = {
    "stop_gambling": lambda r: r["gambling_share_8w"] > 0.02,
    "pause_bnpl": lambda r: r["bnpl_8w"] > 0,
    "claim_support": lambda r: True,
}


def _with_weeks(tx):
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    if "week" not in tx.columns:
        first = tx.groupby("person_id")["date"].transform("min")
        monday = first - pd.to_timedelta(first.dt.weekday, unit="D")
        tx["week"] = (tx["date"] - monday).dt.days // 7
    return tx.sort_values("date", kind="stable").reset_index(drop=True)


def apply_scenario(tx: pd.DataFrame, scenario: str, as_of_week: int) -> pd.DataFrame:
    """Rewrite one customer's last 8 weeks as if they'd made the change; balances follow."""
    tx = _with_weeks(tx)
    win = tx["week"].between(as_of_week - 7, as_of_week)
    drop = pd.Series(False, index=tx.index)
    if scenario == "stop_gambling":
        drop = win & (tx.category == "gambling")
    elif scenario == "pause_bnpl":
        drop = win & (tx.category == "bnpl_repayment")
    delta = (-tx["amount"]).where(drop, 0.0)
    if scenario == "claim_support":
        pid = tx["person_id"].iloc[0]
        add = []
        for w in (as_of_week - 7, as_of_week - 3):
            d = tx.loc[tx.week == w, "date"]
            if len(d):
                add.append({"person_id": pid, "date": d.min(), "week": w, "amount": 150.0,
                            "category": "universal_credit", "merchant": "Support claimed", "balance_after": np.nan})
        if add:
            tx = pd.concat([tx, pd.DataFrame(add)], ignore_index=True).sort_values("date", kind="stable")
            tx["balance_after"] = tx["balance_after"].ffill().bfill()
            delta = tx["merchant"].eq("Support claimed") * 150.0
            drop = pd.Series(False, index=tx.index)
    tx["balance_after"] = tx["balance_after"] + delta.reindex(tx.index).fillna(0).cumsum()
    return tx[~drop.reindex(tx.index).fillna(False)]


def what_if(rm: "RiskModel", tx: pd.DataFrame, scenario: str, as_of_week: int) -> dict:
    before = rm.score_customer(tx)
    after = rm.score_customer(apply_scenario(tx, scenario, as_of_week))
    p0 = float(before.loc[before.week == as_of_week, "risk"].iloc[0])
    p1 = float(after.loc[after.week == as_of_week, "risk"].iloc[0])
    return {"scenario": SCENARIOS[scenario], "before": level_for(p0)[0], "after": level_for(p1)[0],
            "risk_before": round(p0, 3), "risk_after": round(p1, 3)}


def best_actions(rm: "RiskModel", tx: pd.DataFrame, as_of_week: int) -> list:
    """What-ifs that apply to this customer, biggest improvement first."""
    row = rm.score_customer(tx).set_index("week").loc[as_of_week]
    out = [what_if(rm, tx, s, as_of_week) for s, applies in SCENARIO_APPLIES.items() if applies(row)]
    return sorted(out, key=lambda d: d["risk_after"])
