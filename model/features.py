"""
Weekly features for the financial-distress risk model.

build_features(transactions) turns raw transactions into one row per person per week.
Each row only uses data up to and including that week, so the model can score a
customer "as of now" with no peeking into the future.

Works on both the synthetic CSV (has a `week` column) and transactions parsed from
statement PDFs (no `week` column: weeks are counted from the first Monday).
"""
import numpy as np
import pandas as pd

INCOME = ["salary", "universal_credit"]
ESSENTIALS = ["rent", "council_tax", "energy", "phone"]

FEATURES = [
    "balance", "min_balance_4w", "weeks_overdrawn_8w",
    "net_4w", "net_8w", "net_trend_4w", "weeks_net_negative_8w",
    "income_ratio", "on_universal_credit",
    "payday_loans_8w", "payday_repayments_8w", "bnpl_8w",
    "failed_payments_4w", "failed_payments_8w", "overdraft_fees_4w",
    "gambling_share_8w", "gambling_trend_4w",
    "essentials_share_8w", "discretionary_share_8w",
    "months_runway",
]


def _roll(g, col, k, how="sum"):
    r = g[col].rolling(k, min_periods=1)
    return getattr(r, how)().reset_index(level=0, drop=True)


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    if "week" not in tx.columns:
        first = tx.groupby("person_id")["date"].transform("min")
        monday = first - pd.to_timedelta(first.dt.weekday, unit="D")
        tx["week"] = (tx["date"] - monday).dt.days // 7

    tx["abs"] = tx["amount"].abs()
    sums = tx.pivot_table(index=["person_id", "week"], columns="category",
                          values="abs", aggfunc="sum", fill_value=0)
    counts = tx.pivot_table(index=["person_id", "week"], columns="category",
                            values="abs", aggfunc="count", fill_value=0)
    spend = (-tx["amount"].clip(upper=0)).groupby([tx.person_id, tx.week]).sum()
    bal = tx.sort_values(["person_id", "date"], kind="stable").groupby(["person_id", "week"])["balance_after"].last()

    # full grid so quiet weeks still exist
    grid = pd.MultiIndex.from_tuples(
        [(p, w) for p, wmax in tx.groupby("person_id")["week"].max().items() for w in range(int(wmax) + 1)],
        names=["person_id", "week"])
    df = pd.DataFrame(index=grid)

    def col(frame, name):
        return frame[name] if name in frame.columns else 0.0

    df["income"] = sum(col(sums, c) for c in INCOME)
    df["uc"] = col(sums, "universal_credit")
    df["spend"] = spend
    df["essentials"] = sum(col(sums, c) for c in ESSENTIALS)
    df["discretionary"] = col(sums, "discretionary")
    df["gambling"] = col(sums, "gambling")
    df["bnpl"] = col(sums, "bnpl_repayment")
    df["payday_repay"] = col(sums, "payday_loan_repayment")
    df["fees"] = col(sums, "overdraft_fee")
    df["loans"] = col(counts, "payday_loan")
    df["failed"] = col(counts, "failed_payment")
    df["balance"] = bal
    df = df.reindex(grid)
    df["balance"] = df.groupby(level=0)["balance"].ffill().fillna(0)
    df = df.fillna(0).reset_index()

    df["net"] = df["income"] - df["spend"]
    df["overdrawn"] = (df["balance"] < 0).astype(int)
    g = df.groupby("person_id")

    out = df[["person_id", "week", "balance"]].copy()
    out["min_balance_4w"] = _roll(g, "balance", 4, "min")
    out["weeks_overdrawn_8w"] = _roll(g, "overdrawn", 8)
    net4, net8 = _roll(g, "net", 4), _roll(g, "net", 8)
    out["net_4w"], out["net_8w"] = net4, net8
    out["net_trend_4w"] = net4 - (net8 - net4)
    df["net4_neg"] = (net4 < 0).astype(int)
    out["weeks_net_negative_8w"] = _roll(df.groupby("person_id"), "net4_neg", 8)

    inc8, inc16 = _roll(g, "income", 8), _roll(g, "income", 16)
    prev8 = inc16 - inc8
    out["income_ratio"] = np.where(prev8 > 0, inc8 / prev8.replace(0, np.nan), 1.0)
    out["on_universal_credit"] = (_roll(g, "uc", 8) > 0).astype(int)

    out["payday_loans_8w"] = _roll(g, "loans", 8)
    out["payday_repayments_8w"] = _roll(g, "payday_repay", 8)
    out["bnpl_8w"] = _roll(g, "bnpl", 8)
    out["failed_payments_4w"] = _roll(g, "failed", 4)
    out["failed_payments_8w"] = _roll(g, "failed", 8)
    out["overdraft_fees_4w"] = _roll(g, "fees", 4)

    gam4, gam8 = _roll(g, "gambling", 4), _roll(g, "gambling", 8)
    denom = inc8 + 1.0
    out["gambling_share_8w"] = gam8 / denom
    out["gambling_trend_4w"] = gam4 - (gam8 - gam4)
    out["essentials_share_8w"] = _roll(g, "essentials", 8) / denom
    out["discretionary_share_8w"] = _roll(g, "discretionary", 8) / denom

    # months until the money runs out at the current monthly shortfall (12 = not running out)
    shortfall = -net8 / 2.0
    out["months_runway"] = np.where(shortfall > 0, (out["balance"] / shortfall.where(shortfall > 0)).clip(-3, 12), 12.0)

    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    return out
