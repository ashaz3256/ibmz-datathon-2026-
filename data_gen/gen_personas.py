#!/usr/bin/env python3
"""
Synthetic UK bank-transaction generator for the financial-distress early-warning model.

Simulates people's current accounts week by week. Each person follows a persona
(steady, job_loss, gambling, rent_shock, near_miss). Money troubles emerge naturally
from the rules: overdraft use, payday loans, BNPL, bounced bills and finally missed rent.

Outputs (in --out folder):
  transactions.csv  person_id, date, week, amount, category, merchant, balance_after
  weekly.csv        one row per person-week: balance, missed_rent, crisis_within_6w (the label)
  people.csv        ground truth per person: persona, shock week, base amounts (DON'T train on persona)

Only needs numpy + pandas, so it runs on LinuxONE (s390x) without extra installs.

Usage:
  python gen_personas.py --people 2000 --weeks 52 --seed 42 --out data
"""
import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Share of each persona in the population
PERSONAS = {
    "steady": 0.40,
    "job_loss": 0.15,
    "gambling": 0.12,
    "rent_shock": 0.15,
    "near_miss": 0.18,
}

MERCHANTS = {
    "salary": ["Employer Ltd"],
    "universal_credit": ["DWP Universal Credit"],
    "rent": ["Landlord", "Letting Agent"],
    "council_tax": ["Council Tax"],
    "energy": ["Energy Supplier"],
    "phone": ["Mobile Network"],
    "groceries": ["Tesco", "Aldi", "Asda", "Lidl", "Sainsbury's"],
    "transport": ["TfGM", "Northern Rail", "Fuel Station", "Bus Fare"],
    "discretionary": ["Takeaway", "Clothing Store", "Streaming Service", "Pub", "Online Shop"],
    "gambling": ["BetNow", "SpinCasino", "LuckyOdds"],
    "payday_loan": ["QuickCash Loans", "FastFunds"],
    "payday_loan_repayment": ["QuickCash Loans", "FastFunds"],
    "bnpl_repayment": ["PayLater"],
    "overdraft_fee": ["Bank"],
}

START = date(2026, 1, 5)   # a Monday
CYCLE = 4                  # weeks per pay / bill cycle ("month")
LABEL_HORIZON = 6          # predict missed rent within the next N weeks


def simulate_person(pid, persona, rng, weeks):
    txns, weekly = [], []

    # --- base finances, roughly calibrated to UK figures (tweak with ONS data) ---
    income = round(rng.uniform(1600, 2600), 2)           # monthly net pay
    rent = round(rng.uniform(650, 1100), 2)               # monthly rent
    if persona == "rent_shock":                          # rent shocks bite hardest on tighter budgets
        income = round(rng.uniform(1450, 2000), 2)
        rent = round(rng.uniform(800, 1150), 2)
    ctax = round(rng.uniform(110, 170), 2)
    energy = round(rng.uniform(80, 140), 2)
    phone = round(rng.uniform(20, 45), 2)
    groceries = rng.uniform(45, 85)                       # weekly
    transport = rng.uniform(15, 40)                       # weekly
    fun = rng.uniform(30, 80)                             # weekly discretionary
    gamble = rng.uniform(20, 40) if persona == "gambling" else (rng.uniform(0, 8) if rng.random() < 0.3 else 0.0)
    overdraft = float(rng.choice([0, 250, 500, 1000, 1500]))
    balance = round(rng.uniform(300, 1500) if persona != "steady" else rng.uniform(800, 3000), 2)

    pay_week = int(rng.integers(0, CYCLE))
    rent_week = (pay_week + int(rng.integers(0, 2))) % CYCLE
    bill_week = (pay_week + 1) % CYCLE
    shock = int(rng.integers(8, 32))                      # week the persona's change kicks in
    rent_jump = rng.uniform(1.35, 1.60)
    dip_len = int(rng.integers(5, 10))

    state = {"balance": balance, "last_loan": -99, "day": 0}
    scheduled = []      # (week_due, amount, category, merchant)
    missed_rent = set()

    def add(week, amount, category, merchant=None):
        state["day"] = min(6, state["day"] + int(rng.integers(0, 2)))
        d = START + timedelta(weeks=week, days=state["day"])
        merchant = merchant or rng.choice(MERCHANTS[category])
        txns.append((pid, d.isoformat(), week, round(amount, 2), category, merchant, round(state["balance"], 2)))

    def pay(week, amount, category, essential, merchant=None):
        """Try to pay. Essential bills can trigger a payday loan; non-essentials are skipped."""
        amount = round(amount, 2)
        merchant = merchant or rng.choice(MERCHANTS[category])
        if state["balance"] - amount < -overdraft and essential and week - state["last_loan"] >= CYCLE:
            loan = float(round(rng.uniform(200, 500), -1))
            state["balance"] += loan
            state["last_loan"] = week
            add(week, loan, "payday_loan")
            scheduled.append((week + CYCLE, loan * 1.3, "payday_loan_repayment", txns[-1][5]))
        if state["balance"] - amount >= -overdraft:
            state["balance"] -= amount
            add(week, -amount, category, merchant)
            return True
        if essential:
            add(week, 0.0, "failed_payment", f"{merchant} ({category})")
        return False

    for w in range(weeks):
        state["day"] = 0
        cyc = w % CYCLE

        # --- persona modifiers ---
        inc_mult, rent_mult, energy_mult, fun_mult, gamble_w = 1.0, 1.0, 1.0, 1.0, gamble
        income_cat = "salary"
        if persona == "job_loss" and w >= shock:
            inc_mult, fun_mult, income_cat = 0.35, 0.8, "universal_credit"
        elif persona == "rent_shock" and w >= shock:
            rent_mult, energy_mult = rent_jump, 1.3
        elif persona == "near_miss" and shock <= w < shock + dip_len:
            inc_mult, fun_mult = 0.55, 0.5
        elif persona == "gambling":
            gamble_w = min(400.0, gamble * (1.12 ** max(0, w - shock // 2)))

        # --- income ---
        if cyc == pay_week:
            amt = income * inc_mult * rng.normal(1, 0.03)
            state["balance"] += amt
            add(w, amt, income_cat)
            # Gambling harm: a payday binge before the bills go out
            if persona == "gambling" and gamble_w > 1:
                # treated like a must-pay: gamblers borrow to keep betting
                pay(w, 1.5 * gamble_w * rng.normal(1, 0.2), "gambling", essential=True)

        # --- scheduled repayments (payday loans, BNPL) ---
        due = [s for s in scheduled if s[0] == w]
        scheduled[:] = [s for s in scheduled if s[0] != w]
        for _, amt, cat, merch in due:
            pay(w, amt, cat, essential=True, merchant=merch)

        # --- monthly bills ---
        if cyc == rent_week:
            if not pay(w, rent * rent_mult, "rent", essential=True):
                missed_rent.add(w)
        if cyc == bill_week:
            pay(w, ctax, "council_tax", essential=True)
            pay(w, energy * energy_mult * rng.normal(1, 0.08), "energy", essential=True)
            pay(w, phone, "phone", essential=True)

        # --- weekly spending ---
        pay(w, groceries * rng.normal(1, 0.12), "groceries", essential=True)
        pay(w, transport * rng.normal(1, 0.15), "transport", essential=True)

        spend = fun * fun_mult * rng.normal(1, 0.25)
        if spend > 0:
            if state["balance"] < 0 and rng.random() < 0.6:
                # Buy now, pay later: 3 instalments over the next 6 weeks
                for k in (2, 4, 6):
                    scheduled.append((w + k, spend / 3, "bnpl_repayment", "PayLater"))
            else:
                pay(w, spend, "discretionary", essential=False)

        if gamble_w > 1:
            pay(w, gamble_w * rng.normal(1, 0.2), "gambling", essential=False)

        # --- overdraft interest (~39% EAR, charged weekly) ---
        if state["balance"] < 0:
            fee = abs(state["balance"]) * 0.0065
            state["balance"] -= fee
            add(w, -fee, "overdraft_fee")

        weekly.append((pid, w, (START + timedelta(weeks=w)).isoformat(),
                       round(state["balance"], 2), int(w in missed_rent)))

    person = dict(person_id=pid, persona=persona, shock_week=shock,
                  monthly_income=income, monthly_rent=rent, overdraft_limit=overdraft)
    return txns, weekly, person


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=2000)
    ap.add_argument("--weeks", type=int, default=52)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    names, probs = list(PERSONAS), list(PERSONAS.values())

    all_txns, all_weekly, people = [], [], []
    for pid in range(1, args.people + 1):
        persona = rng.choice(names, p=probs)
        t, wk, p = simulate_person(pid, persona, rng, args.weeks)
        all_txns += t
        all_weekly += wk
        people.append(p)

    txns = pd.DataFrame(all_txns, columns=["person_id", "date", "week", "amount", "category", "merchant", "balance_after"])
    weekly = pd.DataFrame(all_weekly, columns=["person_id", "week", "week_start", "balance", "missed_rent"])
    people = pd.DataFrame(people)

    # --- label: will this person miss rent in the next LABEL_HORIZON weeks? ---
    weekly = weekly.sort_values(["person_id", "week"])
    g = weekly.groupby("person_id")["missed_rent"]
    future = sum(g.shift(-k).fillna(0) for k in range(1, LABEL_HORIZON + 1))
    weekly["crisis_within_6w"] = (future > 0).astype(int)
    weekly["already_missed_rent"] = (g.cumsum() > 0).astype(int)
    # The last weeks can't see the full horizon, so mark them for exclusion from training
    weekly["label_complete"] = (weekly["week"] <= args.weeks - 1 - LABEL_HORIZON).astype(int)

    os.makedirs(args.out, exist_ok=True)
    txns.to_csv(os.path.join(args.out, "transactions.csv"), index=False)
    weekly.to_csv(os.path.join(args.out, "weekly.csv"), index=False)
    people.to_csv(os.path.join(args.out, "people.csv"), index=False)

    # --- sanity summary ---
    hit = weekly.groupby("person_id")["missed_rent"].max().rename("ever_missed_rent")
    summary = people.join(hit, on="person_id").groupby("persona")["ever_missed_rent"].agg(["count", "mean"])
    summary.columns = ["people", "share_missed_rent"]
    print(f"{len(txns):,} transactions, {len(weekly):,} person-weeks, {len(people):,} people -> {args.out}/")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
