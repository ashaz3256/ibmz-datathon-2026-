"""
Measures how well Sourov's model works as an early-warning system, on customers it has never seen.

Generates a fresh set of synthetic customers with a different seed from the training data,
scores every week of every customer, and saves the results to agent/model_performance.json
(the app shows these in its Model Performance panel).

Usage (from the repo folder):
    python3 agent/evaluate_model.py                 # 1,000 new customers, seed 7
    python3 agent/evaluate_model.py --people 500 --seed 11
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import risk_bridge as rb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "model_performance.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7, help="different from the training seed (42)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Generating {args.people} new customers (seed {args.seed})...")
        subprocess.run([sys.executable, str(ROOT / "data_gen" / "gen_personas.py"), "--people", str(args.people),
                        "--seed", str(args.seed), "--out", tmp], check=True, stdout=subprocess.DEVNULL)
        weekly, people, tx = rb.load_data(tmp)

        all_rows, caught, leads, missed_total, alerted_never_missed, never_missed = [], 0, [], 0, 0, 0
        for i, pid in enumerate(people.person_id):
            w = weekly[weekly.person_id == pid].sort_values("week").reset_index(drop=True)
            p = people[people.person_id == pid].reset_index(drop=True)
            t = tx[tx.person_id == pid].reset_index(drop=True)
            res, _ = rb._score(w, p, t)
            res = res.merge(w[["week", "crisis_within_6w", "label_complete", "already_missed_rent"]], on="week")
            all_rows.append(res)

            hi = (res.crisis_probability >= rb.ALERT_AT) & (res.week >= rb.WARMUP_WEEKS)
            alerts = res[hi & hi.shift(-1, fill_value=False)]
            first_alert = int(alerts.week.iloc[0]) if len(alerts) else None
            missed = w[w.missed_rent == 1]
            if len(missed):
                missed_total += 1
                first_missed = int(missed.week.iloc[0])
                if first_alert is not None and first_alert < first_missed:
                    caught += 1
                    leads.append(first_missed - first_alert)
            else:
                never_missed += 1
                alerted_never_missed += first_alert is not None
            if (i + 1) % 200 == 0:
                print(f"  scored {i + 1}/{len(people)} customers")

    rows = pd.concat(all_rows)
    # Same rules as training: only weeks whose 6-week outcome is known, before anyone has missed rent
    fair = rows[(rows.label_complete == 1) & (rows.already_missed_rent == 0)]
    result = {
        "customers": int(len(people)),
        "seed": args.seed,
        "roc_auc": round(float(roc_auc_score(fair.crisis_within_6w, fair.crisis_probability)), 3),
        "missed_rent_customers": missed_total,
        "warned_before_missing_pct": round(100 * caught / missed_total, 1) if missed_total else None,
        "median_weeks_warning": float(np.median(leads)) if leads else None,
        "false_alarm_pct": round(100 * alerted_never_missed / never_missed, 1) if never_missed else None,
        "alert_rule": f"risk >= {rb.ALERT_AT} for 2 weeks in a row, after week {rb.WARMUP_WEEKS}",
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
