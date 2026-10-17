#!/usr/bin/env python3
"""
Train the financial-distress risk model.

Predicts: will this customer miss rent in the next 6 weeks?  (crisis_within_6w)

Backend, in order of preference:
  1. IBM Snap ML  (IBM's ML library, accelerated on IBM Z)   pip install snapml
  2. XGBoost                                                pip install xgboost
  3. scikit-learn HistGradientBoosting                      always there as a fallback

Usage (from the repo root):
  python3 -m model.train --data data --out model/risk_model.pkl
  python3 -m model.train --backend xgboost
"""
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from model.features import FEATURES, build_features
from model.risk import ALERT_THRESHOLD, RiskModel


def make_model(backend, pos_weight):
    if backend in ("auto", "snapml"):
        try:
            from snapml import BoostingMachineClassifier
            return "snapml", BoostingMachineClassifier(
                num_round=300, max_depth=5, learning_rate=0.1, n_jobs=4)
        except ImportError:
            if backend == "snapml":
                raise
    if backend in ("auto", "xgboost"):
        try:
            from xgboost import XGBClassifier
            return "xgboost", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
                colsample_bytree=0.9, scale_pos_weight=1.0, eval_metric="aucpr", n_jobs=4)
        except ImportError:
            if backend == "xgboost":
                raise
    from sklearn.ensemble import HistGradientBoostingClassifier
    return "sklearn", HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.08)


def lead_time(test_scores, weekly):
    """How many weeks before the first missed rent did we first raise an alert?"""
    first_miss = weekly[weekly.missed_rent == 1].groupby("person_id").week.min()
    leads, caught = [], 0
    for pid, miss_week in first_miss.items():
        s = test_scores[(test_scores.person_id == pid) & (test_scores.week < miss_week)]
        if s.empty:
            continue
        # alert must be sustained for 2 weeks running to count (less noisy)
        alert = (s.risk >= ALERT_THRESHOLD).astype(int).rolling(2).sum() >= 2
        if alert.any():
            caught += 1
            leads.append(int(miss_week - s.loc[alert.idxmax(), "week"]))
    n = len([p for p in first_miss.index if p in set(test_scores.person_id)])
    return leads, caught, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model/risk_model.pkl")
    ap.add_argument("--backend", default="auto", choices=["auto", "snapml", "xgboost", "sklearn"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"Machine: {platform.machine()}  ({platform.system()} {platform.release()})")
    tx = pd.read_csv(f"{args.data}/transactions.csv")
    weekly = pd.read_csv(f"{args.data}/weekly.csv")

    t0 = time.time()
    feats = build_features(tx)
    df = feats.merge(weekly[["person_id", "week", "crisis_within_6w", "already_missed_rent", "label_complete"]],
                     on=["person_id", "week"])
    print(f"Features: {len(df):,} person-weeks x {len(FEATURES)} features  ({time.time() - t0:.1f}s)")

    # train on weeks where the label is known and the person hasn't already hit crisis
    usable = df[(df.label_complete == 1) & (df.already_missed_rent == 0) & (df.week >= 4)]

    # split by PERSON so we test on customers the model has never seen
    rng = np.random.default_rng(args.seed)
    people = df.person_id.unique()
    test_people = set(rng.choice(people, size=len(people) // 4, replace=False))
    train = usable[~usable.person_id.isin(test_people)]
    test = usable[usable.person_id.isin(test_people)]
    y_tr, y_te = train.crisis_within_6w.values, test.crisis_within_6w.values
    pos_weight = (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())
    print(f"Train: {len(train):,} rows ({y_tr.mean():.1%} positive) | Test: {len(test):,} rows, {len(test_people)} unseen people")

    backend, clf = make_model(args.backend, pos_weight)
    t0 = time.time()
    clf.fit(train[FEATURES].to_numpy(np.float32), y_tr)
    train_s = time.time() - t0
    rm = RiskModel(backend, clf)

    p = rm.predict(test)
    t0 = time.time()
    rm.predict(test)
    per_txn_ms = (time.time() - t0) / len(test) * 1000

    alert = p >= ALERT_THRESHOLD
    metrics = {
        "backend": backend,
        "machine": platform.machine(),
        "train_seconds": round(train_s, 2),
        "score_ms_per_customer_week": round(per_txn_ms, 4),
        "roc_auc": round(roc_auc_score(y_te, p), 3),
        "pr_auc": round(average_precision_score(y_te, p), 3),
        "alert_threshold": ALERT_THRESHOLD,
        "precision_at_alert": round(precision_score(y_te, alert), 3),
        "recall_at_alert": round(recall_score(y_te, alert), 3),
    }

    # early-warning headline numbers on every week of the test people
    all_test = df[df.person_id.isin(test_people)].copy()
    all_test["risk"] = rm.predict(all_test)
    leads, caught, n = lead_time(all_test, weekly)
    never = weekly[weekly.person_id.isin(test_people)].groupby("person_id").missed_rent.max()
    never = never[never == 0].index
    fa = all_test[all_test.person_id.isin(never)]
    false_alarm = (fa.assign(a=fa.risk >= ALERT_THRESHOLD).groupby("person_id").a
                   .apply(lambda s: (s.rolling(2).sum() >= 2).any()).mean())
    metrics.update({
        "crises_in_test": n,
        "crises_caught_early": caught,
        "share_caught_early": round(caught / max(1, n), 3),
        "median_weeks_warning": float(np.median(leads)) if leads else None,
        "false_alarm_rate_people": round(float(false_alarm), 3),
    })

    imp = getattr(clf, "feature_importances_", None)
    if imp is not None:
        top = sorted(zip(FEATURES, imp), key=lambda x: -x[1])[:8]
        metrics["top_features"] = {k: round(float(v), 4) for k, v in top}

    rm.save(args.out)
    with open(args.out.replace(".pkl", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBackend: {backend}  | trained in {train_s:.1f}s")
    print(f"ROC AUC {metrics['roc_auc']}  | PR AUC {metrics['pr_auc']}")
    print(f"At alert threshold {ALERT_THRESHOLD}: precision {metrics['precision_at_alert']}, recall {metrics['recall_at_alert']}")
    print(f"Early warning: caught {caught}/{n} crises ({metrics['share_caught_early']:.0%}) "
          f"a median of {metrics['median_weeks_warning']} weeks before missed rent")
    print(f"False alarms: {metrics['false_alarm_rate_people']:.0%} of people who never missed rent")
    print(f"Scoring speed: {per_txn_ms:.4f} ms per customer-week")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
