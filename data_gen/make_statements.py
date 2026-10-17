#!/usr/bin/env python3
"""
Turn synthetic transactions into monthly bank-statement PDFs (one PDF per person).

The bank ("Northway Bank") is fictional, and every page is stamped SYNTHETIC so the
files can never be mistaken for a real statement.

Usage:
  python make_statements.py --data data --people 7 29 98 --out statements
  python make_statements.py --data data --persona job_loss --n 3 --out statements

Needs: pandas, reportlab  (pip install reportlab)
"""
import argparse
import os
from datetime import date

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BANK = "Northway Bank"   # fictional
W, H = A4
LEFT, RIGHT = 40, W - 40
ROW = 13

# Statement-style transaction codes (like real UK statements)
CODE = {
    "salary": "BGC", "universal_credit": "BGC", "payday_loan": "FPI",
    "rent": "SO", "council_tax": "DD", "energy": "DD", "phone": "DD",
    "payday_loan_repayment": "DD", "bnpl_repayment": "DD",
    "groceries": "CARD", "transport": "CARD", "discretionary": "CARD", "gambling": "CARD",
    "overdraft_fee": "CHG",
}

def line(d, desc, out, inn, bal):
    """One transaction = one fixed-width text line, so pypdf reads it back as a single line."""
    return f"{d:<12}{desc:<36}{out:>12}{inn:>12}{bal:>13}"


def money(x):
    return f"{x:,.2f}"


def describe(row):
    if row.category == "failed_payment":
        return f"UNPAID {row.merchant.upper()}"
    return f"{CODE.get(row.category, 'TFR')} {row.merchant.upper()}"


def header(c, person_id, month_label, page):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(LEFT, H - 50, BANK)
    c.setFont("Helvetica", 9)
    c.drawString(LEFT, H - 64, "Current Account Statement")
    c.drawRightString(RIGHT, H - 50, f"Customer {person_id:05d}")
    c.drawRightString(RIGHT, H - 64, f"Statement period: {month_label}   Page {page}")
    c.setFont("Courier-Bold", 8.5)
    y = H - 96
    c.drawString(LEFT, y, line("Date", "Description", "Paid out", "Paid in", "Balance"))
    c.line(LEFT, y - 4, RIGHT, y - 4)
    return y - ROW - 4


def footer(c):
    c.setFont("Helvetica-Oblique", 7)
    c.drawCentredString(W / 2, 24, "SYNTHETIC DATA - generated for the IBM Z Datathon 2026. Not a real bank statement.")


def make_pdf(person_id, txns, path):
    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle(f"{BANK} statement - customer {person_id:05d}")
    txns = txns.sort_values(["date"], kind="stable").reset_index(drop=True)
    # opening balance of the first transaction = its balance minus its amount
    opening = txns.balance_after.iloc[0] - txns.amount.iloc[0]
    txns["month"] = txns.date.str[:7]

    for month, rows in txns.groupby("month", sort=True):
        label = date.fromisoformat(month + "-01").strftime("%B %Y")
        page = 1
        y = header(c, person_id, label, page)
        c.setFont("Courier", 8.5)
        c.drawString(LEFT, y, line("", "BALANCE BROUGHT FORWARD", "", "", money(opening)))
        y -= ROW
        for r in rows.itertuples():
            if y < 50:
                footer(c)
                c.showPage()
                page += 1
                y = header(c, person_id, label, page)
                c.setFont("Courier", 8.5)
            d = date.fromisoformat(r.date).strftime("%d %b %Y")
            out = money(-r.amount) if r.amount < 0 else ""
            inn = money(r.amount) if r.amount > 0 else ""
            c.drawString(LEFT, y, line(d, describe(r)[:34], out, inn, money(r.balance_after)))
            y -= ROW
        opening = rows.balance_after.iloc[-1]
        footer(c)
        c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="statements")
    ap.add_argument("--people", type=int, nargs="*", help="person_ids to export")
    ap.add_argument("--persona", help="pick people of this persona instead")
    ap.add_argument("--n", type=int, default=3, help="how many people when using --persona")
    args = ap.parse_args()

    txns = pd.read_csv(os.path.join(args.data, "transactions.csv"))
    ids = args.people
    if not ids:
        people = pd.read_csv(os.path.join(args.data, "people.csv"))
        if args.persona:
            people = people[people.persona == args.persona]
        ids = people.person_id.head(args.n).tolist()

    os.makedirs(args.out, exist_ok=True)
    for pid in ids:
        path = os.path.join(args.out, f"statement_{pid:05d}.pdf")
        make_pdf(pid, txns[txns.person_id == pid].copy(), path)
        print("wrote", path)


if __name__ == "__main__":
    main()
