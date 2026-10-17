#!/usr/bin/env python3
"""
Read bank-statement PDFs with pypdf and turn them back into a transactions table.

This is the "customer uploads a statement" path: PDF -> pypdf text -> parsed rows
-> categorised transactions, ready for the risk model.

Amounts are worked out from the running balance (balance now - balance before),
so we never have to guess which column a number came from.

Usage:
  python read_statements.py statements/statement_00007.pdf
  python read_statements.py statements/*.pdf --out parsed.csv

Needs: pandas, pypdf  (pip install pypdf)
"""
import argparse
import re
from datetime import datetime

import pandas as pd
from pypdf import PdfReader

DATE = r"(\d{2} [A-Z][a-z]{2} \d{4})"
NUM = r"-?[\d,]+\.\d{2}"
CUSTOMER = re.compile(r"Customer (\d+)")

# Merchant keyword -> category (how a bank would categorise free-text descriptions)
CATEGORIES = [
    ("UNPAID", "failed_payment"),
    ("DWP UNIVERSAL CREDIT", "universal_credit"),
    ("EMPLOYER", "salary"),
    ("LANDLORD", "rent"), ("LETTING AGENT", "rent"),
    ("COUNCIL TAX", "council_tax"),
    ("ENERGY", "energy"),
    ("MOBILE NETWORK", "phone"),
    ("PAYLATER", "bnpl_repayment"),
    ("FPI QUICKCASH", "payday_loan"), ("FPI FASTFUNDS", "payday_loan"),
    ("QUICKCASH", "payday_loan_repayment"), ("FASTFUNDS", "payday_loan_repayment"),
    ("BETNOW", "gambling"), ("SPINCASINO", "gambling"), ("LUCKYODDS", "gambling"),
    ("TESCO", "groceries"), ("ALDI", "groceries"), ("ASDA", "groceries"),
    ("LIDL", "groceries"), ("SAINSBURY", "groceries"),
    ("TFGM", "transport"), ("NORTHERN RAIL", "transport"),
    ("FUEL", "transport"), ("BUS FARE", "transport"),
    ("CHG BANK", "overdraft_fee"),
]


def categorise(desc):
    for key, cat in CATEGORIES:
        if key in desc:
            return cat
    return "discretionary"


def num(s):
    return float(s.replace(",", ""))


def parse_pdf(path):
    """pypdf gives each text item on its own line, so read the page as a stream:
    a date starts a row, the next line is the description, the numbers after it are
    the amount column(s) and finally the running balance."""
    reader = PdfReader(path)
    lines = []
    for page in reader.pages:
        lines += [l.strip() for l in (page.extract_text() or "").splitlines() if l.strip()]

    customer = next((int(m.group(1)) for l in lines if (m := CUSTOMER.search(l))), None)
    rows, prev, cur, expect_bf = [], None, None, False

    def close(row):
        nonlocal prev
        if row and row["nums"]:
            balance = row["nums"][-1]
            amount = round(balance - prev, 2) if prev is not None else None
            prev = balance
            rows.append({
                "person_id": customer,
                "date": datetime.strptime(row["date"], "%d %b %Y").date().isoformat(),
                "description": row["desc"],
                "amount": amount,
                "category": categorise(row["desc"].upper()),
                "balance_after": balance,
            })

    for line in lines:
        if line.startswith("BALANCE BROUGHT FORWARD"):
            close(cur); cur = None
            tail = line.replace("BALANCE BROUGHT FORWARD", "").strip()
            if re.fullmatch(NUM, tail):
                prev = prev if prev is not None else num(tail)
            else:
                expect_bf = True
            continue
        if expect_bf and re.fullmatch(NUM, line):
            if prev is None:
                prev = num(line)
            expect_bf = False
            continue
        if re.fullmatch(DATE, line):
            close(cur)
            cur = {"date": line, "desc": None, "nums": []}
            continue
        if cur is None:
            continue
        if cur["desc"] is None:
            cur["desc"] = line
        elif re.fullmatch(NUM, line):
            cur["nums"].append(num(line))
        else:                       # header/footer text on a new page ends the row
            close(cur); cur = None
    close(cur)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--out", help="save parsed rows to this CSV")
    args = ap.parse_args()

    df = pd.concat([parse_pdf(p) for p in args.pdfs], ignore_index=True)
    print(df.head(10).to_string(index=False))
    print(f"... {len(df):,} transactions parsed from {len(args.pdfs)} PDF(s)")
    if args.out:
        df.to_csv(args.out, index=False)
        print("saved", args.out)


if __name__ == "__main__":
    main()
