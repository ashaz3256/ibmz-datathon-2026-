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


ROW = re.compile(rf"^{DATE}\s+(.+?)\s+((?:{NUM}\s*)+)$")
BF = re.compile(rf"BALANCE BROUGHT FORWARD\s+({NUM})")


def parse_pdf(path):
    """Each transaction is one line: date, description, amount column(s), balance.
    The amount is worked out from the running balance, so blank columns don't matter."""
    reader = PdfReader(path)
    rows, prev, customer = [], None, None
    for page in reader.pages:
        for text in (page.extract_text() or "").splitlines():
            text = text.strip()
            if customer is None and (m := CUSTOMER.search(text)):
                customer = int(m.group(1))
            if (m := BF.search(text)):
                if prev is None:
                    prev = num(m.group(1))
                continue
            if not (m := ROW.match(text)):
                continue
            d, desc, nums = m.groups()
            balance = num(nums.split()[-1])
            amount = round(balance - prev, 2) if prev is not None else None
            prev = balance
            rows.append({
                "person_id": customer,
                "date": datetime.strptime(d, "%d %b %Y").date().isoformat(),
                "description": desc.strip(),
                "amount": amount,
                "category": categorise(desc.upper()),
                "balance_after": balance,
            })
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
