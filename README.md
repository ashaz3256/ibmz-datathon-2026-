# TeamName_IBM Z Datathon 2026 Project

**An AI early-warning system that spots people drifting into problem debt weeks before they miss rent, and helps them act before it turns into eviction or homelessness. Built on IBM LinuxONE.**

Theme: **Real-Time AI for Critical Decisions** · IBM Z Datathon 2026 · Manchester hub

## The problem

Help for problem debt usually arrives too late: after rent has been missed and eviction is close. Yet the warning signs show up in bank transactions weeks earlier: rising overdraft use, payday loans, buy-now-pay-later, bounced bills, a drop in income. Banks already see these patterns but rarely act on them early.

## Our solution

1. **Risk model (predictive AI):** scores each customer's recent transactions and predicts whether they'll miss rent in the next 6 weeks.
2. **Plain-English risk levels:** customers see *Steady*, *Stretched*, *Under pressure* or *At risk*, plus the trend, never a raw number. Bank support teams see the score to prioritise outreach.
3. **AI agent (generative AI):** explains *why* risk is rising, prioritises bills (rent, council tax and energy first), runs what-if scenarios ("claiming council tax reduction moves you back to Stretched"), and signposts free help such as StepChange and Citizens Advice.
4. **Statement upload:** customers can upload a bank-statement PDF; we parse it with pypdf and score it.

**Responsible AI:** the agent warns, prioritises and signposts. It does not give regulated debt advice. High-risk cases go to a human support team.

## Why IBM Z / LinuxONE

Banks already run on mainframes. Deploying this on a bank's own LinuxONE means customer transaction data never leaves the bank and stays encrypted, while the model scores transactions where the data lives.

## Architecture

```
Transactions / statement PDF ─▶ Risk model (Snap ML / XGBoost) ─▶ AI agent ─▶ Dashboard
                        └──────────── all running in one container on LinuxONE ────────────┘
```

## Repository layout

| Folder | What's in it |
| --- | --- |
| `data_gen/` | Synthetic UK transaction generator, statement PDF maker, pypdf statement reader |
| `model/` | Feature engineering and the risk model |
| `agent/` | LLM agent: explanations, plan, what-ifs |
| `dashboard/` | Streamlit front end |
| `docs/` | Architecture flowchart and LinuxONE screenshots |

## Quick start (on the LinuxONE VM)

```bash
pip3 install -r requirements.txt

# 1. Generate synthetic data (2,000 people x 52 weeks)
python3 data_gen/gen_personas.py --people 2000 --out data

# 2. Optional: make statement PDFs and read them back with pypdf
python3 data_gen/make_statements.py --data data --people 7 29 98 --out statements
python3 data_gen/read_statements.py statements/*.pdf --out parsed.csv
```

## Data

Real bank data is private, so we generate **synthetic UK current-account transactions**, in the same spirit as IBM's own Synthetic Data Sets. Each simulated customer follows a persona: steady, job loss, gambling spiral, rent shock, or near-miss recovery. Money trouble emerges naturally from the rules (overdraft, payday loans, BNPL, bounced bills, missed rent), and every person-week is labelled with whether rent is missed in the next 6 weeks.

All synthetic statements are issued by a fictional bank ("Northway Bank") and marked as synthetic data.

## Open-source libraries used

- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [ReportLab](https://www.reportlab.com/opensource/)
- [pypdf](https://github.com/py-pdf/pypdf)

## Team

| Name | Role |
| --- | --- |
| Arij Shahzad | Front end / integration |
| | Data |
| | ML |
| | GenAI |
| | Pitch |
