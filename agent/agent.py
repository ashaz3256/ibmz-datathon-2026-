"""
Financial distress agent.

Takes the risk model's output for one person and turns it into a short,
plain-English explanation plus a prioritised action plan.

- Tries Granite running locally on the LinuxONE VM (llama-server, OpenAI-style API).
- If Granite isn't reachable or errors, falls back to a template so the demo never breaks.

This is early warning + signposting, NOT regulated debt advice.

Usage:
    python3 agent.py                 # runs the built-in demo case
    python3 agent.py case.json       # runs on a JSON file shaped like DEMO_CASE
"""

import json
import os
import sys

import requests

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))

# Debts in the order people should deal with them (priority debts first).
PRIORITY_ORDER = [
    "rent", "mortgage", "council tax", "energy", "water",
    "tv licence", "child maintenance", "court fines",
    "credit card", "overdraft", "personal loan", "buy now pay later", "payday loan",
]
PRIORITY_DEBTS = set(PRIORITY_ORDER[:8])

SUPPORT = {
    "StepChange": "free debt help, stepchange.org, 0800 138 1111",
    "Citizens Advice": "free advice on debt, benefits and housing, citizensadvice.org.uk",
    "National Debtline": "free debt advice, nationaldebtline.org, 0808 808 4000",
    "Shelter": "housing and eviction help, shelter.org.uk, 0808 800 4444",
    "GamCare": "free gambling support, gamcare.org.uk, 0808 8020 133",
    "Turn2us": "benefits calculator and grants, turn2us.org.uk",
}

SYSTEM_PROMPT = """You are an early-warning assistant inside a UK bank's app.
A risk model has flagged that a customer may be heading into financial difficulty.
Write to the customer directly, in a calm, kind, non-judgemental tone.

Rules:
- Use ONLY the facts given. Never invent numbers, debts or reasons. Copy numbers exactly.
- Describe how serious things are using the risk level wording given. Never say "slightly".
- Explain in 2-3 sentences why their risk is rising.
- Then give a short numbered plan (max 4 steps). Deal with the priority debts first,
  then the others, in the order given.
- NEVER suggest borrowing more: no overdraft or credit limit increases, new loans,
  credit cards, payday loans or buy now pay later.
- NEVER suggest setting up direct debits or new payments.
- If a what-if scenario is given, mention it as encouragement.
- Signpost the free services given. Do not give regulated debt advice;
  say they can get free, confidential help.
- Start with "Hi <first name>,". Under 120 words. Plain English, no markdown headings."""

# Phrases a reply must never contain. If Granite writes one, we use the template instead.
BANNED_PHRASES = [
    "limit increase", "increase your overdraft", "increase your limit", "extend your overdraft",
    "new loan", "take out a loan", "apply for a loan", "payday loan to", "new credit card",
    "balance transfer", "direct debit for", "set up a direct debit", "slightly", "(priority)",
]


# ---------- helpers ----------

def sort_debts(debts):
    def rank(d):
        name = d.get("type", "").lower()
        return PRIORITY_ORDER.index(name) if name in PRIORITY_ORDER else len(PRIORITY_ORDER)
    return sorted(debts, key=rank)


def pick_support(case):
    reasons = " ".join(case.get("top_reasons", [])).lower()
    picks = []
    if "gambl" in reasons:
        picks.append("GamCare")
    if any(d.get("type", "").lower() in ("rent", "mortgage") for d in case.get("debts", [])) or "rent" in reasons:
        picks.append("Shelter")
    if "income" in reasons or "job" in reasons or "salary" in reasons:
        picks.append("Turn2us")
    picks.append("StepChange")
    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:3]


def build_facts(case):
    debts = sort_debts(case.get("debts", []))
    lines = [
        f"Customer first name: {case.get('name', 'there')}",
        f"Risk level: {case['risk_level']} (score {case['risk_score']:.2f} out of 1)",
        "Main reasons from the model: " + "; ".join(case.get("top_reasons", [])),
    ]
    priority = [d for d in debts if d["type"].lower() in PRIORITY_DEBTS]
    other = [d for d in debts if d["type"].lower() not in PRIORITY_DEBTS]
    if priority:
        lines.append("Priority debts (deal with these first): " + "; ".join(
            f"{d['type']} £{d.get('amount', 0):,.0f}" for d in priority))
    if other:
        lines.append("Other debts (after the priority ones): " + "; ".join(
            f"{d['type']} £{d.get('amount', 0):,.0f}" for d in other))
    for w in case.get("what_ifs", []):
        lines.append(f"What-if: if they {w['change']}, risk goes from {w['from']} to {w['to']}")
    support = pick_support(case)
    lines.append("Free services to signpost: " + "; ".join(f"{s} ({SUPPORT[s]})" for s in support))
    return "\n".join(lines)


# ---------- the two ways of writing the message ----------

def stream_chat(messages, max_tokens=220, temperature=0.3, on_token=None):
    """Sends a chat to Granite and reads the reply as it is written (streaming).
    on_token(text) is called for each new piece, so a UI can show words as they appear.
    Returns the full reply."""
    pieces = []
    with requests.post(LLM_URL, stream=True, timeout=(10, LLM_TIMEOUT), json={
            "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": True}) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            piece = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
            if piece:
                pieces.append(piece)
                if on_token:
                    on_token(piece)
    return "".join(pieces).strip()


def check_safe(text):
    """Raises if the reply is too short or contains a banned phrase."""
    if len(text) < 40:
        raise ValueError("model reply too short")
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            raise ValueError(f"unsafe reply (contained '{phrase}')")
    return text


def ask_granite(case, on_token=None):
    text = stream_chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_facts(case)},
    ], max_tokens=220, temperature=0.3, on_token=on_token)
    return check_safe(text)


def template_message(case):
    name = case.get("name", "there")
    reasons = case.get("top_reasons", [])
    debts = sort_debts(case.get("debts", []))
    support = pick_support(case)

    why = ", ".join(reasons[:3]) if reasons else "changes in your spending"
    parts = [f"Hi {name}, we've noticed a few changes in your account that suggest things might be getting tight: {why}."]

    steps = []
    priority = [d for d in debts if d["type"].lower() in PRIORITY_DEBTS]
    other = [d for d in debts if d["type"].lower() not in PRIORITY_DEBTS]
    if priority:
        names = [d["type"] for d in priority]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        steps.append(f"Put {listed} first, as missing these has the most serious consequences.")
    if other:
        steps.append("Pay at least the minimum on your " + ", ".join(d["type"] for d in other) + " and ask lenders about breathing space.")
    for w in case.get("what_ifs", [])[:1]:
        steps.append(f"If you {w['change']}, our model shows your risk dropping from {w['from']} to {w['to']}.")
    steps.append("Talk to " + " or ".join(support[:2]) + " for free, confidential help.")

    parts.append("Here's a simple plan:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
    parts.append("You're not on your own, and acting early makes a real difference.")
    return "\n\n".join(parts)


def run_agent(case, on_token=None):
    """Returns (message, source) where source is 'granite' or 'template'.
    on_token: optional callback to show Granite's words as they are written."""
    if case.get("risk_level", "").lower() in ("steady", "low"):
        return f"Hi {case.get('name', 'there')}, everything looks steady. No action needed.", "rule"
    try:
        return ask_granite(case, on_token), "granite"
    except Exception as e:
        print(f"[agent] Granite not used ({type(e).__name__}: {e}), using template", file=sys.stderr)
        return template_message(case), "template"


# ---------- demo ----------

DEMO_CASE = {
    "name": "Jordan",
    "risk_level": "At Risk",
    "risk_score": 0.82,
    "top_reasons": [
        "gambling spend up 240% over the last 6 weeks",
        "overdraft used on 18 of the last 30 days",
        "two direct debits bounced this month",
    ],
    "debts": [
        {"type": "credit card", "amount": 1850},
        {"type": "rent", "amount": 725},
        {"type": "council tax", "amount": 160},
        {"type": "payday loan", "amount": 300},
    ],
    "what_ifs": [
        {"change": "stop gambling spend", "from": "At Risk", "to": "Steady"},
    ],
}

if __name__ == "__main__":
    case = DEMO_CASE
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            case = json.load(f)
    msg, source = run_agent(case)
    print(f"--- message (via {source}) ---\n{msg}")
