import csv
import threading
import numpy as np
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk_bs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import os
import platform
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# LinuxONE version of Ismail's front end (FrontEnd/PythonFrontEnd.py).
# Changes from his original:
#   - the LLM is IBM Granite, served on this LinuxONE VM by llama.cpp (agent/setup_granite.sh serve),
#     instead of Qwen loaded with llama_cpp (which can't load little-endian models on s390x)
#   - new "Risk & Action Plan" tab: Sourov's XGBoost model scores the customer,
#     and the agent turns that into a safe plain-English plan with Granite
#   - the chat box gets the model's real risk score and the same safety rules
# Run on the VM:  python3 FrontEnd/app_linuxone.py   (needs X11 forwarding to show the window)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import risk_bridge                                  # noqa: E402  Sourov's model + reasons/what-ifs
from agent import run_agent, build_facts, BANNED_PHRASES, stream_chat  # noqa: E402  Granite agent + safety rules

# Granite has 3 CPUs: only one question at a time, otherwise requests pile up and all time out
GRANITE_LOCK = threading.Lock()

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
LLM_BASE = LLM_URL.split("/v1/")[0]
granite_online = False


def check_granite():
    """True if the Granite server on this VM is up."""
    global granite_online
    try:
        granite_online = requests.get(f"{LLM_BASE}/v1/models", timeout=3).ok
    except Exception:
        granite_online = False
    return granite_online


def granite_chat(messages, max_tokens=300, temperature=0.2, timeout=300):
    """Sends a chat to Granite on the LinuxONE VM and returns the reply text."""
    r = requests.post(LLM_URL, timeout=timeout, json={
        "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


print("Checking Granite server on this LinuxONE VM...")
print("Granite online" if check_granite() else
      "Warning: Granite not reachable - start it with: bash agent/setup_granite.sh serve")

# Initialize ttkbootstrap themed window with dark executive look
window = ttk_bs.Window(themename="cyborg")
window.title("Financial Early-Warning Suite - Local ML + IBM Granite on LinuxONE")
window.geometry("1500x1020")

# IBM Design Language Colour Palette
IBM_BLUE = "#0f62fe"
IBM_DARK = "#161616"
IBM_PANEL = "#262626"
IBM_TEXT = "#f4f4f4"
IBM_SUCCESS = "#24a148"

current_financial_context = "No financial dataset loaded yet."

# Main container frame
main_frame = ttk_bs.Frame(window, padding=20)
main_frame.grid(row=0, column=0, sticky="nsew")

window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(2, weight=1)

# Header Title Block
title_label = ttk_bs.Label(
    main_frame, 
    text="FINANCIAL EARLY-WARNING SUITE  |  LOCAL ML + IBM GRANITE ON LINUXONE", 
    font=("Segoe UI", 18, "bold"),
    bootstyle="inverse"
)
title_label.grid(row=0, column=0, sticky="w", pady=(0, 15))

# Top Action Control Frame
control_frame = ttk_bs.Frame(main_frame)
control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))

def ObtainLocalFile():
    """Ask the user for the transaction CSV file."""
    path = filedialog.askopenfilename(
        title="Select Transactions CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return path or None

_category_cache = {}


def AIAutoCategorize(merchant_name):
    """Uses Granite on LinuxONE to map a raw merchant name into a standard financial category.
    Each merchant is only asked about once (cached), so repeat shops don't cost another LLM call."""
    key = merchant_name.strip().lower()
    if key in _category_cache:
        return _category_cache[key]
    if not granite_online:
        return "Discretionary"

    prompt = (
        "Classify the following merchant or transaction name into exactly one standard financial category "
        "(Choose from: Groceries, Transport, Utilities, Entertainment, Salary, Rent, Discretionary). "
        "Return ONLY the category name and nothing else.\n\n"
        f"Merchant Name: {merchant_name}\nCategory:"
    )
    try:
        with GRANITE_LOCK:  # wait our turn so we don't clash with a plan or chat answer
            category = granite_chat([{"role": "user", "content": prompt}], max_tokens=10, temperature=0.0, timeout=60)
        result = "Discretionary"
        for valid in ["Groceries", "Transport", "Utilities", "Entertainment", "Salary", "Rent", "Discretionary"]:
            if valid.lower() in category.lower():
                result = valid
                break
    except Exception:
        result = "Discretionary"
    _category_cache[key] = result
    return result

def EmbedInteractiveCharts(transactions, parent_container):
    """Embed an un-cramped, beautifully spaced 3-panel analytics dashboard."""
    for widget in parent_container.winfo_children():
        widget.destroy()

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 7.5), facecolor=IBM_PANEL)
    gs = fig.add_gridspec(1, 3, wspace=0.35, left=0.08, right=0.95, top=0.88, bottom=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    balances = [t["balance"] for t in transactions]
    
    # 1. Balance Trajectory Line Chart
    ax1.set_facecolor(IBM_DARK)
    ax1.plot(range(len(balances)), balances, color=IBM_BLUE, linewidth=2.5)
    ax1.set_title("Balance Trajectory", color=IBM_TEXT, fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Transaction Index", color=IBM_TEXT, labelpad=8)
    ax1.set_ylabel("Balance (£)", color=IBM_TEXT, labelpad=8)
    ax1.grid(True, linestyle='--', alpha=0.3, color='#525252')
    ax1.tick_params(colors=IBM_TEXT, labelsize=9)
    
    # 2. Spending Breakdown Donut Chart by Category
    categories = {}
    for t in transactions:
        if t["cost"] < 0:
            cat = t["category"].title()
            categories[cat] = categories.get(cat, 0) + abs(t["cost"])
            
    cat_labels = list(categories.keys())
    cat_values = list(categories.values())
    
    ax2.set_facecolor(IBM_DARK)
    if cat_values:
        wedges, texts, autotexts = ax2.pie(
            cat_values, 
            labels=cat_labels, 
            autopct='%1.1f%%', 
            startangle=140, 
            pctdistance=0.75,
            colors=['#0f62fe', '#33b1ff', '#009d9a', '#8a3ffc', '#fa4d56', '#42be65', '#f1c21b', '#ff7eb6']
        )
        for text in texts + autotexts:
            text.set_color(IBM_TEXT)
            text.set_fontsize(8)
    ax2.set_title("Expense Distribution", color=IBM_TEXT, fontsize=12, fontweight='bold', pad=12)

    # 3. Cumulative Cash Flow Bar Chart
    ax3.set_facecolor(IBM_DARK)
    costs = np.array([t["cost"] for t in transactions])
    inflows = np.sum(costs[costs > 0])
    outflows = abs(np.sum(costs[costs < 0]))
    
    bars = ax3.bar(["Total Inflows", "Total Outflows"], [inflows, outflows], color=[IBM_SUCCESS, '#fa4d56'], width=0.45)
    ax3.set_title("Inflows vs Outflows", color=IBM_TEXT, fontsize=12, fontweight='bold', pad=12)
    ax3.set_ylabel("Amount (£)", color=IBM_TEXT, labelpad=8)
    ax3.grid(True, linestyle='--', alpha=0.3, color='#525252')
    ax3.tick_params(colors=IBM_TEXT, labelsize=9)

    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'£{height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', color=IBM_TEXT, fontsize=9, fontweight='bold')

    canvas = FigureCanvasTkAgg(fig, master=parent_container)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    toolbar = NavigationToolbar2Tk(canvas, parent_container)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

def ProcessFileThreaded(path=None, person_id=None):
    """Runs file parsing and AI categorization in a background thread to prevent UI freezing.
    If the CSV has many customers (person_id column), only the chosen customer is loaded."""
    if path is None:
        path = ObtainLocalFile()
    if not path:
        return
    if person_id is None:
        person_id = _customer_from_box()

    StatsBox.delete("1.0", tk.END)
    StatsBox.insert(tk.END, "Processing CSV, running AI Auto-Categorizer & Local ML Predictor... Please wait...\n")
    AnalyseButton.config(state="disabled", text="Processing...")   # from Ismail's version

    def loading_done():
        window.after(0, lambda: AnalyseButton.config(state="normal", text="Load Transactions CSV & Launch Suite"))

    def background_task():
        global current_financial_context
        try:
            transactions = []
            raw_rows = []
            chosen_id = person_id
            with open(path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    window.after(0, lambda: StatsBox.insert(tk.END, "CSV file is empty or missing headers.\n"))
                    loading_done()
                    return

                # Clean header names (strip spaces and lowercase for robust lookup)
                clean_header = [h.strip().lower() for h in header]
                col_map = {name: idx for idx, name in enumerate(clean_header)}
                
                for row in reader:
                    if not row or len(row) < len(header):
                        continue
                    # Team data has every customer in one file: keep just one customer
                    if 'person_id' in col_map:
                        row_id = int(row[col_map['person_id']])
                        if chosen_id is None:
                            chosen_id = row_id
                        if row_id != chosen_id:
                            continue
                        raw_rows.append(row)
                    
                    # Safely retrieve values based on header names or fallback to safe indexes
                    date_val = row[col_map['date']].strip() if 'date' in col_map else (row[1] if len(row) > 1 else "N/A")
                    
                    # Extract merchant / description
                    merchant_val = "Unknown"
                    for m_key in ['merchant', 'name', 'description', 'item']:
                        if m_key in col_map:
                            merchant_val = row[col_map[m_key]].strip()
                            break
                    if merchant_val == "Unknown" and len(row) > 1:
                        merchant_val = row[1].strip()

                    # Extract or Auto-Categorize Category
                    if 'category' in col_map and row[col_map['category']].strip():
                        category_val = row[col_map['category']].strip()
                    else:
                        category_val = AIAutoCategorize(merchant_val)

                    # Extract Amount (Cost) safely, stripping currency symbols if present
                    cost_val = 0.0
                    for a_key in ['amount', 'cost', 'price']:
                        if a_key in col_map:
                            val_str = row[col_map[a_key]].strip().replace("£", "").replace(",", "")
                            cost_val = float(val_str)
                            break
                    
                    # Extract Balance After safely
                    balance_val = 0.0
                    for b_key in ['balance_after', 'balance']:
                        if b_key in col_map:
                            val_str = row[col_map[b_key]].strip().replace("£", "").replace(",", "")
                            balance_val = float(val_str)
                            break

                    transactions.append({
                        "date": date_val,
                        "name": f"{merchant_val} ({category_val})",
                        "category": category_val,
                        "cost": cost_val,
                        "balance": balance_val,
                        "week": int(float(row[col_map['week']])) if 'week' in col_map and row[col_map['week']].strip() else None,
                    })

            if not transactions:
                loading_done()
                window.after(0, lambda: StatsBox.insert(tk.END, "No valid transaction rows found in the CSV file.\n"))
                return

            def update_ui():
                for row in transaction_table.get_children():
                    transaction_table.delete(row)
                StatsBox.delete("1.0", tk.END)

                for t in transactions:
                    transaction_table.insert("", tk.END, values=(t["date"], t["name"], f"£{t['cost']:+,.2f}", f"£{t['balance']:,.2f}"))

                np_costs = np.array([t["cost"] for t in transactions], dtype=float)
                np_balances = np.array([t["balance"] for t in transactions], dtype=float)
                
                total_inflow = np.sum(np_costs[np_costs > 0])
                total_outflow = abs(np.sum(np_costs[np_costs < 0]))
                net_delta = total_inflow - total_outflow
                savings_rate = (net_delta / total_inflow) * 100 if total_inflow > 0 else 0
                coverage_ratio = total_inflow / total_outflow if total_outflow > 0 else float('inf')
                max_expense = abs(np.min(np_costs[np_costs < 0])) if np.any(np_costs < 0) else 0

                cat_spending = {}
                for t in transactions:
                    if t["cost"] < 0:
                        c = t["category"]
                        cat_spending[c] = cat_spending.get(c, 0) + abs(t["cost"])
                
                cat_summary_str = ", ".join([f"{k}: £{v:,.2f}" for k, v in cat_spending.items()])

                global current_financial_context
                current_financial_context = (
                    f"Financial Summary Data:\n"
                    f"- Total Inflows: £{total_inflow:,.2f}\n"
                    f"- Total Outflows: £{total_outflow:,.2f}\n"
                    f"- Net Balance Delta: £{net_delta:,.2f}\n"
                    f"- Savings Rate: {savings_rate:.2f}%\n"
                    f"- Coverage Ratio: {coverage_ratio:.2f}x\n"
                    f"- Minimum Balance Reached: £{np.min(np_balances):,.2f}\n"
                    f"- Maximum Single Expense: £{max_expense:,.2f}\n"
                    f"- Spending by Category Breakdown: {cat_summary_str}"
                )

                StatsBox.insert(tk.END, "========================================================\n")
                StatsBox.insert(tk.END, "        FINANCIAL ANALYTICS REPORT (LINUXONE)          \n")
                StatsBox.insert(tk.END, "========================================================\n\n")
                StatsBox.insert(tk.END, f"• Total Transactions Processed:    {len(transactions):,}\n")
                StatsBox.insert(tk.END, f"• Total Cumulative Inflows:      £{total_inflow:,.2f}\n")
                StatsBox.insert(tk.END, f"• Total Cumulative Outflows:     £{total_outflow:,.2f}\n")
                StatsBox.insert(tk.END, f"• Net Balance Delta:             £{net_delta:+,.2f}\n")
                StatsBox.insert(tk.END, f"• Portfolio Savings Rate:        {savings_rate:.2f}%\n")
                StatsBox.insert(tk.END, f"• Inflow / Outflow Coverage:     {coverage_ratio:.2f}x\n")
                StatsBox.insert(tk.END, f"• Transaction Volatility (StdDev): £{np.std(np_costs):,.2f}\n")
                StatsBox.insert(tk.END, f"• Largest Single Expense:        £{max_expense:,.2f}\n")
                StatsBox.insert(tk.END, f"• Peak Account Balance:          £{np.max(np_balances):,.2f}\n")
                StatsBox.insert(tk.END, f"• Minimum Account Balance:       £{np.min(np_balances):,.2f}\n")

                EmbedInteractiveCharts(transactions, chart_display_frame)

            window.after(0, update_ui)

            # Score this customer with Sourov's model (needs the team's data format)
            if chosen_id is not None and {'week', 'category', 'amount'} <= set(col_map):
                tx_df = pd.DataFrame(raw_rows, columns=clean_header)
                for col in ('person_id', 'week'):
                    tx_df[col] = tx_df[col].astype(int)
                tx_df['amount'] = tx_df['amount'].astype(float)
                window.after(0, lambda: RunRiskAssessment(chosen_id, tx_df))
            else:
                # Ismail's method: any bank CSV -> estimate income/rent -> Sourov's model
                try:
                    est = risk_bridge.estimate_any_csv([
                        {"date": t["date"], "amount": t["cost"], "category": t["category"],
                         "balance_after": t["balance"], "week": t["week"]} for t in transactions])
                    window.after(0, lambda: ShowEstimatedML(est))
                except Exception as ml_err:
                    window.after(0, lambda e=ml_err: ShowMLStats(error=f"{type(e).__name__}: {e}"))
            loading_done()

        except Exception as ex:
            err_type = type(ex).__name__
            err_msg = str(ex)
            window.after(0, lambda: StatsBox.insert(tk.END, f"Error: {err_type}: {err_msg}\nPath tried: {path}\n"))
            loading_done()

    threading.Thread(target=background_task, daemon=True).start()

    
def TriggerPrompt(prompt_text):
    """Instantly injects and runs a preset prompt."""
    UserInput.delete(0, tk.END)
    UserInput.insert(0, prompt_text)
    HandleAIQuery()

CHAT_RULES = """You are a financial early-warning assistant inside a UK bank's app, running on the bank's own
IBM LinuxONE. Answer the customer's question using ONLY the data below.
Rules:
- The risk score and level come from the bank's risk model. Never invent or recalculate a different score.
- Be kind, clear and practical. Plain English, short paragraphs or a short numbered list, under 200 words.
- Priority bills (rent, council tax, energy) always come before credit cards, loans and overdrafts.
- NEVER suggest borrowing more: no overdraft or credit limit increases, new loans, credit cards,
  payday loans or buy now pay later. Never suggest setting up new direct debits.
- You do not give regulated debt advice. For debt help, point to free services such as
  StepChange (0800 138 1111), Citizens Advice, Shelter (housing) or GamCare (gambling)."""


def HandleAIQuery(event=None):
    """Ask Granite on LinuxONE, in a background thread, with the model's real score and safety rules."""
    user_query = UserInput.get().strip()
    if not user_query or user_query.startswith("Ask AI about"):
        return

    notebook.select(tab_ai)
    AIBox.insert(tk.END, f"\n> You: {user_query}\n\n")
    UserInput.delete(0, tk.END)

    if not check_granite():
        AIBox.insert(tk.END, "Granite is offline. Start it on the VM with: bash agent/setup_granite.sh serve\n")
        return
    if not GRANITE_LOCK.acquire(blocking=False):
        AIBox.insert(tk.END, "Granite is busy with another answer. Please wait for it to finish, then ask again.\n")
        return

    AIBox.insert(tk.END, "Granite (running on this LinuxONE VM):\n")
    AIBox.see(tk.END)

    def show(piece):
        def add():
            AIBox.insert(tk.END, piece)
            AIBox.see(tk.END)
        window.after(0, add)

    def ai_background_task():
        try:
            context = current_financial_context
            if current_case:
                context += "\n\nRisk model result:\n" + build_facts(current_case)
            reply = stream_chat(
                [{"role": "system", "content": CHAT_RULES + "\n\nCUSTOMER DATA:\n" + context},
                 {"role": "user", "content": user_query}],
                max_tokens=250, temperature=0.2, on_token=show)
            low = reply.lower()
            if any(p in low for p in BANNED_PHRASES):
                show("\n\n[Safety check: that reply suggested something we never recommend, so ignore it.]\n"
                     "Taking on more credit can make things harder. For free, confidential help with debts, "
                     "contact StepChange on 0800 138 1111 or Citizens Advice.")
            show("\n\n" + "=" * 60 + "\n")
        except Exception as ex:
            err_type = type(ex).__name__
            err_msg = str(ex)
            window.after(0, lambda: AIBox.insert(tk.END, f"\nError generating AI response: {err_type}: {err_msg}\n"))
        finally:
            GRANITE_LOCK.release()

    threading.Thread(target=ai_background_task, daemon=True).start()


# ---------------------------------------------------------------------------
# Risk & Action Plan: Sourov's model -> reasons/what-ifs -> Granite agent
# ---------------------------------------------------------------------------
current_case = None      # latest risk assessment, also given to the chat box
current_tx = None        # this customer's transactions (to re-score other weeks)
LEVEL_STYLE = {"Steady": "success", "Stretched": "info", "Under pressure": "warning", "At risk": "danger"}


def _customer_from_box():
    text = CustomerBox.get().strip()
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return None


def LoadDemoCustomer():
    """Loads the chosen customer straight from the team's data/transactions.csv."""
    pid = _customer_from_box()
    path = risk_bridge.DATA_DIR / "transactions.csv"
    if pid is None:
        RiskHeadline.config(text="Type or pick a customer ID first")
        return
    if not path.exists():
        RiskHeadline.config(text="No data yet - run: python3 data_gen/gen_personas.py --out data")
        return
    ProcessFileThreaded(str(path), pid)


def RescoreWeek():
    pid = current_case["person_id"] if current_case else _customer_from_box()
    if pid is None:
        return
    try:
        week = int(WeekBox.get())
    except ValueError:
        week = None
    RunRiskAssessment(pid, current_tx, week)


_plan_request = 0  # increases each time a customer/week is loaded, so old plans are dropped


def RunRiskAssessment(person_id, tx_df=None, week=None):
    """Scores the customer (fast), shows it, then asks Granite for the plan (slow) in the background."""
    global current_tx, _plan_request
    current_tx = tx_df
    _plan_request += 1
    my_request = _plan_request
    notebook.select(tab_risk)
    RiskHeadline.config(text=f"Scoring customer {person_id} with the risk model...", bootstyle="secondary")
    RiskBox.delete("1.0", tk.END)

    def background():
        global current_case
        try:
            case, weekly_risk = risk_bridge.score_person(person_id, week=week, transactions=tx_df)
            if week is None and case["first_alert_week"] is not None:
                # Demo the moment the early warning first fires, before any missed rent
                case, weekly_risk = risk_bridge.score_person(person_id, week=case["first_alert_week"],
                                                             transactions=tx_df)
        except Exception as e:
            window.after(0, lambda e=e: RiskHeadline.config(text=f"Risk model error: {e}", bootstyle="danger"))
            window.after(0, lambda e=e: ShowMLStats(error=f"{type(e).__name__}: {e}"))
            return
        current_case = case
        window.after(0, lambda: ShowRisk(case, weekly_risk))

        def add(text):
            def do():
                if my_request == _plan_request:  # a newer customer was loaded: drop this one
                    RiskBox.insert(tk.END, text)
                    RiskBox.see(tk.END)
            window.after(0, do)

        if case["risk_level"].lower() in ("steady", "low"):
            message, source = run_agent(case)  # no Granite needed for steady customers
            add(f"\n--- Message to customer ---\n\n{message}\n")
            return

        if not GRANITE_LOCK.acquire(blocking=False):
            add("\nWaiting for Granite to finish the chat answer...\n")
            GRANITE_LOCK.acquire()
        try:
            if my_request != _plan_request:
                return  # user already moved on to another customer or week
            add("\n--- Message to customer (IBM Granite on LinuxONE, writing live) ---\n\n")
            streamed = []
            message, source = run_agent(case, on_token=lambda piece: (streamed.append(piece), add(piece)))
        finally:
            GRANITE_LOCK.release()

        if source == "granite":
            add("\n\n[Checked: passed the safety rules]\n")
        elif streamed:
            add("\n\n[Safety check: Granite's reply broke a rule, so the safe version is used instead]\n\n"
                f"{message}\n")
        else:
            add(f"(Granite unavailable, safe template used)\n\n{message}\n")

    threading.Thread(target=background, daemon=True).start()


ML_HEADER = "=== LOCAL ML RISK MODEL (Sourov's XGBoost) ==="


def _set_ml_context(line):
    """Ismail's version gave the ML score to the chat AI; here it goes to Granite."""
    global current_financial_context
    keep = [l for l in current_financial_context.split("\n") if not l.startswith("- Local ML Crisis Probability")]
    current_financial_context = "\n".join(keep) + "\n" + line


def ShowEstimatedML(est):
    """Ismail's any-CSV result: estimated inputs, so it is labelled as an estimate everywhere."""
    global current_case
    current_case = None  # a different file: don't let the chat talk about the previous customer
    lvl = est["risk_level"]
    RiskHeadline.config(text=f"ESTIMATED from this CSV only  |  week {est['week']}  |  {lvl.upper()}  "
                             f"({est['risk_score']:.2f})  -  load a team customer for the full plan",
                        bootstyle=f"inverse-{LEVEL_STYLE.get(lvl, 'secondary')}")
    start = StatsBox.search(ML_HEADER, "1.0")
    if start:
        StatsBox.delete(start, tk.END)
    e = est["estimated"]
    out = [f"\n{ML_HEADER}\n\n",
           f"• ML Status:                         Loaded - running locally on {platform.machine()} "
           f"({'IBM LinuxONE' if platform.machine() == 's390x' else 'not LinuxONE'}), ESTIMATED inputs\n",
           f"• Local ML Crisis Probability Score: {est['risk_score']:.4f} / 1.0  (latest week {est['week']}, {lvl})\n",
           f"• Highest weekly score:              {est['max_score']:.4f} / 1.0  (week {est['max_week']})\n",
           f"• Estimated from this CSV:           income £{e['monthly_income']:,.0f}/month, "
           f"rent £{e['monthly_rent']:,.0f}/month (from {e['rent_from']}), overdraft limit £{e['overdraft_limit']:,.0f}\n",
           "  (this CSV has no income/rent/overdraft details, so treat the score as a rough guide)\n"]
    if est["top_reasons"]:
        out.append("• Main reasons:                      " + "; ".join(est["top_reasons"]) + "\n")
    out.append("\n--- Weekly Model Predictions (last 12 weeks) ---\n")
    for _, wp in est["weekly"].tail(12).iterrows():
        bal = f"£{wp['balance']:,.2f}" if "balance" in wp else "n/a"
        flag = int(wp["high_risk_flag"]) if "high_risk_flag" in wp else int(wp.crisis_probability > 0.7)
        out.append(f"Week {int(wp.week):>2}: Balance {bal:>11} | Crisis Prob: {wp.crisis_probability:6.2%} | "
                   f"{risk_bridge.level_for(wp.crisis_probability):<14} | High Risk: {flag}\n")
    StatsBox.insert(tk.END, "".join(out))
    _set_ml_context(f"- Local ML Crisis Probability Score: {est['risk_score']:.4f} (Scale 0 to 1, "
                    f"ESTIMATED from this CSV, {lvl}). Main reasons: {'; '.join(est['top_reasons']) or 'none'}")


def ShowMLStats(case=None, weekly_risk=None, error=None):
    """Ismail's ML Status / Local ML Crisis Probability Score section, in the Advanced NumPy Metrics tab.
    Uses the same real-data scores as the Risk tab. Replaces the previous section instead of stacking."""
    start = StatsBox.search(ML_HEADER, "1.0")
    if start:
        StatsBox.delete(start, tk.END)
    out = [f"\n{ML_HEADER}\n\n"]
    if error or case is None:
        out.append(f"• ML Status:                         Error - {error}\n")
    else:
        sel = weekly_risk[weekly_risk.week == case["week"]]
        top = weekly_risk.loc[weekly_risk.crisis_probability.idxmax()]
        out += [
            f"• ML Status:                         Loaded - XGBoost model running locally on "
            f"{platform.machine()} ({'IBM LinuxONE' if platform.machine() == 's390x' else 'not LinuxONE'})\n",
            f"• Local ML Crisis Probability Score: {case['risk_score']:.4f} / 1.0  (week {case['week']}, "
            f"{case['risk_level']})\n",
            f"• Highest weekly score:              {top.crisis_probability:.4f} / 1.0  (week {int(top.week)})\n",
        ]
        if case["first_alert_week"] is not None:
            out.append(f"• Early warning first fired:         week {case['first_alert_week']}\n")
        if case["first_missed_rent_week"] is not None:
            out.append(f"• Rent first missed:                 week {case['first_missed_rent_week']}\n")
        out.append("\n--- Weekly Model Predictions (12 weeks up to the selected week) ---\n")
        window_weeks = weekly_risk[weekly_risk.week <= case["week"]].tail(12)
        for _, wp in window_weeks.iterrows():
            bal = f"£{wp['balance']:,.2f}" if "balance" in wp else "n/a"
            flag = int(wp["high_risk_flag"]) if "high_risk_flag" in wp else int(wp.crisis_probability > 0.7)
            here = "   <- selected week" if int(wp.week) == case["week"] else ""
            out.append(f"Week {int(wp.week):>2}: Balance {bal:>11} | Crisis Prob: {wp.crisis_probability:6.2%} | "
                       f"{risk_bridge.level_for(wp.crisis_probability):<14} | High Risk: {flag}{here}\n")
    StatsBox.insert(tk.END, "".join(out))
    if case is not None:
        _set_ml_context(f"- Local ML Crisis Probability Score: {case['risk_score']:.4f} (Scale 0 to 1, "
                        f"week {case['week']}, {case['risk_level']})")


def ShowRisk(case, weekly_risk):
    lvl = case["risk_level"]
    RiskHeadline.config(
        text=f"Customer {case['person_id']} ({case['name']})  |  week {case['week']}  |  "
             f"{lvl.upper()}  ({case['risk_score']:.2f})",
        bootstyle=f"inverse-{LEVEL_STYLE.get(lvl, 'secondary')}")
    WeekBox.config(from_=int(weekly_risk.week.min()), to=int(weekly_risk.week.max()))
    WeekBox.set(case["week"])

    RiskBox.insert(tk.END, "RISK MODEL (Sourov's XGBoost, running on LinuxONE)\n")
    RiskBox.insert(tk.END, f"  Risk of missing rent in the next 6 weeks: {case['risk_score']:.0%}  ->  {lvl}\n")
    fa, fm = case["first_alert_week"], case["first_missed_rent_week"]
    if fa is not None and fm is not None and fa < fm:
        RiskBox.insert(tk.END, f"  Early warning first fired in week {fa}; rent was first missed in week {fm} "
                               f"({fm - fa} weeks of warning)\n")
    elif fa is not None:
        RiskBox.insert(tk.END, f"  Early warning first fired in week {fa}\n")
    RiskBox.insert(tk.END, "\nWHY\n")
    for r in case["top_reasons"] or ["nothing stands out this week"]:
        RiskBox.insert(tk.END, f"  - {r}\n")
    if case["what_ifs"]:
        RiskBox.insert(tk.END, "\nWHAT-IF (re-scored by the model)\n")
        for w in case["what_ifs"]:
            RiskBox.insert(tk.END, f"  - If they {w['change']} (saves about £{w['saved']:,.0f} over 8 weeks): "
                                   f"{w['from']} {w['from_p']:.0%} -> {w['to']} {w['to_p']:.0%}\n")
    RiskBox.insert(tk.END, "\nAsking Granite for the action plan (words appear as it writes)...\n")
    DrawRiskChart(case, weekly_risk)
    ShowMLStats(case, weekly_risk)


def DrawRiskChart(case, weekly_risk):
    for widget in risk_chart_frame.winfo_children():
        widget.destroy()
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(7, 4.2), facecolor=IBM_PANEL)
    ax = fig.add_subplot(111)
    ax.set_facecolor(IBM_DARK)
    for lo, hi, col in [(0, .10, '#24a148'), (.10, .25, '#33b1ff'), (.25, .50, '#f1c21b'), (.50, 1, '#fa4d56')]:
        ax.axhspan(lo, hi, color=col, alpha=0.08)
    ax.plot(weekly_risk.week, weekly_risk.crisis_probability, color=IBM_BLUE, linewidth=2.2, label="Risk")
    if "missed_rent" in weekly_risk:
        m = weekly_risk[weekly_risk.missed_rent == 1]
        ax.scatter(m.week, m.crisis_probability, color='#fa4d56', marker='x', s=50, zorder=3, label="Rent missed")
    if case["first_alert_week"] is not None:
        ax.axvline(case["first_alert_week"], color='#f1c21b', linestyle='--', linewidth=1.2, label="First alert")
    ax.axvline(case["week"], color=IBM_TEXT, linewidth=1, alpha=0.6, label="Selected week")
    ax.set_ylim(0, 1)
    ax.set_title("Risk of missing rent, week by week", color=IBM_TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel("Week", color=IBM_TEXT)
    ax.set_ylabel("Probability", color=IBM_TEXT)
    ax.tick_params(colors=IBM_TEXT, labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.25, color='#525252')
    ax.legend(loc="upper left", fontsize=8, facecolor=IBM_PANEL)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=risk_chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# Action Button
AnalyseButton = ttk_bs.Button(
    control_frame, 
    text="Load Transactions CSV & Launch Suite", 
    command=ProcessFileThreaded, 
    bootstyle="primary"
)
AnalyseButton.pack(side=tk.LEFT, padx=(0, 10))

# Customer picker: demo one customer from the team's data in one click
ttk_bs.Label(control_frame, text="Customer:").pack(side=tk.LEFT, padx=(15, 5))
try:
    _demo_choices = [f"{pid} - {persona}" for pid, persona in risk_bridge.demo_people(n=2)]
except Exception:
    _demo_choices = []
CustomerBox = ttk_bs.Combobox(control_frame, values=_demo_choices, width=20)
if _demo_choices:
    CustomerBox.set(_demo_choices[0])
CustomerBox.pack(side=tk.LEFT, padx=(0, 5))
ttk_bs.Button(control_frame, text="Load customer", command=LoadDemoCustomer,
              bootstyle="success").pack(side=tk.LEFT, padx=(0, 15))
ttk_bs.Label(control_frame, text="Week:").pack(side=tk.LEFT, padx=(0, 5))
WeekBox = ttk_bs.Spinbox(control_frame, from_=0, to=51, width=5)
WeekBox.pack(side=tk.LEFT, padx=(0, 5))
ttk_bs.Button(control_frame, text="Re-score week", command=RescoreWeek,
              bootstyle="secondary-outline").pack(side=tk.LEFT)
GraniteStatus = ttk_bs.Label(control_frame,
                             text="Granite: online" if granite_online else "Granite: OFFLINE (start the server)",
                             bootstyle="success" if granite_online else "danger")
GraniteStatus.pack(side=tk.RIGHT)

# Notebook layout separating views
notebook = ttk_bs.Notebook(main_frame)
notebook.grid(row=2, column=0, sticky="nsew")

# Tab 0: Risk & Action Plan (model + Granite agent)
tab_risk = ttk_bs.Frame(notebook, padding=15)
notebook.add(tab_risk, text="  Risk & Action Plan  ")
tab_risk.columnconfigure(0, weight=3)
tab_risk.columnconfigure(1, weight=2)
tab_risk.rowconfigure(1, weight=1)
RiskHeadline = ttk_bs.Label(tab_risk, text="Pick a customer and press Load customer",
                            font=("Segoe UI", 14, "bold"), bootstyle="inverse-secondary", padding=10)
RiskHeadline.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
RiskBox = tk.Text(tab_risk, wrap="word", bg="#1e1e1e", fg="#f4f4f4", insertbackground="white",
                  font=("Consolas", 11), padx=10, pady=10)
RiskBox.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
risk_chart_frame = ttk_bs.Frame(tab_risk)
risk_chart_frame.grid(row=1, column=1, sticky="nsew")

# Tab 1: Structured Transactions Table
tab_table = ttk_bs.Frame(notebook, padding=15)
notebook.add(tab_table, text="  Structured Transactions Table  ")
tab_table.columnconfigure(0, weight=1)
tab_table.rowconfigure(0, weight=1)

table_frame = ttk_bs.Frame(tab_table)
table_frame.grid(row=0, column=0, sticky="nsew")
table_frame.columnconfigure(0, weight=1)
table_frame.rowconfigure(0, weight=1)

columns = ("date", "name", "cost", "balance")
transaction_table = ttk_bs.Treeview(table_frame, columns=columns, show="headings", bootstyle="info")
transaction_table.heading("date", text="Date")
transaction_table.heading("name", text="Merchant & AI Category")
transaction_table.heading("cost", text="Amount")
transaction_table.heading("balance", text="Balance After")

transaction_table.column("date", width=180, anchor="center")
transaction_table.column("name", width=450, anchor="w")
transaction_table.column("cost", width=220, anchor="e")
transaction_table.column("balance", width=220, anchor="e")

scrollbar = ttk_bs.Scrollbar(table_frame, orient=tk.VERTICAL, command=transaction_table.yview)
transaction_table.configure(yscrollcommand=scrollbar.set)
transaction_table.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

# Tab 2: Advanced NumPy Statistics Log
tab_stats = ttk_bs.Frame(notebook, padding=15)
notebook.add(tab_stats, text="  Advanced NumPy Metrics  ")
StatsBox = tk.Text(tab_stats, width=120, height=25, bg="#1e1e1e", fg="#f4f4f4", insertbackground="white", font=("Consolas", 11), padx=10, pady=10)
StatsBox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Tab 3: Interactive Visualizations Dashboard
tab_charts = ttk_bs.Frame(notebook, padding=15)
notebook.add(tab_charts, text="  Interactive Visualizations Dashboard  ")
tab_charts.columnconfigure(0, weight=1)
tab_charts.rowconfigure(0, weight=1)

chart_display_frame = ttk_bs.Frame(tab_charts)
chart_display_frame.grid(row=0, column=0, sticky="nsew")

# Tab 4: Granite AI Assistant (chat)
tab_ai = ttk_bs.Frame(notebook, padding=15)
notebook.add(tab_ai, text="  Granite AI Assistant  ")
tab_ai.columnconfigure(0, weight=1)
tab_ai.rowconfigure(0, weight=1)

AIBox = tk.Text(tab_ai, width=120, height=25, bg="#1e1e1e", fg="#f4f4f4", insertbackground="white", font=("Consolas", 11), padx=10, pady=10)
AIBox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
AIBox.insert(tk.END, "Load your CSV dataset, then click any expert prompt button below or type your custom query!\n\n")

# --- Preset Prompt Action Bar ---
preset_frame = ttk_bs.Frame(main_frame, padding=(0, 5, 0, 5))
preset_frame.grid(row=3, column=0, sticky="ew")

ttk_bs.Label(preset_frame, text="Expert AI Prompts:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

btn_p1 = ttk_bs.Button(preset_frame, text="🚨 Explain my risk score", bootstyle="danger-outline",
                       command=lambda: TriggerPrompt("Explain my risk score and the main reasons for it in plain English. What should I deal with first?"))
btn_p1.pack(side=tk.LEFT, padx=4)

btn_p2 = ttk_bs.Button(preset_frame, text="💡 Cost Reduction & Savings Plan", bootstyle="warning-outline", 
                       command=lambda: TriggerPrompt("Examine my category spending breakdown. List the top spending areas where I need to cut costs immediately to improve my savings rate."))
btn_p2.pack(side=tk.LEFT, padx=4)

btn_p3 = ttk_bs.Button(preset_frame, text="📈 Executive Health & Cashflow", bootstyle="info-outline", 
                       command=lambda: TriggerPrompt("Provide a thorough executive summary of my inflows versus outflows, cash flow stability, and overall financial health."))
btn_p3.pack(side=tk.LEFT, padx=4)

btn_p4 = ttk_bs.Button(preset_frame, text="⚠️ Liquidity & Volatility Audit", bootstyle="success-outline", 
                       command=lambda: TriggerPrompt("Analyze my minimum balance lows and expense volatility. What immediate measures should I take to prevent cash shortages?"))
btn_p4.pack(side=tk.LEFT, padx=4)

# Bottom Command Line Entry
UserInput = ttk_bs.Entry(main_frame)
UserInput.grid(row=4, column=0, sticky="ew", pady=(5, 0))
UserInput.insert(0, "Ask AI about your risk score, bills or savings...")
UserInput.bind("<Return>", HandleAIQuery)

window.mainloop()