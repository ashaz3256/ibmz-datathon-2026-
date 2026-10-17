import csv
import threading
import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk_bs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from llama_cpp import Llama

# Add project root directory to path to import model.Predictor
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from model.Predictor import get_crisis_predictions
    print("Backend ML Predictor imported successfully!")
except ImportError as e:
    get_crisis_predictions = None
    print(f"Warning: Could not import get_crisis_predictions: {e}")

# Initialize local Qwen model in background
print("Loading local Qwen model via llama_cpp...")
try:
    llm = Llama.from_pretrained(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="*q4_k_m.gguf",
        verbose=False,
    )
    print("Qwen model loaded successfully!")
except Exception as e:
    llm = None
    print(f"Warning: Could not load local LLM: {e}")

# Initialize ttkbootstrap themed window with dark executive look
window = ttk_bs.Window(themename="cyborg")
window.title("IBM Enterprise Financial Intelligence Suite (Local ML + AI)")
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
    text="IBM ENTERPRISE FINANCIAL INTELLIGENCE PLATFORM (LOCAL ML + AI)", 
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

def AIAutoCategorize(merchant_name):
    """Uses local Qwen model to map a raw merchant/item name into a standard financial category."""
    if not llm:
        return "Discretionary"
    
    prompt = (
        "Classify the following merchant or transaction name into exactly one standard financial category "
        "(Choose from: Groceries, Transport, Utilities, Entertainment, Salary, Rent, Discretionary). "
        "Return ONLY the category name and nothing else.\n\n"
        f"Merchant Name: {merchant_name}\nCategory:"
    )
    try:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        category = out["choices"][0]["message"]["content"].strip()
        for valid in ["Groceries", "Transport", "Utilities", "Entertainment", "Salary", "Rent", "Discretionary"]:
            if valid.lower() in category.lower():
                return valid
        return "Discretionary"
    except Exception:
        return "Discretionary"

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
            cat_values, labels=cat_labels, autopct='%1.1f%%', startangle=140, pctdistance=0.75,
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
        ax3.annotate(f'£{height:,.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', color=IBM_TEXT, fontsize=9, fontweight='bold')

    canvas = FigureCanvasTkAgg(fig, master=parent_container)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    toolbar = NavigationToolbar2Tk(canvas, parent_container)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

def ProcessFileThreaded():
    """Runs local file parsing with case-insensitive column mapping, AI categorization, and Sourov's XGBoost ML model."""
    path = ObtainLocalFile()
    if not path:
        return

    StatsBox.delete("1.0", tk.END)
    StatsBox.insert(tk.END, "Processing CSV, running AI Auto-Categorizer & Local ML Predictor... Please wait...\n")
    AnalyseButton.config(state="disabled", text="Processing...")

    def background_task():
        global current_financial_context
        try:
            df_raw = pd.read_csv(path)
            transactions_list = []
            cleaned_rows = []

            for _, row in df_raw.iterrows():
                # Normalize row keys to lowercase for robust matching ('Date' -> 'date', 'Cost' -> 'cost', etc.)
                row_lower = {str(k).strip().lower(): v for k, v in row.items()}
                
                date_val = str(row_lower.get('date', 'N/A')).strip()
                merchant_val = str(row_lower.get('merchant', row_lower.get('name', 'Unknown'))).strip()
                
                if 'category' in row_lower and pd.notna(row_lower['category']) and str(row_lower['category']).strip():
                    category_val = str(row_lower['category']).strip()
                else:
                    category_val = AIAutoCategorize(merchant_val)

                raw_cost = row_lower.get('amount', row_lower.get('cost', 0))
                cost_val = float(str(raw_cost).replace('£', '').replace(',', ''))

                raw_bal = row_lower.get('balance_after', row_lower.get('balance', 0))
                balance_val = float(str(raw_bal).replace('£', '').replace(',', ''))

                week_val = int(row_lower.get('week', 0)) if pd.notna(row_lower.get('week')) else 0
                person_id_val = int(row_lower.get('person_id', 1)) if pd.notna(row_lower.get('person_id')) else 1

                transactions_list.append({
                    "date": date_val,
                    "name": f"{merchant_val} ({category_val})",
                    "category": category_val,
                    "cost": cost_val,
                    "balance": balance_val
                })

                cleaned_rows.append({
                    'person_id': person_id_val,
                    'date': date_val,
                    'week': week_val,
                    'amount': cost_val,
                    'category': category_val,
                    'merchant': merchant_val,
                    'balance_after': balance_val
                })

            if not transactions_list:
                window.after(0, lambda: StatsBox.insert(tk.END, "No valid transaction rows found in the CSV file.\n"))
                AnalyseButton.config(state="normal", text="Load Transactions CSV & Launch Suite")
                return

            df_tx = pd.DataFrame(cleaned_rows)

            # Build weekly DataFrame for Sourov's predictor
            df_weekly = df_tx.groupby(['person_id', 'week']).agg(
                balance=('balance_after', 'last'),
                monthly_income=('amount', lambda x: x[x > 0].sum() * 4),
                monthly_rent=('amount', lambda x: abs(x[x < 0].sum()) * 4),
                overdraft_limit=('balance_after', lambda x: 500.0)
            ).reset_index()

            df_people = pd.DataFrame([{
                'person_id': 1,
                'monthly_income': float(df_weekly['monthly_income'].mean() if not df_weekly.empty else 3000.0),
                'monthly_rent': float(df_weekly['monthly_rent'].mean() if not df_weekly.empty else 1000.0),
                'overdraft_limit': 500.0
            }])

            # Run Sourov's ML Model directly
            max_crisis_prob = 0.0
            ml_predictions_summary = "Model not loaded."
            weekly_preds = []

            if get_crisis_predictions is not None:
                try:
                    preds = get_crisis_predictions(df_weekly, df_people, df_tx)
                    if isinstance(preds, pd.DataFrame) and 'crisis_probability' in preds.columns:
                        max_crisis_prob = float(preds['crisis_probability'].max())
                        weekly_preds = preds[['week', 'balance', 'crisis_probability', 'high_risk_flag']].to_dict(orient='records')
                    ml_predictions_summary = f"Maximum Bankruptcy/Crisis Probability: {max_crisis_prob:.2%}"
                except Exception as ml_err:
                    ml_predictions_summary = f"ML Execution Error: {ml_err}"

            def update_ui():
                for row in transaction_table.get_children():
                    transaction_table.delete(row)
                StatsBox.delete("1.0", tk.END)

                for t in transactions_list:
                    transaction_table.insert("", tk.END, values=(t["date"], t["name"], f"£{t['cost']:+,.2f}", f"£{t['balance']:,.2f}"))

                np_costs = np.array([t["cost"] for t in transactions_list], dtype=float)
                np_balances = np.array([t["balance"] for t in transactions_list], dtype=float)
                
                total_inflow = np.sum(np_costs[np_costs > 0])
                total_outflow = abs(np.sum(np_costs[np_costs < 0]))
                net_delta = total_inflow - total_outflow
                savings_rate = (net_delta / total_inflow) * 100 if total_inflow > 0 else 0
                coverage_ratio = total_inflow / total_outflow if total_outflow > 0 else float('inf')

                global current_financial_context
                current_financial_context = (
                    f"Financial Summary Data:\n"
                    f"- Local ML Crisis Probability Score: {max_crisis_prob:.4f} (Scale 0 to 1)\n"
                    f"- Total Inflows: £{total_inflow:,.2f}\n"
                    f"- Total Outflows: £{total_outflow:,.2f}\n"
                    f"- Net Balance Delta: £{net_delta:,.2f}\n"
                    f"- Savings Rate: {savings_rate:.2f}%\n"
                    f"- Coverage Ratio: {coverage_ratio:.2f}x\n"
                    f"- Minimum Balance Reached: £{np.min(np_balances):,.2f}"
                )

                StatsBox.insert(tk.END, "========================================================\n")
                StatsBox.insert(tk.END, "         IBM WATSONX LOCAL ML ANALYTICS REPORT          \n")
                StatsBox.insert(tk.END, "========================================================\n\n")
                StatsBox.insert(tk.END, f"• Local ML Crisis Probability Score: {max_crisis_prob:.4f} / 1.0\n")
                StatsBox.insert(tk.END, f"• ML Status:                         {ml_predictions_summary}\n")
                StatsBox.insert(tk.END, f"• Total Transactions Processed:      {len(transactions_list):,}\n")
                StatsBox.insert(tk.END, f"• Total Cumulative Inflows:        £{total_inflow:,.2f}\n")
                StatsBox.insert(tk.END, f"• Total Cumulative Outflows:       £{total_outflow:,.2f}\n")
                StatsBox.insert(tk.END, f"• Net Balance Delta:               £{net_delta:+,.2f}\n")
                StatsBox.insert(tk.END, f"• Portfolio Savings Rate:          {savings_rate:.2f}%\n\n")

                StatsBox.insert(tk.END, "--- Weekly Model Predictions ---\n")
                for wp in weekly_preds[:10]:
                    StatsBox.insert(tk.END, f"Week {wp['week']}: Balance £{wp['balance']:,.2f} | Crisis Prob: {wp['crisis_probability']:.2%} | High Risk: {wp['high_risk_flag']}\n")

                EmbedInteractiveCharts(transactions_list, chart_display_frame)
                AnalyseButton.config(state="normal", text="Load Transactions CSV & Launch Suite")

            window.after(0, update_ui)

        except Exception as ex:
            err_type = type(ex).__name__
            err_msg = str(ex)
            def show_error():
                StatsBox.insert(tk.END, f"Error: {err_type}: {err_msg}\n")
                AnalyseButton.config(state="normal", text="Load Transactions CSV & Launch Suite")
            window.after(0, show_error)

    threading.Thread(target=background_task, daemon=True).start()

def TriggerPrompt(prompt_text):
    UserInput.delete(0, tk.END)
    UserInput.insert(0, prompt_text)
    HandleAIQuery()

def HandleAIQuery(event=None):
    user_query = UserInput.get().strip()
    if not user_query or user_query.startswith("IBM WatsonX"):
        return

    notebook.select(3)
    AIBox.insert(tk.END, f"\n> User Query: {user_query}\n\n")
    UserInput.delete(0, tk.END)

    if not llm:
        AIBox.insert(tk.END, "AI Error: Local LLM model is not loaded.\n")
        return

    AIBox.insert(tk.END, "AI Thinking (WatsonX Local Model + Local ML) in background...\n")

    def ai_background_task():
        try:
            system_instructions = (
                "You are an expert IBM Chief Financial Risk Officer and AI Financial Advisor. "
                "Evaluate the local ML model crisis score (where 0 is safe and 1 is imminent bankruptcy) "
                "provided in the financial data summary. Provide a detailed, structured financial breakdown, "
                "explain the crisis risk, and give actionable recommendations for cost savings.\n\n"
            )

            full_prompt = system_instructions + current_financial_context + "\n\nUser Request: " + user_query + "\n\nDetailed Financial Analysis:"

            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=700,
                temperature=0.2
            )
            response_text = out["choices"][0]["message"]["content"]
            
            def update_ai_box():
                AIBox.insert(tk.END, f"AI Professional Report:\n{response_text}\n\n" + "="*60 + "\n")

            window.after(0, update_ai_box)
        except Exception as ex:
            err_type = type(ex).__name__
            err_msg = str(ex)
            window.after(0, lambda: AIBox.insert(tk.END, f"Error generating AI response: {err_type}: {err_msg}\n"))

    threading.Thread(target=ai_background_task, daemon=True).start()

AnalyseButton = ttk_bs.Button(
    control_frame, 
    text="Load Transactions CSV & Launch Suite", 
    command=ProcessFileThreaded, 
    bootstyle="primary"
)
AnalyseButton.pack(side=tk.LEFT, padx=(0, 10))

notebook = ttk_bs.Notebook(main_frame)
notebook.grid(row=2, column=0, sticky="nsew")

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
transaction_table.heading("name", text="Merchant & Category")
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

tab_stats = ttk_bs.Frame(notebook, padding=15)
notebook.add(tab_stats, text="  Advanced NumPy & ML Metrics  ")
StatsBox = tk.Text(tab_stats, width=120, height=25, bg="#1e1e1e", fg="#f4f4f4", insertbackground="white", font=("Consolas", 11), padx=10, pady=10)
StatsBox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tab_charts = ttk_bs.Frame(notebook, padding=15)
notebook.add(tab_charts, text="  Interactive Visualizations Dashboard  ")
tab_charts.columnconfigure(0, weight=1)
tab_charts.rowconfigure(0, weight=1)

chart_display_frame = ttk_bs.Frame(tab_charts)
chart_display_frame.grid(row=0, column=0, sticky="nsew")

tab_ai = ttk_bs.Frame(notebook, padding=15)
notebook.add(tab_ai, text="  AI WatsonX Financial Assistant  ")
tab_ai.columnconfigure(0, weight=1)
tab_ai.rowconfigure(0, weight=1)

AIBox = tk.Text(tab_ai, width=120, height=25, bg="#1e1e1e", fg="#f4f4f4", insertbackground="white", font=("Consolas", 11), padx=10, pady=10)
AIBox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
AIBox.insert(tk.END, "Load your CSV dataset, then click any expert prompt button below or type your custom query!\n\n")

preset_frame = ttk_bs.Frame(main_frame, padding=(0, 5, 0, 5))
preset_frame.grid(row=3, column=0, sticky="ew")

ttk_bs.Label(preset_frame, text="Expert AI Prompts:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

btn_p1 = ttk_bs.Button(preset_frame, text="🚨 Bankruptcy Risk Score (0-1)", bootstyle="danger-outline", 
                       command=lambda: TriggerPrompt("Give me a calculated bankruptcy risk score from 0 to 1 based on my net balance delta, coverage ratio, and minimum balances, and explain why."))
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

UserInput = ttk_bs.Entry(main_frame)
UserInput.grid(row=4, column=0, sticky="ew", pady=(5, 0))
UserInput.insert(0, "Ask AI about bankruptcy risk and savings analysis...")
UserInput.bind("<Return>", HandleAIQuery)

window.mainloop()