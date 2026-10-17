import csv
import threading
import numpy as np
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk_bs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from llama_cpp import Llama

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
window.title("IBM Enterprise Financial Intelligence Suite (AI-Powered)")
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
    text="IBM ENTERPRISE FINANCIAL INTELLIGENCE PLATFORM (WATSONX AI)", 
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

def ProcessFileThreaded():
    """Runs file parsing and AI categorization in a background thread to prevent UI freezing."""
    path = ObtainLocalFile()
    if not path:
        return

    StatsBox.delete("1.0", tk.END)
    StatsBox.insert(tk.END, "Processing CSV and running AI Auto-Categorizer in background... Please wait...\n")

    def background_task():
        global current_financial_context
        try:
            transactions = []
            with open(path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    window.after(0, lambda: StatsBox.insert(tk.END, "CSV file is empty or missing headers.\n"))
                    return

                # Clean header names (strip spaces and lowercase for robust lookup)
                clean_header = [h.strip().lower() for h in header]
                col_map = {name: idx for idx, name in enumerate(clean_header)}
                
                for row in reader:
                    if not row or len(row) < len(header):
                        continue
                    
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
                        "balance": balance_val
                    })

            if not transactions:
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
                StatsBox.insert(tk.END, "          IBM WATSONX FINANCIAL ANALYTICS REPORT        \n")
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

        except Exception as ex:
            err_type = type(ex).__name__
            err_msg = str(ex)
            window.after(0, lambda: StatsBox.insert(tk.END, f"Error: {err_type}: {err_msg}\nPath tried: {path}\n"))

    threading.Thread(target=background_task, daemon=True).start()

    
def TriggerPrompt(prompt_text):
    """Instantly injects and runs a preset prompt."""
    UserInput.delete(0, tk.END)
    UserInput.insert(0, prompt_text)
    HandleAIQuery()

def HandleAIQuery(event=None):
    """Query the local Qwen model asynchronously in a background thread."""
    user_query = UserInput.get().strip()
    if not user_query or user_query.startswith("IBM WatsonX"):
        return

    notebook.select(3)  # Switch to AI tab
    AIBox.insert(tk.END, f"\n> User Query: {user_query}\n\n")
    UserInput.delete(0, tk.END)

    if not llm:
        AIBox.insert(tk.END, "AI Error: Local LLM model is not loaded.\n")
        return

    AIBox.insert(tk.END, "AI Thinking (WatsonX Local Model in background)... Please wait...\n")

    def ai_background_task():
        try:
            system_instructions = (
                "You are an expert IBM Chief Financial Risk Officer and AI Financial Advisor. "
                "Evaluate bankruptcy risk on a scale from 0 (completely safe) to 1 (imminent bankruptcy) based on the data. "
                "Provide a detailed, structured, professional financial breakdown and actionable recommendations.\n\n"
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

# Action Button
AnalyseButton = ttk_bs.Button(
    control_frame, 
    text="Load Transactions CSV & Launch Suite", 
    command=ProcessFileThreaded, 
    bootstyle="primary"
)
AnalyseButton.pack(side=tk.LEFT, padx=(0, 10))

# Notebook layout separating views
notebook = ttk_bs.Notebook(main_frame)
notebook.grid(row=2, column=0, sticky="nsew")

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

# Tab 4: AI WatsonX Assistant Insights
tab_ai = ttk_bs.Frame(notebook, padding=15)
notebook.add(tab_ai, text="  AI WatsonX Financial Assistant  ")
tab_ai.columnconfigure(0, weight=1)
tab_ai.rowconfigure(0, weight=1)

AIBox = tk.Text(tab_ai, width=120, height=25, bg="#1e1e1e", fg="#f4f4f4", insertbackground="white", font=("Consolas", 11), padx=10, pady=10)
AIBox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
AIBox.insert(tk.END, "Load your CSV dataset, then click any expert prompt button below or type your custom query!\n\n")

# --- Preset Prompt Action Bar ---
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

# Bottom Command Line Entry
UserInput = ttk_bs.Entry(main_frame)
UserInput.grid(row=4, column=0, sticky="ew", pady=(5, 0))
UserInput.insert(0, "Ask AI about bankruptcy risk and savings analysis...")
UserInput.bind("<Return>", HandleAIQuery)

window.mainloop()