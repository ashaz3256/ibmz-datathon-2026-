import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk_bs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Initialize ttkbootstrap themed window with dark executive look
window = ttk_bs.Window(themename="cyborg")
window.title("IBM Enterprise Financial Intelligence Suite")
window.geometry("1400x950")

# IBM Design Language Colour Palette
IBM_BLUE = "#0f62fe"
IBM_DARK = "#161616"
IBM_PANEL = "#262626"
IBM_TEXT = "#f4f4f4"
IBM_SUCCESS = "#24a148"

# Main container frame
main_frame = ttk_bs.Frame(window, padding=15)
main_frame.grid(row=0, column=0, sticky="nsew")

window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(2, weight=1)

# Header Title (Using ttk_bs.Label)
title_label = ttk_bs.Label(
    main_frame, 
    text="IBM ENTERPRISE FINANCIAL INTELLIGENCE PLATFORM", 
    font=("Segoe UI", 16, "bold"),
    bootstyle="inverse"
)
title_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

# Top Action Control Frame
control_frame = ttk_bs.Frame(main_frame)
control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

def ObtainLocalFile():
    """Ask the user for the transaction CSV file."""
    path = filedialog.askopenfilename(
        title="Select Transactions CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return path or None

def EmbedInteractiveCharts(transactions, parent_container):
    """Embed an advanced 3-panel analytics dashboard for transaction data."""
    for widget in parent_container.winfo_children():
        widget.destroy()

    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor(IBM_PANEL)
    
    balances = [t["balance"] for t in transactions]
    
    # 1. Balance Trajectory Line Chart
    ax1.set_facecolor(IBM_DARK)
    ax1.plot(range(len(balances)), balances, color=IBM_BLUE, linewidth=2)
    ax1.set_title("Balance Trajectory", color=IBM_TEXT, fontsize=11, fontweight='bold')
    ax1.set_xlabel("Transaction Sequence", color=IBM_TEXT)
    ax1.set_ylabel("Balance (£)", color=IBM_TEXT)
    ax1.grid(True, linestyle='--', alpha=0.3, color='#525252')
    ax1.tick_params(colors=IBM_TEXT)
    
    # 2. Spending Breakdown Donut/Pie Chart by Category
    categories = {}
    for t in transactions:
        if t["cost"] < 0:  # Outflows only
            cat = t["category"].title()
            categories[cat] = categories.get(cat, 0) + abs(t["cost"])
            
    cat_labels = list(categories.keys())
    cat_values = list(categories.values())
    
    ax2.set_facecolor(IBM_DARK)
    if cat_values:
        ax2.pie(
            cat_values, 
            labels=cat_labels, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=['#0f62fe', '#33b1ff', '#009d9a', '#8a3ffc', '#fa4d56', '#42be65', '#f1c21b', '#ff7eb6']
        )
    ax2.set_title("Expense Distribution by Category", color=IBM_TEXT, fontsize=11, fontweight='bold')

    # 3. Cumulative Cash Flow Bar Chart
    ax3.set_facecolor(IBM_DARK)
    costs = np.array([t["cost"] for t in transactions])
    inflows = np.sum(costs[costs > 0])
    outflows = abs(np.sum(costs[costs < 0]))
    
    ax3.bar(["Total Inflows", "Total Outflows"], [inflows, outflows], color=[IBM_SUCCESS, '#fa4d56'], width=0.5)
    ax3.set_title("Inflows vs Outflows", color=IBM_TEXT, fontsize=11, fontweight='bold')
    ax3.set_ylabel("Amount (£)", color=IBM_TEXT)
    ax3.grid(True, linestyle='--', alpha=0.3, color='#525252')
    ax3.tick_params(colors=IBM_TEXT)

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent_container)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, parent_container)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

def ObtainFileData():
    for row in transaction_table.get_children():
        transaction_table.delete(row)
    StatsBox.delete("1.0", tk.END)

    path = ObtainLocalFile()
    if not path:
        return

    try:
        transactions = []
        with open(path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                StatsBox.insert(tk.END, "CSV file is empty or missing headers.\n")
                return

            col_map = {name.strip(): idx for idx, name in enumerate(header)}
            
            for row in reader:
                if not row or len(row) < len(header):
                    continue
                
                date_val = row[col_map['date']].strip()
                merchant_val = row[col_map['merchant']].strip()
                category_val = row[col_map['category']].strip()
                cost_val = float(row[col_map['amount']])
                balance_val = float(row[col_map['balance_after']])

                # Store each transaction as a dictionary
                transactions.append({
                    "date": date_val,
                    "name": f"{merchant_val} ({category_val})",
                    "category": category_val,
                    "cost": cost_val,
                    "balance": balance_val
                })

        if not transactions:
            StatsBox.insert(tk.END, "No transaction rows found in the CSV file.\n")
            return

        # Insert rows into Treeview Table
        for t in transactions:
            transaction_table.insert("", tk.END, values=(t["date"], t["name"], f"£{t['cost']:+,.2f}", f"£{t['balance']:,.2f}"))

        # NumPy Vector Calculations
        np_costs = np.array([t["cost"] for t in transactions], dtype=float)
        np_balances = np.array([t["balance"] for t in transactions], dtype=float)
        
        total_inflow = np.sum(np_costs[np_costs > 0])
        total_outflow = abs(np.sum(np_costs[np_costs < 0]))
        net_delta = total_inflow - total_outflow

        # Populate Advanced Statistics Box
        StatsBox.insert(tk.END, "=== ADVANCED NUMPY TRANSACTION ANALYTICS ===\n\n")
        StatsBox.insert(tk.END, f"• Total Transactions Processed: {len(transactions)}\n")
        StatsBox.insert(tk.END, f"• Total Cumulative Inflows:  £{total_inflow:,.2f}\n")
        StatsBox.insert(tk.END, f"• Total Cumulative Outflows: £{total_outflow:,.2f}\n")
        StatsBox.insert(tk.END, f"• Net Balance Delta:         £{net_delta:+,.2f}\n")
        StatsBox.insert(tk.END, f"• Transaction Volatility (Std Dev): £{np.std(np_costs):,.2f}\n")
        StatsBox.insert(tk.END, f"• Peak Account Balance:      £{np.max(np_balances):,.2f}\n")
        StatsBox.insert(tk.END, f"• Minimum Account Balance:   £{np.min(np_balances):,.2f}\n\n")

        # Render charts
        EmbedInteractiveCharts(transactions, chart_display_frame)

    except Exception as e:
        StatsBox.insert(tk.END, f"Error: {type(e).__name__}: {e}\nPath tried: {path}\n")

# Action Button
AnalyseButton = ttk_bs.Button(
    control_frame, 
    text="Load Transactions CSV & Initialize Dashboard", 
    command=ObtainFileData, 
    bootstyle="primary"
)
AnalyseButton.pack(side=tk.LEFT, padx=(0, 10))

# Notebook layout separating views
notebook = ttk_bs.Notebook(main_frame)
notebook.grid(row=2, column=0, sticky="nsew")

# Tab 1: Interactive Data Table (Treeview)
tab_table = ttk_bs.Frame(notebook, padding=10)
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

transaction_table.column("date", width=150, anchor="center")
transaction_table.column("name", width=350, anchor="w")
transaction_table.column("cost", width=200, anchor="e")
transaction_table.column("balance", width=200, anchor="e")

scrollbar = ttk_bs.Scrollbar(table_frame, orient=tk.VERTICAL, command=transaction_table.yview)
transaction_table.configure(yscrollcommand=scrollbar.set)
transaction_table.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

# Tab 2: Advanced NumPy Statistics Log
tab_stats = ttk_bs.Frame(notebook, padding=10)
notebook.add(tab_stats, text="  Advanced NumPy Metrics  ")
StatsBox = tk.Text(tab_stats, width=120, height=25, bg="#1e1e1e", fg="#f4f4f4", insertbackground="white", font=("Consolas", 11))
StatsBox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Tab 3: Interactive Charts View
tab_charts = ttk_bs.Frame(notebook, padding=10)
notebook.add(tab_charts, text="  Interactive Visualizations Dashboard  ")
tab_charts.columnconfigure(0, weight=1)
tab_charts.rowconfigure(0, weight=1)

chart_display_frame = ttk_bs.Frame(tab_charts)
chart_display_frame.grid(row=0, column=0, sticky="nsew")

# Bottom Command Line Entry
UserInput = ttk_bs.Entry(main_frame)
UserInput.grid(row=3, column=0, sticky="ew", pady=(10, 0))
UserInput.insert(0, "IBM WatsonX Command Line Interface Ready...")

window.mainloop()