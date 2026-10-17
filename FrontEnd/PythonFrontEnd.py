import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext
import ttkbootstrap as ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize ttkbootstrap themed window
window = ttk.Window(themename="cosmo")
window.title("Financial Summary, Dictionaries & Charts")
window.geometry("1200x900")  # Fixed: using standard 'x' instead of multiplication sign

# Use standard scrolledtext inside the themed window
ChatBox = scrolledtext.ScrolledText(window, width=110, height=25)
ChatBox.grid(row=1, column=0, sticky="nsew", padx=15, pady=15)

def ObtainLocalFile():
    """Ask the user for the financial mock data CSV file."""
    path = filedialog.askopenfilename(
        title="Select Financial Mock Data CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return path or None

def GenerateCharts(data, header, col_map):
    """Generate Matplotlib charts for monthly balance and category spending."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Monthly Balance Trend
    months = [f"M{i+1}" for i in range(len(data))]
    balances = data[:, col_map['balance']]
    ax1.plot(months, balances, marker='o', color='#0275d8', linewidth=2)
    ax1.set_title("Monthly Balance Trend")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Balance (£)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Spending Categories Breakdown
    spend_categories = [name for name in header if name.startswith('spend_') and name != 'spend_salary']
    category_totals = []
    category_labels = []
    
    for cat in spend_categories:
        total = np.sum(data[:, col_map[cat]])
        if total > 0:
            category_totals.append(total)
            category_labels.append(cat.replace('spend_', '').replace('_', ' ').title())
            
    ax2.pie(category_totals, labels=category_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    ax2.set_title("Spending Breakdown by Category")
    
    plt.tight_layout()
    
    # Open a new Tkinter window to display the matplotlib charts
    chart_window = ttk.Toplevel(window)
    chart_window.title("Financial Visualizations")
    chart_window.geometry("1000x600")  # Fixed: using standard 'x'
    
    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def ObtainFileData():
    ChatBox.delete("1.0", tk.END)
    path = ObtainLocalFile()
    if not path:
        return

    try:
        # Read CSV file using standard csv module
        with open(path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                ChatBox.insert(tk.END, "CSV file is empty or missing headers.\n")
                return

            rows = [row for row in reader if row]

        if not rows:
            ChatBox.insert(tk.END, "No data rows found in the CSV file.\n")
            return

        # Convert rows into a 2D NumPy array of floats
        data = np.array([[float(val) for val in row] for row in rows], dtype=float)
        col_map = {name.strip(): idx for idx, name in enumerate(header)}

        # Build a list of transaction dictionaries
        transactions = []
        spend_categories = [name for name in header if name.startswith('spend_')]

        for i, row in enumerate(data):
            month_label = f"Month {i+1}"
            balance_val = row[col_map['balance']]
            
            # Record income/salary entry
            if 'spend_salary' in col_map and row[col_map['spend_salary']] > 0:
                salary_val = row[col_map['spend_salary']]
                transactions.append({
                    "date": month_label,
                    "name": "Salary / Income",
                    "cost": salary_val,
                    "balance": balance_val
                })

            # Record spending categories as individual transaction items
            for cat in spend_categories:
                cost_val = row[col_map[cat]]
                if cost_val > 0:
                    item_name = cat.replace('spend_', '').replace('_', ' ').title()
                    transactions.append({
                        "date": month_label,
                        "name": item_name,
                        "cost": -cost_val,  # Negative for money out
                        "balance": balance_val
                    })

        # Convert to NumPy arrays for vectorized calculations
        np_names = np.array([t["name"] for t in transactions], dtype=str)
        np_costs = np.array([t["cost"] for t in transactions], dtype=float)

        # Display results in ChatBox
        ChatBox.insert(tk.END, f"Successfully mapped {len(transactions)} transaction records into dictionaries & NumPy arrays.\n\n")

        ChatBox.insert(tk.END, "--- Sample of Stored Transaction Dictionaries (First 10) ---\n")
        for t in transactions[:10]:
            ChatBox.insert(tk.END, f"{t}\n")
        ChatBox.insert(tk.END, "...\n\n")

        ChatBox.insert(tk.END, "--- NumPy Vector Summary ---\n")
        ChatBox.insert(tk.END, f"Total Inflows: £{np.sum(np_costs[np_costs > 0]):,.2f}\n")
        ChatBox.insert(tk.END, f"Total Outflows: £{abs(np.sum(np_costs[np_costs < 0])):,.2f}\n")
        ChatBox.insert(tk.END, f"Average Transaction Cost: £{np.mean(np_costs):,.2f}\n\n")

        # Totals by item using NumPy masking
        ChatBox.insert(tk.END, "--- Totals by Category Name ---\n")
        unique_names = np.unique(np_names)
        for name in unique_names:
            mask = (np_names == name)
            cat_total = np.sum(np_costs[mask])
            ChatBox.insert(tk.END, f"{name:25} | Total: £{cat_total:+10,.2f}\n")

        # Trigger chart generation popup window
        GenerateCharts(data, header, col_map)

    except Exception as e:
        ChatBox.insert(tk.END, f"Error: {type(e).__name__}: {e}\nPath tried: {path}\n")

# Configure window grid layout
window.columnconfigure(0, weight=1)
window.rowconfigure(1, weight=1)

# Styled ttkbootstrap widgets
AnalyseButton = ttk.Button(window, text="Load Data & Generate Charts", command=ObtainFileData, bootstyle="success-outline")
AnalyseButton.grid(row=0, column=0, sticky="w", padx=15, pady=15)

UserInput = ttk.Entry(window, bootstyle="info")
UserInput.grid(row=2, column=0, sticky="ew", padx=15, pady=15)

window.mainloop()