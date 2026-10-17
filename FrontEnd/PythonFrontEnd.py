import csv
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, filedialog

window = tk.Tk()
window.title("Financial Summary Analyser (NumPy & CSV)")
window.geometry("1100x850")

ChatBox = scrolledtext.ScrolledText(window, width=110, height=35)
ChatBox.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

def ObtainLocalFile():
    """Ask the user for the financial mock data CSV file."""
    path = filedialog.askopenfilename(
        title="Select Financial Mock Data CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return path or None

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
                ChatBox.insert(tk.END, "CSV file is empty or missing headers.")
                return

            rows = [row for row in reader if row]

        if not rows:
            ChatBox.insert(tk.END, "No data rows found in the CSV file.")
            return

        # Convert rows into a 2D NumPy array of floats
        data = np.array([[float(val) for val in row] for row in rows], dtype=float)

        # Map header names to column indices
        col_map = {name.strip(): idx for idx, name in enumerate(header)}

        # Extract key NumPy arrays
        balances = data[:, col_map['balance']]
        incomes = data[:, col_map['monthly_income']]
        total_spent = data[:, col_map['total_spent']]
        tx_counts = data[:, col_map['tx_count']]

        # Compute NumPy statistical summaries
        avg_balance = np.mean(balances)
        max_balance = np.max(balances)
        min_balance = np.min(balances)
        
        total_yearly_spent = np.sum(total_spent)
        avg_monthly_spent = np.mean(total_spent)
        total_yearly_income = np.sum(incomes)
        avg_tx_count = np.mean(tx_counts)

        # Display Summary
        ChatBox.insert(tk.END, f"=== 1-Year Financial Mock Data Analysis ({len(data)} Months) ===\n\n")
        
        ChatBox.insert(tk.END, "--- Macro Financial Overview ---\n")
        ChatBox.insert(tk.END, f"Total Yearly Income: £{total_yearly_income:,.2f}\n")
        ChatBox.insert(tk.END, f"Total Yearly Spent: £{total_yearly_spent:,.2f}\n")
        ChatBox.insert(tk.END, f"Average Monthly Balance: £{avg_balance:,.2f} (Min: £{min_balance:,.2f}, Max: £{max_balance:,.2f})\n")
        ChatBox.insert(tk.END, f"Average Monthly Spending: £{avg_monthly_spent:,.2f}\n")
        ChatBox.insert(tk.END, f"Average Monthly Transactions: {avg_tx_count:.1f}\n\n")

        # Category Breakdown (Columns starting with 'spend_')
        ChatBox.insert(tk.END, "--- Annual Spending by Category (NumPy Aggregation) ---\n")
        spend_categories = {name: col_map[name] for name in header if name.startswith('spend_')}
        
        for cat_name, col_idx in spend_categories.items():
            category_data = data[:, col_idx]
            annual_total = np.sum(category_data)
            monthly_avg = np.mean(category_data)
            display_name = cat_name.replace('spend_', '').replace('_', ' ').title()
            ChatBox.insert(tk.END, f"{display_name:27} | Annual: £{annual_total:8,.2f} | Monthly Avg: £{monthly_avg:7,.2f}\n")

    except Exception as e:
        ChatBox.insert(tk.END, f"Error: {type(e).__name__}: {e}\nPath tried: {path}")

# Configure window grid layout
window.columnconfigure(0, weight=1)
window.rowconfigure(1, weight=1)

AnalyseButton = tk.Button(window, text="Load Financial Mock Data CSV", command=ObtainFileData)
AnalyseButton.grid(row=0, column=0, sticky="w", padx=10, pady=10)

UserInput = tk.Entry(window)
UserInput.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

window.mainloop()