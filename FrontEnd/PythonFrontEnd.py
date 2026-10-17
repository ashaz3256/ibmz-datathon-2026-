import csv
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, filedialog

window = tk.Tk()
window.title("Financial Summary & Dictionary Analyser")
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
        col_map = {name.strip(): idx for idx, name in enumerate(header)}

        # Build a list of transaction dictionaries (e.g., aggregating monthly totals or category rows)
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

            # Record spending categories as individual transaction items for the dictionary
            for cat in spend_categories:
                cost_val = row[col_map[cat]]
                if cost_val > 0:
                    item_name = cat.replace('spend_', '').replace('_', ' ').title()
                    transactions.append({
                        "date": month_label,
                        "name": item_name,
                        "cost": -cost_val, # Negative for money out
                        "balance": balance_val
                    })

        # Convert to NumPy arrays for vectorized calculations
        np_dates = np.array([t["date"] for t in transactions], dtype=str)
        np_names = np.array([t["name"] for t in transactions], dtype=str)
        np_costs = np.array([t["cost"] for t in transactions], dtype=float)
        np_balances = np.array([t["balance"] for t in transactions], dtype=float)

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

    except Exception as e:
        ChatBox.insert(tk.END, f"Error: {type(e).__name__}: {e}\nPath tried: {path}")

# Configure window grid layout
window.columnconfigure(0, weight=1)
window.rowconfigure(1, weight=1)

AnalyseButton = tk.Button(window, text="Load & Build Transaction Dictionaries", command=ObtainFileData)
AnalyseButton.grid(row=0, column=0, sticky="w", padx=10, pady=10)

UserInput = tk.Entry(window)
UserInput.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

window.mainloop()