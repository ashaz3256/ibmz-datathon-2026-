
import re
import tkinter as tk
from tkinter import scrolledtext, filedialog
from pypdf import PdfReader
 
window = tk.Tk()
window.title("Bank Statement Analyser")
window.geometry("1000x800")
 
ChatBox = scrolledtext.ScrolledText(window, width=100, height=30)
ChatBox.grid(row=1, column=0, sticky="nsew", padx=10)
 
# Matches lines like:  15/10/26 Netflix £103- £2593
#                      15/10/26 Wages £104+ £2697
LINE_PATTERN = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s+"
    r"(?P<item>.+?)\s+"
    r"£(?P<cost>[\d,]+(?:\.\d+)?)\s*(?P<sign>[+-])?\s+"
    r"£(?P<balance>[\d,]+(?:\.\d+)?)\s*$"
)
 
 
def ObtainLocalFile():
    """Ask the user for a PDF and return its path (or None if cancelled)."""
    path = filedialog.askopenfilename(
        title="Select a PDF",
        filetypes=[("PDF files", "*.pdf")]
    )
    return path or None
 
 
def ExtractText(path):
    """Return the text from every page of the PDF."""
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)
 
 
def ParseTransactions(text):
    """Turn statement text into a list of dictionaries, one per transaction."""
    transactions = []
    for line in text.splitlines():
        match = LINE_PATTERN.match(line.strip())
        if not match:
            continue  # header row, blank line, etc.
 
        cost = float(match["cost"].replace(",", ""))
        # "-" means money out, "+" means money in. A bare £0 has no sign.
        if match["sign"] == "-":
            cost = -cost
 
        transactions.append({
            "date": match["date"],
            "item": match["item"],
            "cost": cost,
            "balance": float(match["balance"].replace(",", "")),
        })
    return transactions
 
 
def BuildColumns(transactions):
    """Same data as a dictionary of columns: {'date': [...], 'item': [...], ...}"""
    return {
        key: [t[key] for t in transactions]
        for key in ("date", "item", "cost", "balance")
    }
 
 
def BuildTotalsByItem(transactions):
    """Dictionary of total spent/received per item name."""
    totals = {}
    for t in transactions:
        totals[t["item"]] = totals.get(t["item"], 0) + t["cost"]
    return totals
 
 
def ObtainFileData():
    ChatBox.delete("1.0", tk.END)
    path = ObtainLocalFile()
    if not path:  # user clicked Cancel
        return
 
    try:
        text = ExtractText(path)
        if not text.strip():
            ChatBox.insert(tk.END, "No extractable text in this PDF.")
            return
 
        transactions = ParseTransactions(text)
        if not transactions:
            ChatBox.insert(
                tk.END,
                "No transactions matched. Raw text extracted:\n\n" + text
            )
            return
 
        columns = BuildColumns(transactions)
        totals = BuildTotalsByItem(transactions)
 
        ChatBox.insert(tk.END, "Transactions (list of dicts):\n")
        for t in transactions:
            ChatBox.insert(tk.END, f"{t}\n")
 
        ChatBox.insert(tk.END, "\nColumns (dict of lists):\n")
        for key, values in columns.items():
            ChatBox.insert(tk.END, f"{key}: {values}\n")
 
        ChatBox.insert(tk.END, "\nTotals by item (dict):\n")
        for item, total in totals.items():
            ChatBox.insert(tk.END, f"{item}: {total:+.2f}\n")
            
 
    except Exception as e:
        ChatBox.insert(tk.END, f"Error: {type(e).__name__}: {e}\nPath tried: {path}")
 
 
# Let the chat box stretch when the window is resized
window.columnconfigure(0, weight=1)
window.rowconfigure(1, weight=1)
 
AnalyseButton = tk.Button(window, text="Get Bank statement", command=ObtainFileData)
AnalyseButton.grid(row=0, column=0, sticky="w", padx=10, pady=10)
 
UserInput = tk.Entry(window)
UserInput.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

 
window.mainloop()
