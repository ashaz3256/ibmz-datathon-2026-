# Johar Town Gang_IBM Z Datathon 2026 Project

**An AI early-warning system that spots people drifting into problem debt weeks before they miss rent, and helps them act before it turns into eviction or homelessness. Runs entirely on IBM LinuxONE.**

Theme: **Real-Time AI for Critical Decisions** · IBM Z Datathon 2026 · Manchester hub

## The problem

Help for problem debt usually arrives too late: after rent has been missed and eviction is close. Yet the warning signs show up in bank transactions weeks earlier: a drop in income, rising overdraft use, payday loans, buy-now-pay-later, bounced bills, gambling. Banks already see these patterns but rarely act on them early.

## Our solution

A bank runs this on its own LinuxONE. With the customer's opt-in consent:

1. **Risk model (predictive AI):** an XGBoost model scores every week of a customer's transactions and predicts the chance they miss rent in the next 6 weeks.
2. **Early warning:** the score becomes a plain-English level (*Steady*, *Stretched*, *Under pressure*, *At risk*). An alert fires when risk stays at *Under pressure* or above for 2 weeks.
3. **Reasons and what-ifs:** we show the top reasons behind the score (from the model's own feature contributions) and re-score "what if" scenarios, such as cutting non-essential spending.
4. **AI agent (generative AI):** IBM Granite, running on the same LinuxONE, turns this into a short, kind action plan: priority bills first (rent, council tax, energy), no suggestions to borrow more, and signposting to free help (StepChange, Citizens Advice, Shelter, GamCare). Customers can also ask follow-up questions in a chat.

**Responsible AI:** the agent warns, prioritises and signposts. It never gives regulated debt advice. Every Granite reply is checked against banned phrases (for example "limit increase" or "new loan"). If it fails, a safe template is used instead, and the app says so.

## Results

Measured on **1,000 synthetic customers the model never saw** (a different random seed from the training data):

| Metric | Result |
| --- | --- |
| ROC AUC | 0.95 |
| People who missed rent and were warned first | 98.6% |
| Median warning before rent was missed | 11 weeks |
| False alarms (alerted, but never missed rent) | 37% |

Reproduce with `python3 agent/evaluate_model.py` (results in `agent/model_performance.json`). The false-alarm rate is the main trade-off: the alert is tuned to catch nearly everyone early, so a support team would review alerts before contacting customers.

## Why IBM Z / LinuxONE

Banks already run on mainframes. Here the risk model, the Granite LLM and the app all run on LinuxONE (s390x), so customer transaction data never leaves the bank's own machine. Getting there meant building llama.cpp for s390x and converting Granite to big-endian, since LinuxONE stores numbers in the opposite byte order to a laptop.

## Architecture

```
Customer transactions
        │
        ▼
Risk model (XGBoost, model/Predictor.py) ──▶ risk_bridge.py: level, reasons, what-ifs, alert week
                                                      │
                                                      ▼
                          agent.py ──▶ IBM Granite 3.3 2B (llama.cpp server, port 8080)
                                                      │          + safety check / safe template
                                                      ▼
                                FrontEnd/app_linuxone.py (Risk & Action Plan, charts, chat)

            all running on one IBM LinuxONE Community Cloud VM (Ubuntu 24.04, s390x)
```

## Repository layout

| Path | What's in it |
| --- | --- |
| `data_gen/` | Synthetic UK transaction generator, plus a statement-PDF maker and pypdf reader |
| `model/` | XGBoost risk model: training (`model.py`, `model.ipynb`), saved model, `Predictor.py` |
| `agent/` | Granite agent, bridge from the model to the agent, model evaluation, Granite setup |
| `FrontEnd/app_linuxone.py` | **The demo app**: front end + risk model + Granite, runs on LinuxONE |
| `FrontEnd/PythonFrontEnd.py` | Original laptop version of the front end (uses Qwen via llama-cpp-python) |
| `docs/` | Screenshots |

## Running it on the LinuxONE VM

```bash
git clone https://github.com/ashaz3256/ibmz-datathon-2026-.git
cd ibmz-datathon-2026-

# 1. Build llama.cpp and a big-endian IBM Granite model (one-off, ~20-30 min)
bash agent/setup_granite.sh

# 2. App libraries + synthetic data (one-off)
bash FrontEnd/setup_app.sh

# 3. Start Granite in the background
nohup bash agent/setup_granite.sh serve > ~/granite.log 2>&1 &

# 4. Open the app (see "Showing the window" below)
source ~/llmenv/bin/activate
python3 FrontEnd/app_linuxone.py
```

In the app, pick a customer (e.g. `2 - job_loss`) and press **Load customer**. The risk score and chart appear at once, then Granite writes the plan live.

No window? Try it in the terminal instead: `python3 agent/risk_bridge.py 2 --agent` scores customer 2 and prints Granite's message.

### Showing the window (X11)

The VM has no screen, so the app's window is sent to your laptop over SSH:

1. On Windows, install [VcXsrv](https://github.com/marchaesen/vcxsrv/releases) and start it with big-endian clients allowed (PowerShell):
   `& "C:\Program Files\VcXsrv\vcxsrv.exe" :0 -multiwindow -clipboard -wgl +byteswappedclients`
2. In PuTTY: **Connection → SSH → X11 → Enable X11 forwarding**, then connect.

## Data

Real bank data is private, so we generate **synthetic UK current-account transactions**. Each simulated customer follows a persona: steady, job loss, gambling spiral, rent shock, or near-miss recovery. Money trouble emerges naturally from the rules (overdraft, payday loans, BNPL, bounced bills, missed rent), and every person-week is labelled with whether rent is missed in the next 6 weeks. The CSVs are not committed. Rebuild them with `python3 data_gen/gen_personas.py --out data` (fixed seed, so everyone gets the same customers).

## Open-source software used

- [IBM Granite 3.3 2B Instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) and [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [XGBoost](https://xgboost.ai/), [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/), [ttkbootstrap](https://ttkbootstrap.readthedocs.io/)
- [ReportLab](https://www.reportlab.com/opensource/), [pypdf](https://github.com/py-pdf/pypdf)

## Team

| Name | Role |
| --- | --- |
| Arij Shahzad | GenAI agent, LinuxONE deployment, integration |
| Sourov Aqib | Risk model (XGBoost) |
| Ismail Azfar | Front end |
| Tselmeg Ganzaa | LLM and front end |
