# Agent

Turns the risk model's output into a short, plain-English action plan using **IBM Granite**,
running on the LinuxONE VM. It warns, prioritises and signposts. It never gives regulated debt advice.

| File | What it does |
| --- | --- |
| `setup_granite.sh` | Builds llama.cpp for s390x, downloads Granite, converts it to big-endian, makes a faster Q4 version. `serve` starts it on port 8080 |
| `agent.py` | Sends the model's facts plus strict rules to Granite, streams the reply, checks it for banned phrases, and falls back to a safe template |
| `risk_bridge.py` | Runs the team's model (`model/Predictor.py`) and adds level, top reasons, what-ifs, bills and first alert week |
| `evaluate_model.py` | Measures the model on 1,000 new customers it never saw and saves `model_performance.json` |

```bash
python3 agent/agent.py               # demo customer, no model needed
python3 agent/risk_bridge.py 2 --agent   # score customer 2 with the model, then ask Granite
python3 agent/evaluate_model.py      # re-measure model performance
```
