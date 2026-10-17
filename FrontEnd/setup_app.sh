#!/usr/bin/env bash
# One-off setup for the LinuxONE app (FrontEnd/app_linuxone.py) on the VM.
# Run from the repo folder:  bash FrontEnd/setup_app.sh
# Needs Granite set up first:  bash agent/setup_granite.sh
set -e
cd "$(dirname "$0")/.."

echo "== 1/3 window, chart and ML libraries from Ubuntu (pre-built for s390x) =="
# pip can't build scikit-learn/scipy on s390x, so use Ubuntu's ready-made packages
sudo apt install -y python3-tk python3-sklearn python3-scipy python3-pandas python3-matplotlib

echo "== 2/3 Python packages (same environment as Granite) =="
# let the Granite environment see Ubuntu's packages
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' "$HOME/llmenv/pyvenv.cfg"
source "$HOME/llmenv/bin/activate"
pip install xgboost ttkbootstrap requests
# use Ubuntu's chart library (matches Ubuntu's numpy), not a mismatched pip copy
pip uninstall -y contourpy matplotlib 2>/dev/null || true
python3 -c "import numpy, sklearn, pandas, xgboost, matplotlib, ttkbootstrap; print('libraries OK')"

echo "== 3/3 synthetic customers =="
[ -f data/transactions.csv ] || python3 data_gen/gen_personas.py --out data

echo
echo "Done. Start Granite, then open the app (X11 forwarding on, VcXsrv running):"
echo "  nohup bash agent/setup_granite.sh serve > ~/granite.log 2>&1 &"
echo "  source ~/llmenv/bin/activate && python3 FrontEnd/app_linuxone.py"
