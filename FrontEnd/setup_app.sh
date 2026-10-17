#!/usr/bin/env bash
# One-off setup for the LinuxONE front end (FrontEnd/app_linuxone.py) on the VM.
# Run from the repo folder:  bash FrontEnd/setup_app.sh
# Needs Granite set up first:  bash agent/setup_granite.sh
set -e
cd "$(dirname "$0")/.."

echo "== 1/3 window + chart libraries =="
sudo apt install -y python3-tk libfreetype-dev libpng-dev

echo "== 2/3 Python packages (into the same env as Granite) =="
source "$HOME/llmenv/bin/activate"
pip install pandas numpy xgboost scikit-learn matplotlib ttkbootstrap requests

echo "== 3/3 team data (2,000 synthetic customers) =="
[ -f data/transactions.csv ] || python3 data_gen/gen_personas.py --out data

echo
echo "Done. To open the app (with X11 forwarding on in PuTTY and VcXsrv running on your laptop):"
echo "  source ~/llmenv/bin/activate && python3 FrontEnd/app_linuxone.py"
