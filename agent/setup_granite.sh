#!/usr/bin/env bash
# One-shot setup: builds llama.cpp and gets a big-endian Granite model running
# on the LinuxONE (s390x) VM. Run from your home dir:  bash setup_granite.sh
# Then start the server with:  bash setup_granite.sh serve
set -e

MODEL_DIR="$HOME/models"
MODEL_FILE="$MODEL_DIR/granite-3.3-2b-instruct-be.Q8_0.gguf"
LLAMA="$HOME/llama.cpp"

if [ "$1" = "serve" ]; then
  echo "Starting Granite on port 8080 (Ctrl+C to stop)..."
  exec "$LLAMA/build/bin/llama-server" -m "$MODEL_FILE" --port 8080 -t "$(nproc)" -c 4096
fi

echo "== 1/4 installing packages =="
sudo apt update
sudo apt install -y git cmake build-essential libopenblas-dev python3-pip python3-venv

echo "== 2/4 building llama.cpp (10-15 min) =="
[ -d "$LLAMA" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
cd "$LLAMA"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release -j"$(nproc)"

echo "== 3/4 downloading Granite and converting to big-endian =="
python3 -m venv "$HOME/llmenv"
source "$HOME/llmenv/bin/activate"
pip install -U pip
pip install -r requirements.txt huggingface_hub
mkdir -p "$MODEL_DIR"
huggingface-cli download ibm-granite/granite-3.3-2b-instruct --local-dir "$MODEL_DIR/granite-hf"
python3 convert_hf_to_gguf.py "$MODEL_DIR/granite-hf" --bigendian --outtype q8_0 --outfile "$MODEL_FILE"

echo "== 4/4 quick test =="
"$LLAMA/build/bin/llama-cli" -m "$MODEL_FILE" -p "Say hello in one sentence." -n 30 -no-cnv || true

echo
echo "Done. Start the server with:  bash setup_granite.sh serve"
echo "Then in another terminal:     python3 agent.py"
