#!/usr/bin/env bash
# Builds llama.cpp and a big-endian Granite model on the LinuxONE (s390x) VM.
#
#   bash agent/setup_granite.sh            full setup (first time, ~20-30 min)
#   bash agent/setup_granite.sh quantize   only make the faster Q4 model (if you already have Q8)
#   bash agent/setup_granite.sh serve      start Granite on port 8080 (uses Q4 if it exists)
set -e

MODEL_DIR="$HOME/models"
MODEL_Q8="$MODEL_DIR/granite-3.3-2b-instruct-be.Q8_0.gguf"
MODEL_Q4="$MODEL_DIR/granite-3.3-2b-instruct-be.Q4_K_M.gguf"
LLAMA="$HOME/llama.cpp"

make_q4() {
  echo "== making the faster Q4 model from Q8 =="
  "$LLAMA/build/bin/llama-quantize" --allow-requantize "$MODEL_Q8" "$MODEL_Q4" Q4_K_M
  ls -lh "$MODEL_Q4"
}

if [ "$1" = "serve" ]; then
  MODEL="$MODEL_Q8"
  [ -f "$MODEL_Q4" ] && MODEL="$MODEL_Q4"
  echo "Starting Granite ($(basename "$MODEL")) on port 8080 (Ctrl+C to stop)..."
  exec "$LLAMA/build/bin/llama-server" -m "$MODEL" --port 8080 -t "$(nproc)" -c 4096
fi

if [ "$1" = "quantize" ]; then
  make_q4
  echo "Done. Restart the server with:  bash agent/setup_granite.sh serve"
  exit 0
fi

echo "== 1/5 installing packages =="
sudo apt update
sudo apt install -y git cmake build-essential pkg-config libopenblas-dev \
  libjpeg-dev zlib1g-dev python3-pip python3-venv

echo "== 2/5 building llama.cpp (10-15 min) =="
[ -d "$LLAMA" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
cd "$LLAMA"
# The VM runs on a z17, but Ubuntu's GCC 13 doesn't know z17 (arch15) yet.
# z17 runs z16 code fine, so target z16 instead.
sed -i 's/-march=arch15/-march=z16/g' ggml/src/ggml-cpu/CMakeLists.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release -j"$(nproc)"

echo "== 3/5 downloading Granite and converting to big-endian =="
python3 -m venv "$HOME/llmenv"
source "$HOME/llmenv/bin/activate"
pip install -U pip
pip install -r requirements.txt huggingface_hub requests
mkdir -p "$MODEL_DIR"
huggingface-cli download ibm-granite/granite-3.3-2b-instruct --local-dir "$MODEL_DIR/granite-hf"
python3 convert_hf_to_gguf.py "$MODEL_DIR/granite-hf" --bigendian --outtype q8_0 --outfile "$MODEL_Q8"

echo "== 4/5 =="
make_q4

echo "== 5/5 done =="
echo "Start the server with:  bash agent/setup_granite.sh serve"
echo "Then in another window: source ~/llmenv/bin/activate && python3 agent/agent.py"
