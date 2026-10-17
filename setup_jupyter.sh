#!/usr/bin/env bash
# One-shot setup for the LinuxONE VM:
#   Docker -> IBM's s390x Jupyter Lab image -> our repo in ~/shared -> open port 38888
# Based on https://github.com/linuxone-community-cloud/jupyter-lab-ml
#
# Usage (on the VM):
#   curl -fsSL https://raw.githubusercontent.com/ashaz3256/ibmz-datathon-2026-/main/setup_jupyter.sh | bash -s -- MyToken
set -euo pipefail

TOKEN="${1:-datathon2026}"
IMAGE="registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest"
REPO="https://github.com/ashaz3256/ibmz-datathon-2026-"
SHARED="$HOME/shared"
PORT=38888

echo "== Machine: $(uname -m) =="

echo "== 1/5 Installing Docker (skipped if already there) =="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh
fi
sudo systemctl start docker
sudo usermod -aG docker "$USER" || true

echo "== 2/5 Shared folder + repo =="
mkdir -p "$SHARED"
chmod a+w "$SHARED"
if [ -d "$SHARED/ibmz-datathon-2026-/.git" ]; then
  git -C "$SHARED/ibmz-datathon-2026-" pull --ff-only || true
else
  command -v git >/dev/null 2>&1 || sudo apt-get install -y git
  git clone "$REPO" "$SHARED/ibmz-datathon-2026-"
fi
chmod -R a+w "$SHARED/ibmz-datathon-2026-"

echo "== 3/5 Pulling IBM's Jupyter Lab image (a few minutes) =="
sudo docker pull "$IMAGE"

echo "== 4/5 Starting Jupyter Lab =="
sudo docker rm -f notebook >/dev/null 2>&1 || true
start() {
  sudo docker run -p "$PORT:8888" --name notebook "$@" \
    -v "$SHARED:/home/jovyan/shared" -d "$IMAGE" \
    jupyter lab --ServerApp.token="$TOKEN" >/dev/null
}
start
sleep 8
if ! sudo docker ps --format '{{.Names}}' | grep -q '^notebook$'; then
  echo "   First start failed, retrying with seccomp=unconfined..."
  sudo docker rm -f notebook >/dev/null 2>&1 || true
  start --security-opt seccomp=unconfined
  sleep 8
fi
sudo docker ps --filter name=notebook --format '   {{.Names}}: {{.Status}}'

echo "== 5/5 Opening port $PORT =="
sudo iptables -C INPUT -p tcp --dport "$PORT" -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT -p tcp --dport "$PORT" -j ACCEPT

IP=$(curl -fsS --max-time 5 https://ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
echo
echo "=============================================="
echo " Jupyter Lab is running."
echo " Open:  http://$IP:$PORT"
echo " Token: $TOKEN"
echo " Your repo is in the 'shared' folder."
echo "=============================================="
echo "If the page won't load: sudo docker logs notebook"
