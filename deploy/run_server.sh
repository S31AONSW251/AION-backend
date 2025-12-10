#!/usr/bin/env bash
set -euo pipefail

# Deploy helper for AION-backend (venv + gunicorn + systemd + nginx)
# Run this on the EC2 instance from the repo root: sudo ./deploy/run_server.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Project dir: $PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found, installing python3"
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip
fi

echo "Installing OS packages needed for image processing and builds..."
sudo apt-get update
sudo apt-get install -y build-essential libgl1 libglib2.0-0 libsm6 libxrender1 nginx

# create venv
if [ ! -d "$PROJECT_DIR/venv" ]; then
  echo "Creating virtualenv..."
  python3 -m venv venv
fi

echo "Activating virtualenv"
source venv/bin/activate

echo "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

REQ_FILE=requirements-cpu.txt
if [ ! -f "$REQ_FILE" ]; then
  echo "requirements-cpu.txt not found in project root. Please create or copy it."
  exit 1
fi

echo "Installing Python requirements (no cache to save disk)"
pip install --no-cache-dir -r "$REQ_FILE" || true

# install CPU-only torch explicitly if not present
if ! python -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
  echo "Installing CPU-only PyTorch (this may take a few minutes)"
  pip install --no-cache-dir "torch==2.9.1+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html || true
fi

echo "Ensuring gunicorn is installed"
pip install --no-cache-dir gunicorn || true

# Create systemd unit
SERVICE_PATH=/etc/systemd/system/aion.service
echo "Creating systemd service at $SERVICE_PATH (requires sudo)"
sudo tee "$SERVICE_PATH" > /dev/null <<'SERVICE_EOF'
[Unit]
Description=AION Backend
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/AION-backend
Environment="PATH=/home/ubuntu/AION-backend/venv/bin"
ExecStart=/home/ubuntu/AION-backend/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 server:app
Restart=on-failure

[Install]
WantedBy=multi-user.target
SERVICE_EOF

echo "Reloading systemd and enabling service"
sudo systemctl daemon-reload
sudo systemctl enable --now aion.service || true

# Nginx site
NGINX_CONF=/etc/nginx/sites-available/aion
echo "Writing nginx site file to $NGINX_CONF (requires sudo)"
sudo tee "$NGINX_CONF" > /dev/null <<'NGINX_EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX_EOF

sudo ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/aion
sudo nginx -t && sudo systemctl restart nginx || echo "nginx test failed or restart failed"

echo "Deployment script finished. Check service status with: sudo systemctl status aion.service"
echo "If something failed, inspect logs: sudo journalctl -u aion.service -n 200 --no-pager"

exit 0
