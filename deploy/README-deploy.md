# AION Backend deploy helper

Files in this folder:
- `run_server.sh` - helper script to create venv, install CPU-friendly requirements, install CPU PyTorch, create systemd unit and configure Nginx. Run on the EC2 host from the project root.
- `aion.service` - example systemd unit (for review or manual install in `/etc/systemd/system/`).
- `nginx_aion.conf` - example nginx site file (for review or manual install in `/etc/nginx/sites-available/`).

Quick usage (on EC2, from project root):

```bash
# make script executable
chmod +x deploy/run_server.sh

# run (requires sudo for systemd/nginx parts):
sudo ./deploy/run_server.sh
```

Notes:
- The script installs CPU-only PyTorch (to avoid huge CUDA wheels). If you need GPU support, remove the CPU pin and use a GPU instance with appropriate drivers.
- The script expects the project to be at `/home/ubuntu/AION-backend` for the systemd unit. If your path differs, edit `deploy/aion.service` and `deploy/run_server.sh` accordingly before running.
- If your EC2 disk is small, consider adding swap or increasing the EBS volume before running `pip install` for large packages.
