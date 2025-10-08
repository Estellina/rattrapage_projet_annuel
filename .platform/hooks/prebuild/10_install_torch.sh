#!/bin/bash
# Prebuild hook for AWS Elastic Beanstalk (Python platform)
# Installs PyTorch CPU wheels compatible with Python 3.11 on AL2023.
# This runs before your app is deployed, inside the EB EC2 instance.
set -euo pipefail

echo "[prebuild] Installing PyTorch (CPU)..."

# Activate the Elastic Beanstalk virtualenv
if [ -d "/var/app/venv" ]; then
  VENV_DIR="$(ls -d /var/app/venv/* | head -n 1)"
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
fi

# Upgrade pip then install torch/vision/audio (CPU wheels)
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu       torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

echo "[prebuild] PyTorch CPU installed successfully."
