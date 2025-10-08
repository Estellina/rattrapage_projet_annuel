    #!/usr/bin/env bash
    set -euo pipefail

    echo "[prebuild] Torch prebuild hook running..."

    PYBIN="$(command -v python3 || command -v python)"
    PIP="$PYBIN -m pip"

    # Check if torch + torchvision already installed
    if "$PYBIN" - <<'PY'
import importlib, sys
for m in ("torch","torchvision"):
    try:
        importlib.import_module(m)
    except Exception:
        sys.exit(1)
print("ok")
PY
    then
        echo "[prebuild] Skipping: torch/torchvision already installed."
        exit 0
    fi

    echo "[prebuild] Installing CPU wheels for torch + torchvision"
    "$PIP" install --upgrade pip
    # CPU-only wheels from the official PyTorch index (compatible with Amazon Linux 2023 / Python 3.11)
    "$PIP" install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1

    echo "[prebuild] Torch installation completed."
