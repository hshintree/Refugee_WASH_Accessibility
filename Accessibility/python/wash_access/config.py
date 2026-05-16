from __future__ import annotations

import os
from pathlib import Path

# Default: Accessibility/ directory (contains ./data/)
# wash_access/config.py -> python/ -> Accessibility/
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent


def get_data_root() -> Path:
    env = os.environ.get("WASH_ACC_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DATA_ROOT.resolve()


def get_out_dir() -> Path:
    env = os.environ.get("WASH_ACC_OUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return get_data_root() / "out"
