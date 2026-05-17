import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Keep all HuggingFace cache (datasets, models, hub) inside the repo so the
# project is self-contained — no surprise gigabytes under ~/.cache.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HF_CACHE = _REPO_ROOT / "data" / ".hf_cache"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_HF_CACHE / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_CACHE / "hub"))

import random
import numpy as np
import torch


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
