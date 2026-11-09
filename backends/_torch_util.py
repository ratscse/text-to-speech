"""Small torch helpers shared by the ML backends (imported only inside workers)."""
from __future__ import annotations

import gc
import os

import numpy as np


def setup_threads(options: dict) -> None:
    import torch

    n = int(options.get("threads") or os.cpu_count() or 4)
    torch.set_num_threads(max(1, n))


def to_numpy(wav) -> np.ndarray:
    try:
        import torch

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def seed_all(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
