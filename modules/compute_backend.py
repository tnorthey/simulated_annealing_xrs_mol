from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Backend:
    name: str
    xp: Any
    is_gpu: bool
    is_emulated: bool


def get_backend(backend_name: str, emulate: bool = False) -> Backend:
    """Return backend module and flags for CPU/CUDA."""
    name = str(backend_name).lower().strip()
    if name == "cpu":
        return Backend(name="cpu", xp=np, is_gpu=False, is_emulated=False)
    if name != "cuda":
        raise ValueError(
            f"Unsupported gpu_backend '{backend_name}'. Use 'cpu' or 'cuda'."
        )

    if emulate:
        return Backend(name="cuda", xp=np, is_gpu=False, is_emulated=True)

    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "CUDA backend selected but CuPy is not available. "
            "Install cupy or set gpu_emulation_bool = true."
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "CUDA backend selected but no GPU device was detected."
        ) from exc

    if device_count < 1:  # pragma: no cover - environment dependent
        raise RuntimeError("CUDA backend selected but no GPU devices found.")

    return Backend(name="cuda", xp=cp, is_gpu=True, is_emulated=False)


def to_numpy(x, xp):
    """Convert backend array to numpy if needed."""
    if xp is np:
        return x
    return xp.asnumpy(x)
