"""
Единая логика выбора устройства для скриптов Проекта (как в run_experiment.py).
"""
from __future__ import annotations

import torch


def resolve_device(name: str, *, no_cuda: bool = False) -> torch.device:
    """
    name:
      - "auto" — CUDA (если есть и не отключён) → MPS (Apple Silicon) → CPU
      - "cpu" | "cuda" | "mps" — явно
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available() and not no_cuda:
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(
    dtype_name: str,
    device: torch.device,
) -> torch.dtype:
    """
    dtype_name: "auto" | "float16" | "bfloat16" | "float32"
    Для auto: float16 на cuda/mps (быстрее и меньше VRAM), float32 на CPU (стабильнее).
    """
    if dtype_name == "auto":
        return torch.float16 if device.type in ("cuda", "mps") else torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
