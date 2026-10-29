from __future__ import annotations

from typing import Any, Dict

import numpy as np


def array_to_list(arr: np.ndarray) -> list:
    return arr.astype(float).tolist()


def list_to_array(lst: list) -> np.ndarray:
    return np.array(lst, dtype=np.float64)


def pack_weights(weights: np.ndarray, bias: float) -> Dict[str, Any]:
    return {"weights": array_to_list(weights), "bias": float(bias)}


def unpack_weights(payload: Dict[str, Any]) -> tuple:
    return list_to_array(payload["weights"]), float(payload["bias"])
