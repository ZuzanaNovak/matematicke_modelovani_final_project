# utils.py
import os
import numpy as np

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def tanh(z):
    # safe tanh to avoid overflow
    import numpy as np
    return np.tanh(np.clip(z, -50.0, 50.0))

def value_at(t_query, T, series) -> float:
    """Return value of `series` at time closest to t_query."""
    import numpy as np
    return float(series[np.argmin(np.abs(T - t_query))])
