import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.config import PLOT_DIR


def save_plot(fig, session_id: str, tool_name: str) -> str:
    """Save a matplotlib Figure to PLOT_DIR. Returns the absolute path string."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_id = f"{session_id}_{tool_name}_{uuid.uuid4().hex[:8]}"
    path = PLOT_DIR / f"{plot_id}.png"
    fig.savefig(path, bbox_inches="tight", dpi=100)
    return str(path)


def generate_background_data(schema: dict, n: int = 200) -> np.ndarray:
    """Generate synthetic background samples from schema-defined feature ranges.

    For float/int features: uniform sampling between [min, max].
    For categorical features: uniform sampling over encoded integer indices.
    """
    rng = np.random.default_rng(42)
    features = schema["features"]
    cols = []
    for _name, spec in features.items():
        ftype = spec.get("type", "float")
        if ftype in ("float", "int"):
            lo = float(spec.get("min", 0.0))
            hi = float(spec.get("max", 1.0))
            cols.append(rng.uniform(lo, hi, n))
        elif ftype == "categorical":
            n_cats = max(1, len(spec.get("values", [])))
            cols.append(rng.integers(0, n_cats, n).astype(float))
        else:
            cols.append(rng.uniform(0.0, 1.0, n))
    if not cols:
        return np.empty((n, 0))
    return np.column_stack(cols)


def summarize_background(background_data: np.ndarray, k: int = 50) -> Any:
    """Return a shap.kmeans summary for use with KernelExplainer.

    Reduces compute cost for large background datasets.
    """
    import shap

    k = min(k, len(background_data))
    return shap.kmeans(background_data, k)
