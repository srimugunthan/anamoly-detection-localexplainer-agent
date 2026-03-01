import asyncio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.agent.state import ExplainerState
from app.services.model_loader import get_scorer
from app.services.plot_generator import generate_background_data, save_plot

_PDP_TIMEOUT = 60.0
_MAX_PDP_FEATURES = 3
_GRID_RESOLUTION = 50


async def pdp_node(state: ExplainerState) -> dict:
    """LangGraph node: compute Partial Dependence Plots for top features."""
    if state.get("input_df") is None:
        errors = list(state.get("errors", []))
        errors.append("pdp_skipped: input_df not available (upstream error)")
        return {"errors": errors}

    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _compute_pdp, state),
            timeout=_PDP_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        errors = list(state.get("errors", []))
        errors.append("pdp_timeout: PDP computation exceeded 60s")
        return {"errors": errors}
    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"pdp_error: {exc}")
        return {"errors": errors}


def _compute_pdp(state: ExplainerState) -> dict:
    model = state["model"]
    input_df = state["input_df"]
    schema = state["schema"]
    background_data = state.get("background_data")
    session_id = state["session_id"]
    top_features = state.get("top_features")

    feature_names = list(input_df.columns)

    if background_data is None:
        background_data = generate_background_data(schema)

    scorer = get_scorer(model)

    # Use top_features if already computed (e.g., sequential graph), else first 3 from schema
    if top_features:
        features_to_plot = [f for f in top_features if f in feature_names][:_MAX_PDP_FEATURES]
    else:
        features_to_plot = feature_names[:_MAX_PDP_FEATURES]

    pdp_plot_paths = []
    for feature_name in features_to_plot:
        feat_idx = feature_names.index(feature_name)
        spec = schema["features"].get(feature_name, {})

        if spec.get("type") == "categorical":
            values_list = spec.get("values", [])
            grid = np.arange(len(values_list), dtype=float)
        else:
            col_data = background_data[:, feat_idx]
            lo = spec.get("min", float(col_data.min()))
            hi = spec.get("max", float(col_data.max()))
            grid = np.linspace(lo, hi, _GRID_RESOLUTION)

        # Compute mean anomaly score across background at each grid point
        mean_scores = []
        for val in grid:
            X_temp = background_data.copy()
            X_temp[:, feat_idx] = val
            mean_scores.append(float(np.mean(scorer(X_temp))))

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, mean_scores, lw=2, color="steelblue", label="PDP")

        current_val = float(input_df.iloc[0][feature_name])
        ax.axvline(
            x=current_val,
            color="red",
            linestyle="--",
            lw=1.5,
            label=f"Current: {current_val:.3g}",
        )

        ax.set_xlabel(feature_name)
        ax.set_ylabel("Mean anomaly score")
        ax.set_title(f"Partial Dependence: {feature_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = save_plot(fig, session_id, f"pdp_{feature_name}")
        plt.close("all")
        pdp_plot_paths.append(path)

    return {"pdp_plot_paths": pdp_plot_paths}
