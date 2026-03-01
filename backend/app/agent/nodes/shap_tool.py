import asyncio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from app.agent.state import ExplainerState
from app.services.model_loader import TREE_MODELS, get_scorer
from app.services.plot_generator import (
    generate_background_data,
    save_plot,
    summarize_background,
)

_SHAP_TIMEOUT = 60.0


async def shap_node(state: ExplainerState) -> dict:
    """LangGraph node: compute SHAP explanations and generate plots."""
    # Skip if upstream model load failed
    if state.get("input_df") is None:
        errors = list(state.get("errors", []))
        errors.append("shap_skipped: input_df not available (upstream error)")
        return {"errors": errors}

    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _compute_shap, state),
            timeout=_SHAP_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        errors = list(state.get("errors", []))
        errors.append("shap_timeout: SHAP computation exceeded 60s")
        return {"errors": errors}
    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"shap_error: {exc}")
        return {"errors": errors}


def _compute_shap(state: ExplainerState) -> dict:
    model = state["model"]
    model_type = state["model_type"]
    input_df = state["input_df"]
    schema = state["schema"]
    background_data = state.get("background_data")
    session_id = state["session_id"]

    feature_names = list(input_df.columns)

    if background_data is None:
        background_data = generate_background_data(schema)

    if model_type in TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(input_df)
        single_exp = explanation[0]
        shap_vals = np.asarray(single_exp.values)
        base_val = single_exp.base_values
        if isinstance(base_val, np.ndarray):
            base_val = float(base_val.flat[0])
        else:
            base_val = float(base_val)
        # Multi-output classifiers return (n_features, n_classes).
        # Take the LAST column (anomaly/positive class) so f(x) in the waterfall
        # matches the anomaly probability shown in the prediction badge.
        if shap_vals.ndim == 2:
            class_idx = shap_vals.shape[1] - 1  # last class = anomaly class for binary
            shap_vals = shap_vals[:, class_idx]
            base_val_arr = np.asarray(single_exp.base_values)
            base_val = float(base_val_arr.flat[class_idx] if base_val_arr.ndim > 0 and base_val_arr.size > 1 else base_val_arr.flat[0])
            single_exp = shap.Explanation(
                values=shap_vals,
                base_values=base_val,
                data=input_df.values[0],
                feature_names=feature_names,
            )
    else:
        background_summary = summarize_background(background_data)
        scorer = get_scorer(model)
        explainer = shap.KernelExplainer(scorer, background_summary)
        shap_vals_raw = explainer.shap_values(input_df.values, nsamples=100)
        if isinstance(shap_vals_raw, list):
            shap_vals = np.asarray(shap_vals_raw[0][0])
        else:
            shap_vals = np.asarray(shap_vals_raw[0])
        ev = explainer.expected_value
        base_val = float(np.asarray(ev).flat[0])
        single_exp = shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=input_df.values[0],
            feature_names=feature_names,
        )

    # Ensure strictly 1-D before scalar conversion (numpy >=1.25 rejects non-0d arrays).
    # Use last column (anomaly class) to stay consistent with predict_proba[:, -1].
    shap_vals = np.asarray(shap_vals, dtype=float)
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[:, -1]

    shap_values_dict = {fn: float(v) for fn, v in zip(feature_names, shap_vals)}
    top_features = sorted(
        shap_values_dict, key=lambda k: abs(shap_values_dict[k]), reverse=True
    )[:5]

    # --- Waterfall plot ---
    plt.close("all")
    shap.plots.waterfall(single_exp, show=False)
    waterfall_path = save_plot(plt.gcf(), session_id, "shap_waterfall")
    plt.close("all")

    # --- SHAP impact bar chart (force-plot equivalent, more reliable as static PNG) ---
    sorted_pairs = sorted(zip(feature_names, shap_vals.tolist()), key=lambda x: x[1])
    names = [p[0] for p in sorted_pairs]
    vals = [p[1] for p in sorted_pairs]
    colors = ["#ff0051" if v < 0 else "#008bfb" for v in vals]

    fig_force, ax = plt.subplots(figsize=(8, max(3, len(feature_names) * 0.5 + 1)))
    ax.barh(names, vals, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(f"Feature Impact  |  base = {base_val:.4f}")
    fig_force.tight_layout()
    force_path = save_plot(fig_force, session_id, "shap_force")
    plt.close("all")

    return {
        "shap_values": shap_values_dict,
        "shap_plot_path": waterfall_path,
        "shap_force_plot_path": force_path,
        "top_features": top_features,
    }
