import asyncio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

from app.agent.state import ExplainerState
from app.services.model_loader import get_scorer
from app.services.plot_generator import generate_background_data, save_plot

_LIME_TIMEOUT = 60.0


async def lime_node(state: ExplainerState) -> dict:
    """LangGraph node: compute LIME explanations and generate a bar chart."""
    if state.get("input_df") is None:
        errors = list(state.get("errors", []))
        errors.append("lime_skipped: input_df not available (upstream error)")
        return {"errors": errors}

    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _compute_lime, state),
            timeout=_LIME_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        errors = list(state.get("errors", []))
        errors.append("lime_timeout: LIME computation exceeded 60s")
        return {"errors": errors}
    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"lime_error: {exc}")
        return {"errors": errors}


def _compute_lime(state: ExplainerState) -> dict:
    model = state["model"]
    input_df = state["input_df"]
    schema = state["schema"]
    background_data = state.get("background_data")
    session_id = state["session_id"]

    feature_names = list(input_df.columns)
    features = schema["features"]

    # Categorical feature indices (already encoded as integers in input_df)
    categorical_indices = [
        i
        for i, (name, spec) in enumerate(features.items())
        if spec.get("type") == "categorical"
    ]

    if background_data is None:
        background_data = generate_background_data(schema)

    scorer = get_scorer(model)

    explainer = LimeTabularExplainer(
        training_data=background_data,
        feature_names=feature_names,
        categorical_features=categorical_indices,
        mode="regression",
    )

    input_row = input_df.values[0]
    exp = explainer.explain_instance(
        data_row=input_row,
        predict_fn=scorer,
        num_features=len(feature_names),
    )

    lime_weights = dict(exp.as_list())

    # Bar chart
    fig = exp.as_pyplot_figure()
    lime_path = save_plot(fig, session_id, "lime")
    plt.close("all")

    return {
        "lime_weights": lime_weights,
        "lime_plot_path": lime_path,
    }
