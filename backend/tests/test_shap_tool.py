import os

import pytest

from app.agent.nodes.predict import predict_node
from app.agent.nodes.shap_tool import shap_node


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _base_state(model_bytes, schema, record, background_data):
    return {
        "session_id": "test-shap",
        "model_bytes": model_bytes,
        "schema": schema,
        "input_record": record,
        "background_data": background_data,
        "model": None,
        "model_type": "",
        "prediction_label": "",
        "anomaly_score": 0.0,
        "input_df": None,
        "shap_values": None,
        "shap_plot_path": None,
        "shap_force_plot_path": None,
        "lime_weights": None,
        "lime_plot_path": None,
        "pdp_plot_paths": None,
        "top_features": None,
        "explanation_summary": None,
        "feature_contributions": None,
        "errors": [],
    }


async def _run(model_bytes, schema, record, background_data):
    base = _base_state(model_bytes, schema, record, background_data)
    predict_result = await predict_node(base)
    state = {**base, **predict_result}
    return state, await shap_node(state)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shap_node_isolation_forest(
    isolation_forest_bytes, sample_schema, normal_record, background_data_array
):
    state, result = await _run(
        isolation_forest_bytes, sample_schema, normal_record, background_data_array
    )

    assert "shap_values" in result
    assert set(result["shap_values"].keys()) == {"f1", "f2", "f3", "f4"}

    assert result["shap_plot_path"].endswith(".png")
    assert result["shap_force_plot_path"].endswith(".png")
    assert os.path.exists(result["shap_plot_path"])
    assert os.path.exists(result["shap_force_plot_path"])

    assert isinstance(result["top_features"], list)
    assert 1 <= len(result["top_features"]) <= 5
    assert all(f in {"f1", "f2", "f3", "f4"} for f in result["top_features"])

    assert result.get("errors", []) == []


@pytest.mark.asyncio
async def test_shap_node_ocsvm(
    ocsvm_bytes, sample_schema, normal_record, background_data_array
):
    state, result = await _run(
        ocsvm_bytes, sample_schema, normal_record, background_data_array
    )

    assert "shap_values" in result
    assert set(result["shap_values"].keys()) == {"f1", "f2", "f3", "f4"}
    assert result["shap_plot_path"].endswith(".png")
    assert os.path.exists(result["shap_plot_path"])
    assert result.get("errors", []) == []


@pytest.mark.asyncio
async def test_shap_node_skips_without_input_df(sample_schema):
    """If predict_node failed (input_df is None), shap_node appends a skip error."""
    state = _base_state(b"bad", sample_schema, {"f1": 0.5}, None)
    # input_df stays None — simulate upstream failure
    result = await shap_node(state)
    assert any("shap_skipped" in e for e in result.get("errors", []))


@pytest.mark.asyncio
async def test_shap_values_are_floats(
    isolation_forest_bytes, sample_schema, normal_record, background_data_array
):
    _, result = await _run(
        isolation_forest_bytes, sample_schema, normal_record, background_data_array
    )
    for v in result["shap_values"].values():
        assert isinstance(v, float)
