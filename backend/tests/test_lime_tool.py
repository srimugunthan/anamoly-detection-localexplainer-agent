import os

import pytest

from app.agent.nodes.lime_tool import lime_node
from app.agent.nodes.predict import predict_node


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _base_state(model_bytes, schema, record, background_data):
    return {
        "session_id": "test-lime",
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
    return state, await lime_node(state)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lime_node_isolation_forest(
    isolation_forest_bytes, sample_schema, normal_record, background_data_array
):
    _, result = await _run(
        isolation_forest_bytes, sample_schema, normal_record, background_data_array
    )

    assert "lime_weights" in result
    assert len(result["lime_weights"]) > 0

    assert result["lime_plot_path"].endswith(".png")
    assert os.path.exists(result["lime_plot_path"])

    assert result.get("errors", []) == []


@pytest.mark.asyncio
async def test_lime_node_ocsvm(
    ocsvm_bytes, sample_schema, normal_record, background_data_array
):
    _, result = await _run(
        ocsvm_bytes, sample_schema, normal_record, background_data_array
    )

    assert "lime_weights" in result
    assert result["lime_plot_path"].endswith(".png")
    assert os.path.exists(result["lime_plot_path"])
    assert result.get("errors", []) == []


@pytest.mark.asyncio
async def test_lime_node_skips_without_input_df(sample_schema):
    state = _base_state(b"bad", sample_schema, {"f1": 0.5}, None)
    result = await lime_node(state)
    assert any("lime_skipped" in e for e in result.get("errors", []))


@pytest.mark.asyncio
async def test_lime_weights_are_dict(
    isolation_forest_bytes, sample_schema, normal_record, background_data_array
):
    _, result = await _run(
        isolation_forest_bytes, sample_schema, normal_record, background_data_array
    )
    assert isinstance(result["lime_weights"], dict)
    for v in result["lime_weights"].values():
        assert isinstance(v, float)
