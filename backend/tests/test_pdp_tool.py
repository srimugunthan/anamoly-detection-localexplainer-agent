import os

import pytest

from app.agent.nodes.pdp_tool import pdp_node
from app.agent.nodes.predict import predict_node


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _base_state(model_bytes, schema, record, background_data):
    return {
        "session_id": "test-pdp",
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


async def _run(model_bytes, schema, record, background_data, top_features=None):
    base = _base_state(model_bytes, schema, record, background_data)
    predict_result = await predict_node(base)
    state = {**base, **predict_result, "top_features": top_features}
    return state, await pdp_node(state)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pdp_node_isolation_forest(
    isolation_forest_bytes, sample_schema, normal_record, background_data_array
):
    _, result = await _run(
        isolation_forest_bytes, sample_schema, normal_record, background_data_array
    )

    assert "pdp_plot_paths" in result
    paths = result["pdp_plot_paths"]
    # 4 features → up to 3 PDP plots
    assert 1 <= len(paths) <= 3
    for p in paths:
        assert p.endswith(".png")
        assert os.path.exists(p)

    assert result.get("errors", []) == []


@pytest.mark.asyncio
async def test_pdp_node_respects_top_features(
    isolation_forest_bytes, sample_schema, normal_record, background_data_array
):
    _, result = await _run(
        isolation_forest_bytes,
        sample_schema,
        normal_record,
        background_data_array,
        top_features=["f3", "f1"],  # only 2 features
    )
    assert len(result["pdp_plot_paths"]) == 2


@pytest.mark.asyncio
async def test_pdp_node_ocsvm(
    ocsvm_bytes, sample_schema, normal_record, background_data_array
):
    _, result = await _run(
        ocsvm_bytes, sample_schema, normal_record, background_data_array
    )
    assert len(result["pdp_plot_paths"]) >= 1
    assert all(os.path.exists(p) for p in result["pdp_plot_paths"])


@pytest.mark.asyncio
async def test_pdp_node_skips_without_input_df(sample_schema):
    state = _base_state(b"bad", sample_schema, {"f1": 0.5}, None)
    result = await pdp_node(state)
    assert any("pdp_skipped" in e for e in result.get("errors", []))
