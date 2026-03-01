"""Integration tests for the LangGraph agent graph (Phase 3)."""
from unittest.mock import AsyncMock

import pytest

from app.agent.graph import build_graph


def _make_initial_state(
    model_bytes: bytes,
    schema: dict,
    record: dict,
    background_data=None,
    session_id: str = "test-session",
) -> dict:
    """Build a full initial state dict with all ExplainerState keys."""
    return {
        "session_id": session_id,
        "model_bytes": model_bytes,
        "schema": schema,
        "input_record": record,
        "background_data": background_data,
        # Not yet set by caller
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_interpret():
    """Async mock for interpret_node that returns a fixed dict."""
    return AsyncMock(
        return_value={
            "explanation_summary": "Mocked LLM summary.",
            "feature_contributions": [{"feature": "f1", "contribution": 0.5}],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_runs_isolation_forest(
    monkeypatch,
    isolation_forest_bytes,
    sample_schema,
    normal_record,
    background_data_array,
    mock_interpret,
):
    """Full graph invoke with IsolationForest should populate all expected fields."""
    # Patch interpret_node in the graph module namespace (where it is imported)
    monkeypatch.setattr("app.agent.graph.interpret_node", mock_interpret)

    graph = build_graph()
    initial_state = _make_initial_state(
        model_bytes=isolation_forest_bytes,
        schema=sample_schema,
        record=normal_record,
        background_data=background_data_array,
    )

    result = await graph.ainvoke(initial_state)

    # Prediction
    assert result["prediction_label"] in ("anomaly", "normal"), (
        f"Unexpected prediction_label: {result['prediction_label']!r}"
    )

    # SHAP
    assert result.get("shap_values") is not None, "shap_values should be populated"
    assert isinstance(result["shap_values"], dict)
    assert len(result["shap_values"]) > 0, "shap_values dict should not be empty"

    # LIME
    assert result.get("lime_weights") is not None, "lime_weights should be populated"
    assert isinstance(result["lime_weights"], dict)
    assert len(result["lime_weights"]) > 0, "lime_weights dict should not be empty"

    # PDP
    assert result.get("pdp_plot_paths") is not None, "pdp_plot_paths should be populated"
    assert isinstance(result["pdp_plot_paths"], list)
    assert len(result["pdp_plot_paths"]) > 0, "pdp_plot_paths list should not be empty"

    # LLM interpretation (mocked)
    assert isinstance(result.get("explanation_summary"), str)


@pytest.mark.asyncio
async def test_graph_propagates_errors_on_bad_model(
    monkeypatch,
    sample_schema,
    normal_record,
    mock_interpret,
):
    """Graph should not raise an exception for invalid model bytes; errors list must be non-empty."""
    monkeypatch.setattr("app.agent.graph.interpret_node", mock_interpret)

    graph = build_graph()
    garbage_bytes = b"this is not a valid model"
    initial_state = _make_initial_state(
        model_bytes=garbage_bytes,
        schema=sample_schema,
        record=normal_record,
    )

    # Should not raise
    result = await graph.ainvoke(initial_state)

    assert isinstance(result.get("errors"), list), "errors should be a list"
    assert len(result["errors"]) > 0, "errors list should be non-empty after bad model"


@pytest.mark.asyncio
async def test_graph_parallel_tools_all_run(
    monkeypatch,
    isolation_forest_bytes,
    sample_schema,
    normal_record,
    background_data_array,
    mock_interpret,
):
    """After a successful run, all three tool outputs (shap, lime, pdp) must be populated."""
    monkeypatch.setattr("app.agent.graph.interpret_node", mock_interpret)

    graph = build_graph()
    initial_state = _make_initial_state(
        model_bytes=isolation_forest_bytes,
        schema=sample_schema,
        record=normal_record,
        background_data=background_data_array,
    )

    result = await graph.ainvoke(initial_state)

    # All three parallel tools must have run and populated their keys
    assert result.get("shap_values") is not None, "shap_values missing — SHAP tool did not run"
    assert result.get("lime_weights") is not None, "lime_weights missing — LIME tool did not run"
    assert result.get("pdp_plot_paths") is not None, "pdp_plot_paths missing — PDP tool did not run"

    # Verify keys are genuinely populated (not None / empty)
    assert isinstance(result["shap_values"], dict) and len(result["shap_values"]) > 0
    assert isinstance(result["lime_weights"], dict) and len(result["lime_weights"]) > 0
    assert isinstance(result["pdp_plot_paths"], list) and len(result["pdp_plot_paths"]) > 0
