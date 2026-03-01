"""Tests for the interpret_node (Phase 4 — LLM Interpretation Node).

All tests avoid real LLM API calls by either:
- Mocking ``app.agent.nodes.interpret.call_llm`` via unittest.mock.patch
  (works because call_llm is imported at module level in interpret.py).
- Setting ANTHROPIC_API_KEY to an empty string so the node uses the
  rule-based fallback path.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.nodes.interpret import interpret_node


# ---------------------------------------------------------------------------
# Shared fixture: a minimal ExplainerState dict
# ---------------------------------------------------------------------------


def _base_state(
    shap_plot_path: str | None = None,
    lime_plot_path: str | None = None,
    shap_force_plot_path: str | None = None,
    pdp_plot_paths: list[str] | None = None,
) -> dict:
    return {
        "session_id": "test-interpret",
        "model_bytes": b"",
        "schema": {
            "features": {
                "f1": {"type": "float", "min": 0.0, "max": 1.0},
                "f2": {"type": "float", "min": 0.0, "max": 1.0},
                "f3": {"type": "float", "min": 0.0, "max": 1.0},
                "f4": {"type": "float", "min": 0.0, "max": 1.0},
            }
        },
        "input_record": {"f1": 0.9, "f2": 0.1, "f3": 0.6, "f4": 0.8},
        "background_data": None,
        "model": None,
        "model_type": "IsolationForest",
        "prediction_label": "anomaly",
        "anomaly_score": -0.312,
        "input_df": None,
        # Explainability outputs
        "shap_values": {"f1": -0.3, "f2": 0.1, "f3": 0.05, "f4": -0.2},
        "shap_plot_path": shap_plot_path,
        "shap_force_plot_path": shap_force_plot_path,
        "lime_weights": {"f1 <= 0.5": -0.25, "f3 > 0.3": 0.12},
        "lime_plot_path": lime_plot_path,
        "pdp_plot_paths": pdp_plot_paths,
        "top_features": ["f1", "f4", "f2", "f3"],
        # LLM outputs (to be filled by interpret_node)
        "explanation_summary": None,
        "feature_contributions": None,
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Helper: a valid LLM JSON response
# ---------------------------------------------------------------------------


def _valid_llm_json() -> str:
    return json.dumps(
        {
            "summary": "The record was flagged as an anomaly primarily due to f1 being unusually low.",
            "top_contributors": [
                {
                    "feature": "f1",
                    "impact": "high",
                    "direction": "decreases_anomaly",
                    "reason": "f1 is well below the expected range.",
                },
                {
                    "feature": "f4",
                    "impact": "high",
                    "direction": "decreases_anomaly",
                    "reason": "f4 deviates significantly from the baseline.",
                },
            ],
            "consistency_note": "SHAP and LIME agree that f1 is the dominant driver.",
            "next_steps": ["Investigate f1 data pipeline", "Review f4 thresholds"],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: mock LLM returns valid JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interpret_node_with_mock_llm():
    """When LLM returns valid JSON, explanation_summary is non-empty and
    feature_contributions is a list."""
    state = _base_state()

    valid_json = _valid_llm_json()

    with (
        patch("app.agent.nodes.interpret.ANTHROPIC_API_KEY", "fake-key-abc"),
        patch("app.agent.nodes.interpret.LLM_PROVIDER", "anthropic"),
        patch(
            "app.agent.nodes.interpret.call_llm",
            new=AsyncMock(return_value=valid_json),
        ),
    ):
        result = await interpret_node(state)

    assert "explanation_summary" in result
    assert isinstance(result["explanation_summary"], str)
    assert len(result["explanation_summary"]) > 0

    assert "feature_contributions" in result
    assert isinstance(result["feature_contributions"], list)


# ---------------------------------------------------------------------------
# Test 2: no API key → rule-based fallback (no LLM call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interpret_node_fallback_no_llm():
    """When ANTHROPIC_API_KEY is empty, interpret_node must return a rule-based
    summary without calling the LLM."""
    state = _base_state()

    with (
        patch("app.agent.nodes.interpret.ANTHROPIC_API_KEY", ""),
        patch("app.agent.nodes.interpret.LLM_PROVIDER", "anthropic"),
        patch(
            "app.agent.nodes.interpret.call_llm",
            new=AsyncMock(side_effect=AssertionError("LLM should NOT be called")),
        ) as mock_llm,
    ):
        result = await interpret_node(state)
        # Ensure the LLM was not called
        mock_llm.assert_not_called()

    assert "explanation_summary" in result
    assert isinstance(result["explanation_summary"], str)
    assert len(result["explanation_summary"]) > 0

    # Rule-based summary should mention the top SHAP feature
    assert "f1" in result["explanation_summary"]

    assert "feature_contributions" in result
    assert isinstance(result["feature_contributions"], list)


# ---------------------------------------------------------------------------
# Test 3: LLM returns non-JSON → graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interpret_node_invalid_json_from_llm():
    """When the LLM returns a non-JSON string, explanation_summary should be
    set to the raw string and feature_contributions should be an empty list."""
    state = _base_state()

    raw_text = "Sorry, I cannot provide a JSON response right now."

    with (
        patch("app.agent.nodes.interpret.ANTHROPIC_API_KEY", "fake-key-abc"),
        patch("app.agent.nodes.interpret.LLM_PROVIDER", "anthropic"),
        patch(
            "app.agent.nodes.interpret.call_llm",
            new=AsyncMock(return_value=raw_text),
        ),
    ):
        result = await interpret_node(state)

    assert result["explanation_summary"] == raw_text
    assert result["feature_contributions"] == []


# ---------------------------------------------------------------------------
# Test 4: missing plot files → no FileNotFoundError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interpret_node_missing_plots():
    """With all plot paths set to None, interpret_node must return a summary
    without raising any file-not-found errors."""
    state = _base_state(
        shap_plot_path=None,
        lime_plot_path=None,
        shap_force_plot_path=None,
        pdp_plot_paths=None,
    )

    with (
        patch("app.agent.nodes.interpret.ANTHROPIC_API_KEY", ""),
        patch("app.agent.nodes.interpret.LLM_PROVIDER", "anthropic"),
    ):
        result = await interpret_node(state)

    assert "explanation_summary" in result
    assert isinstance(result["explanation_summary"], str)
    assert len(result["explanation_summary"]) > 0
    assert "feature_contributions" in result
    assert isinstance(result["feature_contributions"], list)
