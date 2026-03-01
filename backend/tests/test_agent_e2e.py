"""End-to-end tests for the FastAPI /api/explain endpoint.

Uses httpx.AsyncClient with ASGITransport to exercise the full HTTP stack
without starting a real server. The LLM interpret_node falls back to
rule-based mode when ANTHROPIC_API_KEY is not set (the default in CI).
"""
import io
import json
import os

import httpx
import joblib
import pytest
from httpx import ASGITransport

from app.agent.graph import build_graph
from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Async httpx client wired directly to the FastAPI ASGI app.

    ASGITransport does not trigger the FastAPI lifespan, so we initialize
    app.state manually here (mirrors what the lifespan does at startup).
    """
    app.state.sessions = {}
    app.state.graph = build_graph()
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
    app.state.sessions.clear()


@pytest.fixture
async def session_id(client, isolation_forest_bytes, sample_schema):
    """Upload an IsolationForest model and return the session_id."""
    response = await client.post(
        "/api/upload-model",
        data={"schema": json.dumps(sample_schema)},
        files={
            "model_file": (
                "model.joblib",
                isolation_forest_bytes,
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


# ---------------------------------------------------------------------------
# Upload tests (sanity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_returns_session_id(client, isolation_forest_bytes, sample_schema):
    response = await client.post(
        "/api/upload-model",
        data={"schema": json.dumps(sample_schema)},
        files={
            "model_file": (
                "model.joblib",
                isolation_forest_bytes,
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["model_type"] == "IsolationForest"
    assert data["feature_count"] == 4
    assert data["status"] == "ready"


# ---------------------------------------------------------------------------
# Explain endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explain_returns_200_with_full_result(client, session_id):
    """Full explain call: prediction, SHAP, LIME, PDP, and summary all present."""
    response = await client.post(
        "/api/explain",
        json={
            "session_id": session_id,
            "input_record": {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5},
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    # Prediction
    pred = data["prediction"]
    assert pred["label"] in ("anomaly", "normal")
    assert pred["model_type"] == "IsolationForest"
    assert isinstance(pred["anomaly_score"], float)

    # Explanations
    expl = data["explanations"]
    assert expl["shap_values"] is not None
    assert set(expl["shap_values"].keys()) == {"f1", "f2", "f3", "f4"}
    assert expl["lime_weights"] is not None
    assert expl["top_features"] is not None

    # Plots
    plots = expl["plots"]
    assert plots["shap_plot_url"] is not None
    assert plots["shap_plot_url"].startswith("/api/plot/")
    assert plots["lime_plot_url"] is not None
    assert len(plots["pdp_plot_urls"]) >= 1

    # Summary (rule-based fallback when no API key)
    summary = data["summary"]
    assert summary["text"] is not None
    assert isinstance(summary["text"], str)

    # No errors
    assert data["errors"] == []


@pytest.mark.asyncio
async def test_explain_invalid_session_returns_404(client):
    response = await client.post(
        "/api/explain",
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "input_record": {"f1": 0.5},
        },
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_explain_missing_features_returns_errors(client, session_id):
    """Providing an incomplete record should produce non-fatal errors, not crash."""
    response = await client.post(
        "/api/explain",
        json={
            "session_id": session_id,
            "input_record": {"f1": 0.5},  # f2, f3, f4 missing
        },
    )
    # Pipeline handles errors gracefully — returns 200 with errors list
    assert response.status_code == 200
    data = response.json()
    assert len(data["errors"]) > 0
    assert any("validation" in e for e in data["errors"])


@pytest.mark.asyncio
async def test_plot_endpoint_serves_png(client, session_id):
    """After /api/explain, the returned plot URL should serve a valid PNG."""
    response = await client.post(
        "/api/explain",
        json={
            "session_id": session_id,
            "input_record": {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5},
        },
    )
    assert response.status_code == 200
    shap_url = response.json()["explanations"]["plots"]["shap_plot_url"]
    assert shap_url is not None

    plot_resp = await client.get(shap_url)
    assert plot_resp.status_code == 200
    assert plot_resp.headers["content-type"] == "image/png"
    # PNG magic bytes
    assert plot_resp.content[:4] == b"\x89PNG"


@pytest.mark.asyncio
async def test_plot_endpoint_404_for_unknown_id(client):
    response = await client.get("/api/plot/nonexistent_plot_id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_explain_anomalous_record(client, session_id):
    """An out-of-distribution record should be labelled as anomaly."""
    response = await client.post(
        "/api/explain",
        json={
            "session_id": session_id,
            "input_record": {"f1": 10.0, "f2": 10.0, "f3": 10.0, "f4": 10.0},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"]["label"] == "anomaly"
    assert data["errors"] == []
