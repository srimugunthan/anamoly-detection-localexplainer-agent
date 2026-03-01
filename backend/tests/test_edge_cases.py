"""M7 edge-case tests covering the scenarios listed in plan.md Phase 7.

Scenarios:
  1. Corrupt model file → 422 with error message
  2. Schema missing 'features' key → 422
  3. Path traversal attempt on /api/plot → 400
  4. No reference CSV (synthetic background) → pipeline completes, no errors
  5. Large reference CSV (>1000 rows) → K-means summary applied, pipeline completes
  6. One-Class SVM → KernelExplainer fallback, correct score
  7. LLM API failure (mocked) → 200 with plots + raw values, graceful degradation
"""
import io
import json
from unittest.mock import AsyncMock, patch

import httpx
import joblib
import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport

from app.agent.graph import build_graph
from app.main import app


# ---------------------------------------------------------------------------
# Shared client fixture (same pattern as test_agent_e2e.py)
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
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
    """Upload an IsolationForest and return session_id."""
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
    return response.json()["session_id"]


# ---------------------------------------------------------------------------
# Scenario 1: Corrupt model file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_corrupt_model_returns_422(client, sample_schema):
    """Uploading random bytes as a model should return 422."""
    response = await client.post(
        "/api/upload-model",
        data={"schema": json.dumps(sample_schema)},
        files={
            "model_file": (
                "bad.pkl",
                b"this is not a valid model binary",
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 422
    assert "detail" in response.json()


# ---------------------------------------------------------------------------
# Scenario 2: Schema validation — missing 'features' key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_schema_missing_features(client, isolation_forest_bytes):
    """Schema JSON without a 'features' key should return 422."""
    bad_schema = {"columns": ["f1", "f2"]}
    response = await client.post(
        "/api/upload-model",
        data={"schema": json.dumps(bad_schema)},
        files={
            "model_file": (
                "model.joblib",
                isolation_forest_bytes,
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Scenario 3: Path traversal prevention
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plot_path_traversal_rejected(client):
    """plot_id containing '..' or '/' should return 400, not serve files."""
    # Note: null bytes are rejected by the HTTP client before reaching the server,
    # so we only test the patterns the server sanitizer must handle.
    traversal_ids = ["../etc/passwd", "../../secret", "foo/bar"]
    for bad_id in traversal_ids:
        r = await client.get(f"/api/plot/{bad_id}")
        assert r.status_code in (400, 404), (
            f"Expected 400/404 for id {bad_id!r}, got {r.status_code}"
        )


# ---------------------------------------------------------------------------
# Scenario 4: No reference CSV → synthetic background generated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explain_without_reference_csv(client, isolation_forest_bytes, sample_schema):
    """When no reference CSV is provided, synthetic background is used — pipeline must complete."""
    upload = await client.post(
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
    assert upload.status_code == 200
    sid = upload.json()["session_id"]

    explain = await client.post(
        "/api/explain",
        json={"session_id": sid, "input_record": {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5}},
    )
    assert explain.status_code == 200
    data = explain.json()
    assert data["prediction"]["label"] in ("anomaly", "normal")
    assert data["explanations"]["shap_values"] is not None


# ---------------------------------------------------------------------------
# Scenario 5: Large reference CSV (>1000 rows) → K-means applied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explain_with_large_reference_csv(client, isolation_forest_bytes, sample_schema):
    """A reference CSV with >1000 rows should be processed (K-means summary) without error."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(0, 1, (1200, 4)), columns=["f1", "f2", "f3", "f4"])
    csv_bytes = df.to_csv(index=False).encode()

    upload = await client.post(
        "/api/upload-model",
        data={"schema": json.dumps(sample_schema)},
        files={
            "model_file": ("model.joblib", isolation_forest_bytes, "application/octet-stream"),
            "reference_csv": ("bg.csv", csv_bytes, "text/csv"),
        },
    )
    assert upload.status_code == 200
    sid = upload.json()["session_id"]

    explain = await client.post(
        "/api/explain",
        json={"session_id": sid, "input_record": {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5}},
    )
    assert explain.status_code == 200
    data = explain.json()
    assert data["errors"] == []
    assert data["explanations"]["shap_values"] is not None


# ---------------------------------------------------------------------------
# Scenario 6: One-Class SVM → KernelExplainer fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explain_ocsvm_kernel_explainer_fallback(client, ocsvm_bytes, sample_schema):
    """One-Class SVM should trigger KernelExplainer and still produce SHAP values."""
    upload = await client.post(
        "/api/upload-model",
        data={"schema": json.dumps(sample_schema)},
        files={
            "model_file": ("ocsvm.joblib", ocsvm_bytes, "application/octet-stream"),
        },
    )
    assert upload.status_code == 200
    data = upload.json()
    assert data["model_type"] == "OneClassSVM"
    sid = data["session_id"]

    explain = await client.post(
        "/api/explain",
        json={"session_id": sid, "input_record": {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5}},
    )
    assert explain.status_code == 200
    body = explain.json()
    assert body["prediction"]["label"] in ("anomaly", "normal")
    assert isinstance(body["prediction"]["anomaly_score"], float)
    assert body["explanations"]["shap_values"] is not None


# ---------------------------------------------------------------------------
# Scenario 7: LLM API failure (mocked) → graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_failure_still_returns_plots(client, session_id):
    """If call_llm raises LLMError, the response must still include SHAP/LIME plots."""
    from app.services.llm_client import LLMError

    with (
        patch("app.agent.nodes.interpret._has_api_key", return_value=True),
        patch(
            "app.agent.nodes.interpret.call_llm",
            new=AsyncMock(side_effect=LLMError("Simulated LLM failure")),
        ),
    ):
        explain = await client.post(
            "/api/explain",
            json={
                "session_id": session_id,
                "input_record": {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5},
            },
        )
    # LLM errors are non-fatal — should return 200 with rule-based fallback
    assert explain.status_code == 200
    data = explain.json()
    # Plots must still be present
    assert data["explanations"]["plots"]["shap_plot_url"] is not None
    assert data["explanations"]["plots"]["lime_plot_url"] is not None
    # Rule-based summary should still be set
    assert data["summary"]["text"] is not None
