import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["explain"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ExplainRequest(BaseModel):
    session_id: str
    input_record: dict


class PredictionResult(BaseModel):
    label: str
    anomaly_score: float
    model_type: str


class PlotURLs(BaseModel):
    shap_plot_url: Optional[str] = None
    shap_force_plot_url: Optional[str] = None
    lime_plot_url: Optional[str] = None
    pdp_plot_urls: list[str] = []


class ExplanationsResult(BaseModel):
    shap_values: Optional[dict] = None
    lime_weights: Optional[dict] = None
    top_features: Optional[list[str]] = None
    plots: PlotURLs


class SummaryResult(BaseModel):
    text: Optional[str] = None
    feature_contributions: Optional[list[dict]] = None


class ExplainResponse(BaseModel):
    prediction: PredictionResult
    explanations: ExplanationsResult
    summary: SummaryResult
    errors: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_to_plot_url(path: Optional[str]) -> Optional[str]:
    """Convert an absolute plot file path to a relative API URL."""
    if not path:
        return None
    return f"/api/plot/{Path(path).stem}"


def _parse_background_csv(csv_bytes: bytes, schema: dict) -> Optional[np.ndarray]:
    """Parse uploaded background CSV bytes into a numpy array (schema column order)."""
    try:
        feature_names = list(schema["features"].keys())
        df = pd.read_csv(io.BytesIO(csv_bytes))
        available = [f for f in feature_names if f in df.columns]
        if not available:
            return None
        return df[available].to_numpy(dtype=float)
    except Exception:
        return None


def _build_initial_state(
    session_id: str,
    model_bytes: bytes,
    schema: dict,
    input_record: dict,
    background_data,
) -> dict:
    return {
        "session_id": session_id,
        "model_bytes": model_bytes,
        "schema": schema,
        "input_record": input_record,
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


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/explain", response_model=ExplainResponse)
async def explain(body: ExplainRequest, request: Request) -> ExplainResponse:
    """Run the LangGraph anomaly explanation pipeline for a given input record."""

    # 1. Retrieve session
    session = request.app.state.sessions.get(body.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload a model first via POST /api/upload-model.",
        )

    # 2. Parse optional background CSV
    background_np = None
    if session.background_data:
        background_np = _parse_background_csv(session.background_data, session.schema)

    # 3. Build initial ExplainerState
    state = _build_initial_state(
        session_id=body.session_id,
        model_bytes=session.model_bytes,
        schema=session.schema,
        input_record=body.input_record,
        background_data=background_np,
    )

    # 4. Run compiled graph (pre-compiled at app startup)
    compiled = request.app.state.graph
    result: dict = await compiled.ainvoke(state)

    # 5. Build plot URLs from absolute file paths
    pdp_urls = [
        _path_to_plot_url(p)
        for p in (result.get("pdp_plot_paths") or [])
        if p
    ]

    return ExplainResponse(
        prediction=PredictionResult(
            label=result.get("prediction_label") or "unknown",
            anomaly_score=result.get("anomaly_score") or 0.0,
            model_type=result.get("model_type") or "",
        ),
        explanations=ExplanationsResult(
            shap_values=result.get("shap_values"),
            lime_weights=result.get("lime_weights"),
            top_features=result.get("top_features"),
            plots=PlotURLs(
                shap_plot_url=_path_to_plot_url(result.get("shap_plot_path")),
                shap_force_plot_url=_path_to_plot_url(result.get("shap_force_plot_path")),
                lime_plot_url=_path_to_plot_url(result.get("lime_plot_path")),
                pdp_plot_urls=pdp_urls,
            ),
        ),
        summary=SummaryResult(
            text=result.get("explanation_summary"),
            feature_contributions=result.get("feature_contributions"),
        ),
        errors=result.get("errors") or [],
    )
