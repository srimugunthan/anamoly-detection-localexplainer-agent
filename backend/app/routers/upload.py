import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import MAX_MODEL_SIZE_BYTES, PLOT_DIR
from app.services.model_loader import ModelLoadError, load_model
from app.services.schema_validator import SchemaValidationError, validate_schema

router = APIRouter(prefix="/api", tags=["upload"])


class SessionData(BaseModel):
    model_bytes: bytes
    schema: dict
    background_data: bytes | None = None
    created_at: datetime
    model_type: str

    model_config = {"arbitrary_types_allowed": True}


class UploadResponse(BaseModel):
    session_id: str
    model_type: str
    feature_count: int
    status: str


@router.post("/upload-model", response_model=UploadResponse)
async def upload_model(
    request: Request,
    model_file: UploadFile = File(..., description="Serialized model (.pkl or .joblib)"),
    schema: str = Form(..., description="JSON schema string describing features"),
    reference_csv: UploadFile | None = File(None, description="Optional reference/background CSV"),
) -> UploadResponse:
    """Accept a serialized model + JSON schema. Returns a session_id for subsequent /explain calls."""

    # --- Validate file size ---
    model_bytes = await model_file.read()
    if len(model_bytes) > MAX_MODEL_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Model file exceeds the {MAX_MODEL_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    # --- Parse and validate schema ---
    try:
        schema_dict = json.loads(schema)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON schema: {exc}") from exc

    try:
        validate_schema(schema_dict)
    except SchemaValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # --- Load model to detect type and confirm it's valid ---
    try:
        _, model_type = load_model(model_bytes)
    except ModelLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # --- Optional reference CSV ---
    background_bytes: bytes | None = None
    if reference_csv is not None:
        background_bytes = await reference_csv.read()

    # --- Store session ---
    session_id = str(uuid.uuid4())
    request.app.state.sessions[session_id] = SessionData(
        model_bytes=model_bytes,
        schema=schema_dict,
        background_data=background_bytes,
        created_at=datetime.now(tz=timezone.utc),
        model_type=model_type,
    )

    feature_count = len(schema_dict["features"])
    return UploadResponse(
        session_id=session_id,
        model_type=model_type,
        feature_count=feature_count,
        status="ready",
    )


@router.get("/plot/{plot_id}", name="serve_plot")
async def serve_plot(plot_id: str) -> FileResponse:
    """Serve a generated PNG plot by its ID (filename stem without the .png extension)."""
    # Reject any path traversal attempts
    if any(c in plot_id for c in ("/", "\\", "..", "\x00")):
        raise HTTPException(status_code=400, detail="Invalid plot_id.")
    path = PLOT_DIR / f"{plot_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(str(path), media_type="image/png")
