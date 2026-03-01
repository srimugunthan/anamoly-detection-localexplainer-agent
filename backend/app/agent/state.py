import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class ExplainerState(TypedDict):
    # --- Inputs ---
    session_id: str
    model_bytes: bytes
    schema: dict
    input_record: dict
    background_data: Optional[Any]      # np.ndarray or None

    # --- Loaded objects (not serialized across HTTP) ---
    model: Optional[Any]
    model_type: str

    # --- Prediction ---
    prediction_label: str               # "anomaly" | "normal"
    anomaly_score: float
    input_df: Optional[Any]             # pd.DataFrame, single row

    # --- Explainability outputs ---
    shap_values: Optional[dict]
    shap_plot_path: Optional[str]
    shap_force_plot_path: Optional[str]
    lime_weights: Optional[dict]
    lime_plot_path: Optional[str]
    pdp_plot_paths: Optional[list[str]]
    top_features: Optional[list[str]]

    # --- LLM interpretation ---
    explanation_summary: Optional[str]
    feature_contributions: Optional[list[dict]]

    # --- Errors (non-fatal, accumulated across nodes) ---
    errors: Annotated[list[str], operator.add]
