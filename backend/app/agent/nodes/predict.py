import pandas as pd

from app.agent.state import ExplainerState
from app.services.model_loader import ModelLoadError, get_scorer, load_model
from app.services.schema_validator import FieldValidationError, validate_record

# Outlier-detection models (IsolationForest, OCSVM, …) use the -1/+1 convention:
#   predict() → -1 = anomaly, +1 = normal
# Classifiers (RandomForest, ExtraTrees, …) use the 0/1 convention:
#   predict() →  0 = normal,  1 = anomaly
_OUTLIER_MODELS = {"IsolationForest", "LocalOutlierFactor", "EllipticEnvelope", "OneClassSVM"}


async def predict_node(state: ExplainerState) -> dict:
    """LangGraph node: load model, validate record, predict label and anomaly score.

    Reads from state: model_bytes, schema, input_record
    Writes to state:  model, model_type, input_df, prediction_label, anomaly_score, errors
    """
    errors: list[str] = list(state.get("errors", []))

    # 1. Load model
    try:
        model, model_type = load_model(state["model_bytes"])
    except ModelLoadError as exc:
        errors.append(f"model_load: {exc}")
        return {"errors": errors}

    # 2. Validate input record
    try:
        validated = validate_record(state["input_record"], state["schema"])
    except FieldValidationError as exc:
        errors.append(f"validation: {exc}")
        return {"errors": errors}

    # 3. Convert to single-row DataFrame
    feature_names = list(state["schema"]["features"].keys())
    input_df = pd.DataFrame([validated], columns=feature_names)

    # 4. Predict label — convention differs by model family
    raw_label = int(model.predict(input_df)[0])
    if model_type in _OUTLIER_MODELS:
        # -1 = anomaly, +1 = normal
        prediction_label = "anomaly" if raw_label == -1 else "normal"
    else:
        # Classifier convention: 0 = normal, any non-zero = anomaly
        prediction_label = "anomaly" if raw_label != 0 else "normal"

    # 5. Compute anomaly score — prefer a continuous score over a hard class label
    #    score_samples / decision_function: higher = more normal (outlier models)
    #    predict_proba[:, 1]: probability of the anomaly class (classifiers)
    if hasattr(model, "score_samples"):
        anomaly_score = float(model.score_samples(input_df)[0])
    elif hasattr(model, "decision_function"):
        anomaly_score = float(model.decision_function(input_df)[0])
    elif hasattr(model, "predict_proba"):
        anomaly_score = float(model.predict_proba(input_df)[0, 1])
    else:
        anomaly_score = float(model.predict(input_df)[0])

    return {
        "model": model,
        "model_type": model_type,
        "input_df": input_df,
        "prediction_label": prediction_label,
        "anomaly_score": anomaly_score,
        "errors": errors,
    }
