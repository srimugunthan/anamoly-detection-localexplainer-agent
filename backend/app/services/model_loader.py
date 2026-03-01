import io
import pickle
from typing import Any, Callable

import joblib


class ModelLoadError(Exception):
    pass


TREE_MODELS = {"IsolationForest", "RandomForestClassifier", "ExtraTreesClassifier"}


def load_model(file_bytes: bytes) -> tuple[Any, str]:
    """Load a model from raw bytes. Tries joblib first, then pickle.

    Returns (model_object, model_type_string).
    Raises ModelLoadError if neither loader succeeds.
    """
    for loader in (_load_joblib, _load_pickle):
        try:
            model = loader(file_bytes)
            return model, type(model).__name__
        except Exception:
            continue
    raise ModelLoadError(
        "Could not deserialize model. Ensure the file is a valid .pkl or .joblib artifact."
    )


def get_scorer(model: Any) -> Callable:
    """Return the best anomaly scoring method available on the model.

    Priority: score_samples > decision_function > predict
    Higher scores from score_samples/decision_function = more normal (less anomalous).
    """
    if hasattr(model, "score_samples"):
        return model.score_samples
    if hasattr(model, "decision_function"):
        return model.decision_function
    return model.predict


def _load_joblib(file_bytes: bytes) -> Any:
    return joblib.load(io.BytesIO(file_bytes))


def _load_pickle(file_bytes: bytes) -> Any:
    return pickle.loads(file_bytes)  # noqa: S301
