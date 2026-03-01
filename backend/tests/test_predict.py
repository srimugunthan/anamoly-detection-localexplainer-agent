import pytest

from app.agent.nodes.predict import predict_node
from app.services.model_loader import ModelLoadError, get_scorer, load_model


# ---------------------------------------------------------------------------
# model_loader
# ---------------------------------------------------------------------------

def test_load_joblib_isolation_forest(isolation_forest_bytes):
    model, model_type = load_model(isolation_forest_bytes)
    assert model_type == "IsolationForest"
    assert hasattr(model, "predict")


def test_load_joblib_ocsvm(ocsvm_bytes):
    _, model_type = load_model(ocsvm_bytes)
    assert model_type == "OneClassSVM"


def test_load_joblib_lof(lof_bytes):
    _, model_type = load_model(lof_bytes)
    assert model_type == "LocalOutlierFactor"


def test_load_pickle_fallback(pickle_isolation_forest_bytes):
    model, model_type = load_model(pickle_isolation_forest_bytes)
    assert model_type == "IsolationForest"


def test_load_invalid_bytes():
    with pytest.raises(ModelLoadError):
        load_model(b"not a model")


def test_get_scorer_prefers_score_samples(isolation_forest_bytes):
    model, _ = load_model(isolation_forest_bytes)
    scorer = get_scorer(model)
    # IsolationForest has score_samples
    assert scorer == model.score_samples


def test_get_scorer_decision_function_fallback():
    # Mock a model that only has decision_function (no score_samples)
    class _FakeModel:
        def decision_function(self, X):
            return [0.0]

        def predict(self, X):
            return [1]

    model = _FakeModel()
    scorer = get_scorer(model)
    assert scorer == model.decision_function


# ---------------------------------------------------------------------------
# predict_node
# ---------------------------------------------------------------------------

def _make_state(model_bytes, schema, record, errors=None):
    return {
        "session_id": "test-session",
        "model_bytes": model_bytes,
        "schema": schema,
        "input_record": record,
        "background_data": None,
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
        "errors": errors or [],
    }


@pytest.mark.asyncio
async def test_predict_node_isolation_forest_normal(
    isolation_forest_bytes, sample_schema, normal_record
):
    state = _make_state(isolation_forest_bytes, sample_schema, normal_record)
    result = await predict_node(state)

    assert result["model_type"] == "IsolationForest"
    assert result["prediction_label"] in ("anomaly", "normal")
    assert isinstance(result["anomaly_score"], float)
    assert result["input_df"] is not None
    assert list(result["input_df"].columns) == ["f1", "f2", "f3", "f4"]
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_predict_node_isolation_forest_anomalous(
    isolation_forest_bytes, sample_schema, anomalous_record
):
    state = _make_state(isolation_forest_bytes, sample_schema, anomalous_record)
    result = await predict_node(state)
    # Out-of-distribution record should be flagged as anomaly
    assert result["prediction_label"] == "anomaly"


@pytest.mark.asyncio
async def test_predict_node_ocsvm(ocsvm_bytes, sample_schema, normal_record):
    state = _make_state(ocsvm_bytes, sample_schema, normal_record)
    result = await predict_node(state)
    assert result["model_type"] == "OneClassSVM"
    assert result["prediction_label"] in ("anomaly", "normal")
    assert isinstance(result["anomaly_score"], float)


@pytest.mark.asyncio
async def test_predict_node_lof(lof_bytes, sample_schema, normal_record):
    state = _make_state(lof_bytes, sample_schema, normal_record)
    result = await predict_node(state)
    assert result["model_type"] == "LocalOutlierFactor"
    assert result["prediction_label"] in ("anomaly", "normal")


@pytest.mark.asyncio
async def test_predict_node_invalid_model_bytes(sample_schema, normal_record):
    state = _make_state(b"garbage", sample_schema, normal_record)
    result = await predict_node(state)
    assert any("model_load" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_predict_node_missing_field(isolation_forest_bytes, sample_schema):
    incomplete_record = {"f1": 0.5, "f2": 0.5}  # missing f3, f4
    state = _make_state(isolation_forest_bytes, sample_schema, incomplete_record)
    result = await predict_node(state)
    assert any("validation" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_predict_node_invalid_field_type(isolation_forest_bytes, sample_schema):
    bad_record = {"f1": "not-a-float", "f2": 0.5, "f3": 0.5, "f4": 0.5}
    state = _make_state(isolation_forest_bytes, sample_schema, bad_record)
    result = await predict_node(state)
    assert any("validation" in e for e in result["errors"])
