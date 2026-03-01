import io
import pickle

import joblib
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


@pytest.fixture
def sample_schema() -> dict:
    return {
        "features": {
            "f1": {"type": "float", "min": 0.0, "max": 1.0},
            "f2": {"type": "float", "min": 0.0, "max": 1.0},
            "f3": {"type": "float", "min": 0.0, "max": 1.0},
            "f4": {"type": "float", "min": 0.0, "max": 1.0},
        }
    }


@pytest.fixture
def normal_record() -> dict:
    return {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5}


@pytest.fixture
def anomalous_record() -> dict:
    return {"f1": 10.0, "f2": 10.0, "f3": 10.0, "f4": 10.0}


def _train_isolation_forest() -> IsolationForest:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 4))
    model = IsolationForest(n_estimators=10, random_state=42)
    model.fit(X)
    return model


def _train_ocsvm() -> OneClassSVM:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 4))
    model = OneClassSVM(nu=0.05)
    model.fit(X)
    return model


def _train_lof() -> LocalOutlierFactor:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 4))
    model = LocalOutlierFactor(novelty=True)
    model.fit(X)
    return model


@pytest.fixture
def isolation_forest_bytes() -> bytes:
    buf = io.BytesIO()
    joblib.dump(_train_isolation_forest(), buf)
    return buf.getvalue()


@pytest.fixture
def ocsvm_bytes() -> bytes:
    buf = io.BytesIO()
    joblib.dump(_train_ocsvm(), buf)
    return buf.getvalue()


@pytest.fixture
def lof_bytes() -> bytes:
    buf = io.BytesIO()
    joblib.dump(_train_lof(), buf)
    return buf.getvalue()


@pytest.fixture
def pickle_isolation_forest_bytes() -> bytes:
    return pickle.dumps(_train_isolation_forest())


@pytest.fixture
def background_data_array(sample_schema) -> np.ndarray:
    """Small synthetic background dataset (50 rows) for tool-node tests."""
    from app.services.plot_generator import generate_background_data

    return generate_background_data(sample_schema, n=50)
