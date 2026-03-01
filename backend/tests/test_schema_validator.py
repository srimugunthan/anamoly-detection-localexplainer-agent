import pytest

from app.services.schema_validator import (
    FieldValidationError,
    SchemaValidationError,
    validate_record,
    validate_schema,
)


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

def test_validate_schema_valid(sample_schema):
    validate_schema(sample_schema)  # should not raise


def test_validate_schema_missing_features_key():
    with pytest.raises(SchemaValidationError):
        validate_schema({})


def test_validate_schema_empty_features():
    with pytest.raises(SchemaValidationError):
        validate_schema({"features": {}})


def test_validate_schema_bad_type():
    with pytest.raises(SchemaValidationError):
        validate_schema({"features": {"x": {"type": "unknown"}}})


def test_validate_schema_categorical_missing_values():
    with pytest.raises(SchemaValidationError):
        validate_schema({"features": {"x": {"type": "categorical"}}})


# ---------------------------------------------------------------------------
# validate_record
# ---------------------------------------------------------------------------

def test_validate_record_valid_floats(sample_schema, normal_record):
    result = validate_record(normal_record, sample_schema)
    assert result == {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5}


def test_validate_record_coerces_string_float(sample_schema):
    record = {"f1": "0.3", "f2": "0.4", "f3": "0.5", "f4": "0.6"}
    result = validate_record(record, sample_schema)
    assert result["f1"] == pytest.approx(0.3)


def test_validate_record_missing_field(sample_schema):
    record = {"f1": 0.1, "f2": 0.2}  # missing f3, f4
    with pytest.raises(FieldValidationError) as exc_info:
        validate_record(record, sample_schema)
    assert exc_info.value.field in ("f3", "f4")


def test_validate_record_bad_float_type(sample_schema):
    record = {"f1": "not-a-number", "f2": 0.2, "f3": 0.3, "f4": 0.4}
    with pytest.raises(FieldValidationError) as exc_info:
        validate_record(record, sample_schema)
    assert exc_info.value.field == "f1"


def test_validate_record_categorical_valid():
    schema = {
        "features": {
            "color": {"type": "categorical", "values": ["red", "green", "blue"]},
        }
    }
    result = validate_record({"color": "green"}, schema)
    assert result["color"] == 1  # index of "green"


def test_validate_record_categorical_invalid():
    schema = {
        "features": {
            "color": {"type": "categorical", "values": ["red", "green", "blue"]},
        }
    }
    with pytest.raises(FieldValidationError) as exc_info:
        validate_record({"color": "purple"}, schema)
    assert exc_info.value.field == "color"


def test_validate_record_int_field():
    schema = {"features": {"count": {"type": "int", "min": 0, "max": 100}}}
    result = validate_record({"count": 42}, schema)
    assert result["count"] == 42
    assert isinstance(result["count"], int)
