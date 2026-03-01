from typing import Any


class FieldValidationError(Exception):
    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class SchemaValidationError(Exception):
    pass


def validate_schema(schema: dict) -> None:
    """Check that the schema has the expected top-level structure.

    Expected format:
        {
          "features": {
            "feature_name": {
              "type": "float" | "int" | "categorical",
              "min": <number>,        # required for float/int
              "max": <number>,        # required for float/int
              "values": [<str>, ...]  # required for categorical
            }
          }
        }
    """
    if "features" not in schema or not isinstance(schema["features"], dict):
        raise SchemaValidationError("Schema must have a 'features' dict at the top level.")
    if not schema["features"]:
        raise SchemaValidationError("Schema 'features' must not be empty.")
    for name, spec in schema["features"].items():
        ftype = spec.get("type")
        if ftype not in ("float", "int", "categorical"):
            raise SchemaValidationError(
                f"Feature '{name}': type must be 'float', 'int', or 'categorical', got '{ftype}'."
            )
        if ftype == "categorical" and "values" not in spec:
            raise SchemaValidationError(
                f"Feature '{name}': categorical feature must have a 'values' list."
            )


def validate_record(record: dict, schema: dict) -> dict[str, Any]:
    """Validate and coerce a raw input record against the schema.

    - float/int fields: coerced to the declared numeric type.
    - categorical fields: value must be in spec['values']; encoded as its list index.

    Returns a dict ready for pd.DataFrame construction.
    Raises FieldValidationError on the first invalid field.
    """
    features = schema.get("features", {})
    validated: dict[str, Any] = {}

    for name, spec in features.items():
        if name not in record:
            raise FieldValidationError(name, "missing required field")

        raw = record[name]
        ftype = spec.get("type", "float")

        if ftype == "float":
            try:
                validated[name] = float(raw)
            except (ValueError, TypeError):
                raise FieldValidationError(name, f"expected float, got {type(raw).__name__!r}")

        elif ftype == "int":
            try:
                validated[name] = int(raw)
            except (ValueError, TypeError):
                raise FieldValidationError(name, f"expected int, got {type(raw).__name__!r}")

        elif ftype == "categorical":
            allowed = spec.get("values", [])
            if raw not in allowed:
                raise FieldValidationError(
                    name, f"invalid value {raw!r}, must be one of {allowed}"
                )
            validated[name] = allowed.index(raw)

    return validated
