import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "theory.v1.schema.json"
VALID_FIXTURE_DIRECTORY = REPOSITORY_ROOT / "examples" / "theories"
INVALID_FIXTURE_DIRECTORY = VALID_FIXTURE_DIRECTORY / "invalid"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize("fixture_name", ["entailed.json", "contradicted.json", "unknown.json"])
def test_valid_theory_fixture_passes_schema(
    validator: Draft202012Validator, fixture_name: str
) -> None:
    instance = load_json(VALID_FIXTURE_DIRECTORY / fixture_name)

    errors = list(validator.iter_errors(instance))

    assert errors == []


@pytest.mark.parametrize(
    ("fixture_name", "expected_validator", "expected_path"),
    [
        ("wrong-predicate-arity.json", "maxItems", ("facts", 0, "arguments")),
        ("missing-source-reference.json", "required", ("facts", 0)),
        ("unsafe-identifier.json", "pattern", ("predicates", 0, "name")),
        ("unsupported-structure.json", "additionalProperties", ()),
    ],
)
def test_invalid_theory_fixture_fails_for_intended_reason(
    validator: Draft202012Validator,
    fixture_name: str,
    expected_validator: str,
    expected_path: tuple[str | int, ...],
) -> None:
    instance = load_json(INVALID_FIXTURE_DIRECTORY / fixture_name)

    errors = list(validator.iter_errors(instance))

    assert errors, f"{fixture_name} unexpectedly passed the schema"
    assert any(
        error.validator == expected_validator and tuple(error.absolute_path) == expected_path
        for error in errors
    ), [error.message for error in errors]
