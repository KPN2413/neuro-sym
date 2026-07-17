import json
from pathlib import Path

from jsonschema import Draft202012Validator

from verilogic_ns_api.datasets.proofwriter import ProofWriterLoader
from verilogic_ns_api.research.models import Split
from verilogic_ns_api.research.schema_export import normalized_example_schema

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "benchmark-example.v1.schema.json"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "proofwriter" / "proofwriter-dataset-V2020.12.3"


def test_normalized_schema_is_valid_and_matches_typed_model() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    assert schema == normalized_example_schema()


def test_normalized_synthetic_examples_validate_against_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    examples = ProofWriterLoader(FIXTURE_ROOT).iter_examples(variant="depth-1", split=Split.TRAIN)

    for example in examples:
        errors = list(validator.iter_errors(example.model_dump(mode="json")))
        assert errors == []
