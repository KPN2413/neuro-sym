from __future__ import annotations

import json
from pathlib import Path

from verilogic_ns_api.semantic_parsing.models import CandidateQueryOutput, CandidateTheoryOutput
from verilogic_ns_api.validation_correction.models import (
    ControllerTrace,
    QueryCriticReport,
    ReliabilityEvidence,
    TheoryCriticReport,
    ValidationFeedback,
)


def export_schemas(root: Path) -> None:
    schemas = {
        "validation-feedback.v1.schema.json": ValidationFeedback.model_json_schema(),
        "semantic-critic-theory.v1.schema.json": TheoryCriticReport.model_json_schema(),
        "semantic-critic-query.v1.schema.json": QueryCriticReport.model_json_schema(),
        "semantic-correction-theory.v1.schema.json": CandidateTheoryOutput.model_json_schema(),
        "semantic-correction-query.v1.schema.json": CandidateQueryOutput.model_json_schema(),
        "reliability-evidence.v1.schema.json": ReliabilityEvidence.model_json_schema(),
        "correction-trace.v1.schema.json": ControllerTrace.model_json_schema(),
    }
    directory = root / "schemas"
    for name, schema in schemas.items():
        (directory / name).write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
