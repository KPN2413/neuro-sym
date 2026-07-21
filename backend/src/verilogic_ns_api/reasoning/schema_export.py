from __future__ import annotations

import json
from pathlib import Path

from verilogic_ns_api.reasoning.models import ProofDAG, ReasoningOutput


def proof_schema() -> dict[str, object]:
    return ProofDAG.model_json_schema()


def reasoning_result_schema() -> dict[str, object]:
    return ReasoningOutput.model_json_schema()


def schema_text(model: type[ProofDAG] | type[ReasoningOutput]) -> str:
    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"


def export_schemas(repository_root: Path) -> None:
    outputs = {
        repository_root / "schemas" / "proof.v1.schema.json": (
            json.dumps(proof_schema(), indent=2, sort_keys=True) + "\n"
        ),
        repository_root / "schemas" / "reasoning-result.v1.schema.json": (
            json.dumps(reasoning_result_schema(), indent=2, sort_keys=True) + "\n"
        ),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8", newline="\n")
