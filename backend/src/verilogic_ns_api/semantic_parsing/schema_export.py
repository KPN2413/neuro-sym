from __future__ import annotations

import json
from pathlib import Path

from verilogic_ns_api.semantic_parsing.models import CandidateQueryOutput, CandidateTheoryOutput


def export_parser_schemas(root: Path) -> tuple[Path, Path]:
    targets = (
        (root / "schemas/neural-theory-output.v1.schema.json", CandidateTheoryOutput),
        (root / "schemas/neural-query-output.v1.schema.json", CandidateQueryOutput),
    )
    for path, model in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return targets[0][0], targets[1][0]
