from __future__ import annotations

import json
import os
from pathlib import Path

from verilogic_ns_api.baselines.models import BaselineOutput, sha256_text


def baseline_output_schema() -> dict:
    schema = BaselineOutput.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://verilogic-ns.local/schemas/llm-baseline-output.v1.schema.json"
    schema["title"] = "VeriLogic-NS LLM Baseline Output v1"
    schema["description"] = "Strict three-label output contract for direct and few-shot baselines."
    return schema


def write_schema(path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(baseline_output_schema(), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_schema_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != baseline_output_schema():
        raise ValueError(f"Structured-output schema is not the generated v1 schema: {path}")
    return sha256_text(path.read_text(encoding="utf-8"))
