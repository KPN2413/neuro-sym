from __future__ import annotations

import argparse
import json
from pathlib import Path

from verilogic_ns_api.research.models import BenchmarkExample


def normalized_example_schema() -> dict:
    schema = BenchmarkExample.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://verilogic-ns.local/schemas/benchmark-example.v1.schema.json"
    schema["title"] = "VeriLogic-NS Normalized Benchmark Example v1"
    schema["description"] = (
        "Versioned, model-independent normalized example contract for ProofWriter evaluation."
    )
    return schema


def write_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(normalized_example_schema(), stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the normalized benchmark JSON Schema")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_schema(args.output)


if __name__ == "__main__":
    main()
