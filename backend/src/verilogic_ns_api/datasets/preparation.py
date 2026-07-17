from __future__ import annotations

import json
import os
import shutil
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from verilogic_ns_api.datasets.errors import ExistingDataError, SamplingError
from verilogic_ns_api.datasets.proofwriter import LoaderStats, ProofWriterLoader
from verilogic_ns_api.research.models import Split


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare_proofwriter(
    *,
    data_source: Path,
    output_root: Path,
    variant: str,
    splits: list[Split],
    allow_test: bool = False,
    max_examples_per_split: int | None = None,
    force: bool = False,
    dataset_manifest_reference: str | None = None,
) -> Path:
    if Split.TEST in splits and not allow_test:
        raise SamplingError("Preparing the test split requires allow_test=true")
    loader = ProofWriterLoader(
        data_source,
        dataset_manifest_reference=dataset_manifest_reference,
    )
    output_root = output_root.resolve()
    target = (output_root / loader.dataset_version / "OWA" / variant).resolve()
    if not target.is_relative_to(output_root) or target == output_root:
        raise ValueError("Preparation target escapes the configured output root")
    if target.exists():
        if not force:
            raise ExistingDataError(f"Prepared output already exists: {target}")
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.preparing-{uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    counts: dict[str, int] = {}
    stats_by_split: dict[str, Any] = {}
    try:
        for split in splits:
            stats = LoaderStats()
            output_path = temporary / f"normalized-{split.value}.jsonl"
            count = 0
            with output_path.open("w", encoding="utf-8", newline="\n") as stream:
                for example in loader.iter_examples(
                    variant=variant,
                    split=split,
                    stats=stats,
                    max_examples=max_examples_per_split,
                ):
                    stream.write(example.model_dump_json())
                    stream.write("\n")
                    count += 1
                stream.flush()
                os.fsync(stream.fileno())
            counts[split.value] = count
            stats_by_split[split.value] = stats.as_dict()

        manifest = {
            "schema_version": "1.0",
            "dataset_name": "ProofWriter",
            "dataset_version": loader.dataset_version,
            "world_assumption": "OWA",
            "variant": variant,
            "splits": [split.value for split in splits],
            "example_counts": counts,
            "max_examples_per_split": max_examples_per_split,
            "partial": max_examples_per_split is not None,
            "source": str(data_source.resolve()),
            "dataset_manifest_reference": dataset_manifest_reference,
            "prepared_at": datetime.now(UTC).isoformat(),
            "loader_stats": stats_by_split,
        }
        _write_json(temporary / "preparation-manifest.json", manifest)
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        with suppress(FileNotFoundError):
            temporary.rmdir()
    return target
