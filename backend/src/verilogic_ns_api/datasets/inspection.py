from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from verilogic_ns_api.datasets.acquisition import sha256_file
from verilogic_ns_api.datasets.proofwriter import LoaderStats, ProofWriterLoader
from verilogic_ns_api.datasets.sampling import LeakageAccumulator
from verilogic_ns_api.research.models import Split, WorldAssumption


def inspect_proofwriter(
    data_source: Path,
    *,
    variants: list[str] | None = None,
    splits: list[Split] | None = None,
    max_examples_per_split: int | None = None,
) -> dict[str, Any]:
    loader = ProofWriterLoader(data_source)
    layout = loader.layout
    available_owa = layout.files.get(WorldAssumption.OPEN, {})
    selected_variants = variants or sorted(available_owa)
    selected_splits = splits or [Split.TRAIN, Split.DEVELOPMENT, Split.TEST]

    label_distribution: Counter[str] = Counter()
    depth_distribution: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    file_reports: dict[str, Any] = {}
    leakage = LeakageAccumulator()
    total_examples = 0
    invalid_records = 0
    invalid_questions = 0
    duplicate_ids = 0
    inspected_variants: set[str] = set()
    inspected_splits: set[Split] = set()

    for variant in selected_variants:
        if variant not in available_owa:
            raise ValueError(f"Unknown OWA variant {variant!r}")
        for split in selected_splits:
            if split not in available_owa[variant]:
                continue
            inspected_variants.add(variant)
            inspected_splits.add(split)
            stats = LoaderStats()
            for example in loader.iter_examples(
                variant=variant,
                split=split,
                strict=False,
                stats=stats,
                max_examples=max_examples_per_split,
            ):
                total_examples += 1
                label_distribution[example.gold_label.value] += 1
                depth_key = (
                    str(example.reasoning_depth)
                    if example.reasoning_depth is not None
                    else "missing"
                )
                depth_distribution[depth_key] += 1
                leakage.add(example)
            invalid_records += stats.invalid_records
            invalid_questions += stats.invalid_questions
            duplicate_ids += stats.duplicate_ids
            missing_fields.update(stats.missing_fields)
            file_reports[f"{variant}/{split.value}"] = stats.as_dict()

    overlap_report = leakage.report()
    source_provenance = {
        "source_type": "directory" if data_source.is_dir() else "zip",
        "sha256": None if data_source.is_dir() else sha256_file(data_source),
        "checksum_status": "not-applicable" if data_source.is_dir() else "locally-observed",
        "publisher_verified_checksum": False,
    }
    return {
        "schema_version": "1.0",
        "dataset_name": "ProofWriter",
        "dataset_version": loader.dataset_version,
        "data_source": str(data_source.resolve()),
        "source_provenance": source_provenance,
        "available_layout": layout.serializable(),
        "inspected_world_assumption": WorldAssumption.OPEN.value,
        "inspected_variants": sorted(inspected_variants),
        "inspected_splits": sorted(split.value for split in inspected_splits),
        "partial": max_examples_per_split is not None,
        "max_examples_per_split": max_examples_per_split,
        "example_count": total_examples,
        "label_distribution": dict(sorted(label_distribution.items())),
        "reasoning_depth_distribution": dict(
            sorted(depth_distribution.items(), key=lambda item: item[0])
        ),
        "missing_field_counts": dict(sorted(missing_fields.items())),
        "invalid_record_count": invalid_records,
        "invalid_question_count": invalid_questions,
        "duplicate_id_count": duplicate_ids,
        "overlap_report": overlap_report.model_dump(mode="json"),
        "file_reports": file_reports,
        "copyrighted_examples_included": False,
    }
