from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from verilogic_ns_api.baselines.models import PairedComparison
from verilogic_ns_api.baselines.selection import load_manifest, validate_pilot_manifest
from verilogic_ns_api.research.models import (
    MetricReport,
    PredictionLabel,
    PredictionRecord,
    RunManifest,
)


def _load_predictions(path: Path) -> list[PredictionRecord]:
    return [
        PredictionRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fairness_settings(configuration: dict) -> dict:
    return {
        key: configuration.get(key)
        for key in ("provider", "dataset", "pricing", "seed", "predictor_version")
    }


def compare_runs(
    direct_directory: Path,
    few_shot_directory: Path,
    selection_manifest_path: Path,
) -> PairedComparison:
    direct_manifest = RunManifest.model_validate_json(
        (direct_directory / "run-manifest.json").read_text(encoding="utf-8")
    )
    few_manifest = RunManifest.model_validate_json(
        (few_shot_directory / "run-manifest.json").read_text(encoding="utf-8")
    )
    if direct_manifest.predictor_name != "direct-llm":
        raise ValueError("First run is not the direct LLM baseline")
    if few_manifest.predictor_name != "few-shot-llm":
        raise ValueError("Second run is not the few-shot LLM baseline")
    if _fairness_settings(direct_manifest.configuration) != _fairness_settings(
        few_manifest.configuration
    ):
        raise ValueError("Direct and few-shot model, dataset, pricing, or seed settings differ")

    direct_metrics = MetricReport.model_validate_json(
        (direct_directory / "metrics.json").read_text(encoding="utf-8")
    )
    few_metrics = MetricReport.model_validate_json(
        (few_shot_directory / "metrics.json").read_text(encoding="utf-8")
    )
    direct = _load_predictions(direct_directory / "predictions.jsonl")
    few = _load_predictions(few_shot_directory / "predictions.jsonl")
    if [item.example_id for item in direct] != [item.example_id for item in few]:
        raise ValueError("Direct and few-shot runs do not use identical ordered example IDs")

    selection = load_manifest(selection_manifest_path)
    validate_pilot_manifest(selection)
    expected_ids = [entry.example_id for entry in selection.entries]
    if [item.example_id for item in direct] != expected_ids:
        raise ValueError("Run predictions do not match the frozen pilot manifest order")
    gold = {entry.example_id: entry.label.value for entry in selection.entries}
    depths = {entry.example_id: entry.reasoning_depth for entry in selection.entries}

    outcomes = defaultdict(int)
    disagreement: dict[str, dict[str, int]] = {
        label.value: {other.value: 0 for other in PredictionLabel} for label in PredictionLabel
    }
    for direct_item, few_item in zip(direct, few, strict=True):
        direct_correct = direct_item.predicted_label.value == gold[direct_item.example_id]
        few_correct = few_item.predicted_label.value == gold[few_item.example_id]
        if direct_correct and few_correct:
            outcomes["both_correct"] += 1
        elif direct_correct:
            outcomes["direct_only_correct"] += 1
        elif few_correct:
            outcomes["few_shot_only_correct"] += 1
        else:
            outcomes["both_incorrect"] += 1
        disagreement[direct_item.predicted_label.value][few_item.predicted_label.value] += 1

    all_depths = sorted(set(depths.values()))
    all_labels = sorted(set(direct_metrics.per_label_metrics) | set(few_metrics.per_label_metrics))
    return PairedComparison(
        direct_run_id=direct_manifest.run_id,
        few_shot_run_id=few_manifest.run_id,
        example_count=len(direct),
        accuracy_delta=few_metrics.accuracy - direct_metrics.accuracy,
        coverage_delta=few_metrics.coverage - direct_metrics.coverage,
        per_depth_accuracy_delta={
            str(depth): (
                few_metrics.per_depth_metrics[str(depth)].accuracy
                - direct_metrics.per_depth_metrics[str(depth)].accuracy
            )
            for depth in all_depths
        },
        per_label_f1_delta={
            label: (
                few_metrics.per_label_metrics[label].f1 - direct_metrics.per_label_metrics[label].f1
            )
            for label in all_labels
        },
        both_correct=outcomes["both_correct"],
        direct_only_correct=outcomes["direct_only_correct"],
        few_shot_only_correct=outcomes["few_shot_only_correct"],
        both_incorrect=outcomes["both_incorrect"],
        prediction_disagreement_matrix=disagreement,
    )


def write_comparison(path: Path, comparison: PairedComparison) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(comparison.model_dump(mode="json"), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
