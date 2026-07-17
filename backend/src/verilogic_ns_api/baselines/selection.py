from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from verilogic_ns_api.baselines.models import (
    SelectionEntry,
    SelectionManifest,
    sha256_json,
)
from verilogic_ns_api.datasets.proofwriter import ProofWriterLoader
from verilogic_ns_api.research.models import BenchmarkExample, GoldLabel, Split

PILOT_DEPTHS = (0, 1, 2, 3, 5)
PILOT_PER_CELL = 2
PILOT_SIZE = len(PILOT_DEPTHS) * len(GoldLabel) * PILOT_PER_CELL
DEMONSTRATION_SIZE = 6
DEFAULT_SELECTION_SEED = 20260713
SAMPLER_VERSION = "phase3-v1"


class SelectionError(ValueError):
    pass


def _order_key(example: BenchmarkExample, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{example.example_id}".encode()).hexdigest()
    return digest, example.example_id


def select_pilot_examples(
    examples: Iterable[BenchmarkExample], *, seed: int = DEFAULT_SELECTION_SEED
) -> list[BenchmarkExample]:
    groups: dict[tuple[int, GoldLabel], list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        if example.split is not Split.DEVELOPMENT:
            raise SelectionError("Pilot selection accepts development examples only")
        if example.reasoning_depth in PILOT_DEPTHS:
            groups[(example.reasoning_depth, example.gold_label)].append(example)

    unavailable = {
        f"depth={depth},label={label.value}": PILOT_PER_CELL - len(groups[(depth, label)])
        for depth in PILOT_DEPTHS
        for label in GoldLabel
        if len(groups[(depth, label)]) < PILOT_PER_CELL
    }
    if unavailable:
        raise SelectionError(f"Pilot matrix has unavailable cells: {unavailable}")

    selected = []
    for depth in PILOT_DEPTHS:
        for label in GoldLabel:
            selected.extend(
                sorted(groups[(depth, label)], key=lambda item: _order_key(item, seed))[
                    :PILOT_PER_CELL
                ]
            )
    if len({example.provenance.content_sha256 for example in selected}) != PILOT_SIZE:
        raise SelectionError("Pilot selection contains duplicate normalized content")
    return sorted(selected, key=lambda item: _order_key(item, seed))


def select_demonstrations(
    examples: Iterable[BenchmarkExample], *, seed: int = DEFAULT_SELECTION_SEED
) -> list[BenchmarkExample]:
    groups: dict[GoldLabel, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        if example.split is not Split.TRAIN:
            raise SelectionError("Demonstration selection accepts training examples only")
        groups[example.gold_label].append(example)

    selected: list[BenchmarkExample] = []
    unavailable: list[str] = []
    for label in GoldLabel:
        shallow = sorted(
            (item for item in groups[label] if item.reasoning_depth == 0),
            key=lambda item: _order_key(item, seed),
        )
        nontrivial = sorted(
            (item for item in groups[label] if (item.reasoning_depth or 0) >= 2),
            key=lambda item: _order_key(item, seed),
        )
        if not shallow or not nontrivial:
            unavailable.append(label.value)
            continue
        selected.extend([shallow[0], nontrivial[0]])
    if unavailable:
        raise SelectionError(
            "Could not select one shallow and one nontrivial training demonstration for: "
            + ", ".join(unavailable)
        )
    if len({item.example_id for item in selected}) != DEMONSTRATION_SIZE:
        raise SelectionError("Demonstration selection contains duplicate IDs")
    if len({item.provenance.content_sha256 for item in selected}) != DEMONSTRATION_SIZE:
        raise SelectionError("Demonstration selection contains duplicate normalized content")
    return sorted(selected, key=lambda item: _order_key(item, seed))


def create_selection_manifest(
    *,
    selection_kind: str,
    archive_sha256: str,
    variant: str,
    split: Split,
    seed: int,
    examples: list[BenchmarkExample],
) -> SelectionManifest:
    payload = {
        "schema_version": "1.0",
        "selection_kind": selection_kind,
        "dataset_name": "ProofWriter",
        "dataset_version": "V2020.12.3",
        "archive_sha256": archive_sha256,
        "world_assumption": "OWA",
        "variant": variant,
        "split": split.value,
        "seed": seed,
        "sampler_version": SAMPLER_VERSION,
        "entries": [
            SelectionEntry(
                example_id=example.example_id,
                content_sha256=example.provenance.content_sha256,
                reasoning_depth=example.reasoning_depth,
                label=example.gold_label,
                split=example.split,
            ).model_dump(mode="json")
            for example in examples
        ],
    }
    payload["manifest_hash"] = sha256_json(payload)
    return SelectionManifest.model_validate(payload)


def write_manifest(path: Path, manifest: SelectionManifest) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest.model_dump(mode="json"), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_manifest(path: Path) -> SelectionManifest:
    return SelectionManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_selected_examples(
    loader: ProofWriterLoader, manifest: SelectionManifest
) -> list[BenchmarkExample]:
    expected = {entry.example_id: entry for entry in manifest.entries}
    observed: dict[str, BenchmarkExample] = {}
    for example in loader.iter_examples(variant=manifest.variant, split=manifest.split):
        entry = expected.get(example.example_id)
        if entry is None:
            continue
        if example.provenance.content_sha256 != entry.content_sha256:
            raise SelectionError(f"Content hash mismatch for {example.example_id}")
        if example.reasoning_depth != entry.reasoning_depth:
            raise SelectionError(f"Depth mismatch for {example.example_id}")
        if example.gold_label is not entry.label:
            raise SelectionError(f"Label mismatch for {example.example_id}")
        observed[example.example_id] = example
        if len(observed) == len(expected):
            break
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise SelectionError(f"Selection manifest IDs not found: {missing}")
    return [observed[entry.example_id] for entry in manifest.entries]


def validate_pilot_manifest(manifest: SelectionManifest) -> None:
    if manifest.selection_kind != "pilot" or manifest.split is not Split.DEVELOPMENT:
        raise SelectionError("Pilot manifest must contain development examples")
    if len(manifest.entries) != PILOT_SIZE:
        raise SelectionError(f"Pilot manifest must contain exactly {PILOT_SIZE} examples")
    distribution = Counter((entry.reasoning_depth, entry.label) for entry in manifest.entries)
    expected = {(depth, label): PILOT_PER_CELL for depth in PILOT_DEPTHS for label in GoldLabel}
    if distribution != expected:
        raise SelectionError(f"Pilot distribution mismatch: {dict(distribution)}")


def validate_demonstration_manifest(manifest: SelectionManifest) -> None:
    if manifest.selection_kind != "demonstrations" or manifest.split is not Split.TRAIN:
        raise SelectionError("Demonstration manifest must contain training examples")
    if len(manifest.entries) != DEMONSTRATION_SIZE:
        raise SelectionError(
            f"Demonstration manifest must contain exactly {DEMONSTRATION_SIZE} examples"
        )
    labels = Counter(entry.label for entry in manifest.entries)
    if labels != Counter({label: 2 for label in GoldLabel}):
        raise SelectionError("Demonstrations must contain exactly two examples per label")
    for label in GoldLabel:
        depths = [entry.reasoning_depth for entry in manifest.entries if entry.label is label]
        if 0 not in depths or not any(depth >= 2 for depth in depths):
            raise SelectionError(
                f"Demonstrations for {label.value} require shallow and nontrivial depths"
            )


def validate_no_selection_overlap(
    demonstrations: SelectionManifest, pilot: SelectionManifest
) -> None:
    demo_ids = {entry.example_id for entry in demonstrations.entries}
    pilot_ids = {entry.example_id for entry in pilot.entries}
    demo_hashes = {entry.content_sha256 for entry in demonstrations.entries}
    pilot_hashes = {entry.content_sha256 for entry in pilot.entries}
    if demo_ids & pilot_ids:
        raise SelectionError("Training demonstrations overlap pilot example IDs")
    if demo_hashes & pilot_hashes:
        raise SelectionError("Training demonstrations overlap pilot normalized content")
