from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from verilogic_ns_api.baselines.models import BaselineConfig, SelectionManifest, sha256_json
from verilogic_ns_api.baselines.prompts import Demonstration, PromptTemplate, load_prompt_template
from verilogic_ns_api.baselines.schema import load_schema_hash
from verilogic_ns_api.baselines.selection import (
    load_manifest,
    load_selected_examples,
    validate_demonstration_manifest,
    validate_no_selection_overlap,
    validate_pilot_manifest,
)
from verilogic_ns_api.datasets.proofwriter import ProofWriterLoader
from verilogic_ns_api.research.models import BenchmarkExample


@dataclass(frozen=True)
class PreparedBaseline:
    config: BaselineConfig
    config_hash: str
    repository_root: Path
    template: PromptTemplate
    output_schema_hash: str
    pilot_manifest: SelectionManifest
    pilot_examples: list[BenchmarkExample]
    demonstration_manifest: SelectionManifest | None
    demonstrations: list[Demonstration]


def repository_root(start: Path) -> Path:
    candidate = start.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists() and (parent / "AGENTS.md").is_file():
            return parent
    raise ValueError(f"Could not locate repository root from {start}")


def resolve_repository_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Configured path escapes the repository: {configured_path}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> BaselineConfig:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Baseline configuration must be a YAML mapping")
    return BaselineConfig.model_validate(payload)


def prepare_baseline(config_path: Path) -> PreparedBaseline:
    root = repository_root(config_path)
    config = load_config(config_path)
    config_hash = sha256_json(config)
    data_source = resolve_repository_path(root, config.dataset.data_source)
    observed_archive_hash = file_sha256(data_source)
    if observed_archive_hash != config.dataset.archive_sha256:
        raise ValueError(
            "ProofWriter archive hash mismatch: "
            f"expected {config.dataset.archive_sha256}, observed {observed_archive_hash}"
        )

    template = load_prompt_template(
        resolve_repository_path(root, config.prompt.path),
        version=config.prompt.version,
        expected_sha256=config.prompt.expected_sha256,
    )
    output_schema_hash = load_schema_hash(
        resolve_repository_path(root, config.prompt.output_schema_path)
    )
    if output_schema_hash != config.prompt.output_schema_sha256:
        raise ValueError("Structured-output schema hash does not match configuration")

    pilot_manifest = load_manifest(resolve_repository_path(root, config.dataset.selection_manifest))
    validate_pilot_manifest(pilot_manifest)
    if pilot_manifest.archive_sha256 != config.dataset.archive_sha256:
        raise ValueError("Pilot manifest archive hash does not match configuration")
    if pilot_manifest.variant != config.dataset.variant:
        raise ValueError("Pilot manifest variant does not match configuration")

    loader = ProofWriterLoader(
        data_source,
        dataset_version=config.dataset.version,
        dataset_manifest_reference=config.dataset.dataset_manifest_reference,
    )
    pilot_examples = load_selected_examples(loader, pilot_manifest)

    demonstration_manifest = None
    demonstrations: list[Demonstration] = []
    if config.prompt.demonstration_manifest is not None:
        demonstration_manifest = load_manifest(
            resolve_repository_path(root, config.prompt.demonstration_manifest)
        )
        validate_demonstration_manifest(demonstration_manifest)
        if demonstration_manifest.archive_sha256 != config.dataset.archive_sha256:
            raise ValueError("Demonstration manifest archive hash does not match configuration")
        if demonstration_manifest.variant != config.dataset.variant:
            raise ValueError("Demonstration manifest variant does not match configuration")
        validate_no_selection_overlap(demonstration_manifest, pilot_manifest)
        demonstration_examples = load_selected_examples(loader, demonstration_manifest)
        demonstrations = [
            Demonstration(example=example.for_prediction(), label=entry.label)
            for example, entry in zip(
                demonstration_examples, demonstration_manifest.entries, strict=True
            )
        ]

    return PreparedBaseline(
        config=config,
        config_hash=config_hash,
        repository_root=root,
        template=template,
        output_schema_hash=output_schema_hash,
        pilot_manifest=pilot_manifest,
        pilot_examples=pilot_examples,
        demonstration_manifest=demonstration_manifest,
        demonstrations=demonstrations,
    )
