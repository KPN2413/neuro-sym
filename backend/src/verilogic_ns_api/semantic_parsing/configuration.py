from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from verilogic_ns_api.baselines.configuration import (
    file_sha256,
    repository_root,
    resolve_repository_path,
)
from verilogic_ns_api.baselines.selection import load_manifest, load_selected_examples
from verilogic_ns_api.datasets.proofwriter import ProofWriterLoader
from verilogic_ns_api.reasoning.models import sha256_payload
from verilogic_ns_api.research.models import BenchmarkExample, Split
from verilogic_ns_api.semantic_parsing.models import (
    CandidateQueryOutput,
    CandidateTheoryOutput,
    ParserExperimentConfig,
)
from verilogic_ns_api.semantic_parsing.prompts import PromptRegistry


@dataclass(frozen=True)
class PreparedParserExperiment:
    config: ParserExperimentConfig
    config_path: Path
    root: Path
    theory_prompt: str
    query_prompt: str
    pilot_examples: tuple[BenchmarkExample, ...]
    calibration_examples: tuple[BenchmarkExample, ...]


def load_parser_config(path: Path) -> ParserExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic-parser config must be a YAML mapping")
    return ParserExperimentConfig.model_validate(payload)


def prepare_parser_experiment(path: Path) -> PreparedParserExperiment:
    root = repository_root(path)
    config = load_parser_config(path)
    archive = resolve_repository_path(root, config.data_source)
    if file_sha256(archive) != config.archive_sha256:
        raise ValueError("ProofWriter archive hash differs from parser config")

    registry = PromptRegistry(root)
    theory_prompt, theory_hash = registry.load(config.theory_prompt)
    query_prompt, query_hash = registry.load(config.query_prompt)
    if theory_hash != config.theory_prompt_sha256 or query_hash != config.query_prompt_sha256:
        raise ValueError("semantic-parser prompt hash mismatch")
    if sha256_payload(CandidateTheoryOutput.model_json_schema()) != config.theory_schema_sha256:
        raise ValueError("theory structured-output schema hash mismatch")
    if sha256_payload(CandidateQueryOutput.model_json_schema()) != config.query_schema_sha256:
        raise ValueError("query structured-output schema hash mismatch")

    pilot_path = resolve_repository_path(root, config.pilot_manifest)
    calibration_path = resolve_repository_path(root, config.calibration_manifest)
    if file_sha256(pilot_path) != config.pilot_manifest_sha256:
        raise ValueError("pilot manifest file hash mismatch")
    if file_sha256(calibration_path) != config.calibration_manifest_sha256:
        raise ValueError("calibration manifest file hash mismatch")
    pilot_manifest = load_manifest(pilot_path)
    calibration_manifest = load_manifest(calibration_path)
    loader = ProofWriterLoader(
        archive,
        dataset_version=config.dataset_version,
        dataset_manifest_reference="datasets/proofwriter/provenance.observed.json",
    )
    pilot = tuple(load_selected_examples(loader, pilot_manifest))
    calibration = tuple(load_selected_examples(loader, calibration_manifest))
    if len(pilot) != 30 or any(item.split is not Split.DEVELOPMENT for item in pilot):
        raise ValueError("parser pilot must contain exactly 30 development examples")
    if any(item.split is not Split.TRAIN for item in calibration):
        raise ValueError("parser calibration must be training-only")
    if {item.example_id for item in pilot} & {item.example_id for item in calibration}:
        raise ValueError("parser calibration overlaps the development pilot")
    return PreparedParserExperiment(
        config=config,
        config_path=path,
        root=root,
        theory_prompt=theory_prompt,
        query_prompt=query_prompt,
        pilot_examples=pilot,
        calibration_examples=calibration,
    )
