from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from verilogic_ns_api.baselines.configuration import (
    file_sha256,
    repository_root,
    resolve_repository_path,
)
from verilogic_ns_api.reasoning.models import sha256_payload
from verilogic_ns_api.semantic_parsing.configuration import (
    PreparedParserExperiment,
    prepare_parser_experiment,
)
from verilogic_ns_api.semantic_parsing.models import CandidateQueryOutput, CandidateTheoryOutput
from verilogic_ns_api.semantic_parsing.prompts import PromptRegistry
from verilogic_ns_api.validation_correction.models import (
    CorrectionExperimentConfig,
    QueryCriticReport,
    TaskKind,
    TheoryCriticReport,
    ValidationFeedback,
)


@dataclass(frozen=True)
class PreparedCorrectionExperiment:
    config: CorrectionExperimentConfig
    config_path: Path
    root: Path
    phase5: PreparedParserExperiment
    prompts: dict[TaskKind, tuple[str, str]]


def load_correction_config(path: Path) -> CorrectionExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("validation-correction config must be a YAML mapping")
    return CorrectionExperimentConfig.model_validate(payload)


def prepare_correction_experiment(path: Path) -> PreparedCorrectionExperiment:
    root = repository_root(path)
    config = load_correction_config(path)
    phase5_path = resolve_repository_path(root, config.phase5_config)
    if file_sha256(phase5_path) != config.phase5_config_sha256:
        raise ValueError("frozen Phase 5 configuration hash mismatch")
    phase5 = prepare_parser_experiment(phase5_path)
    if config.runtime != phase5.config.runtime:
        raise ValueError("Phase 6 must reuse the exact frozen Phase 5 runtime")

    registry = PromptRegistry(root)
    prompt_specs = {
        TaskKind.CRITIC_THEORY: (
            config.critic_theory_prompt,
            config.critic_theory_prompt_sha256,
        ),
        TaskKind.CRITIC_QUERY: (
            config.critic_query_prompt,
            config.critic_query_prompt_sha256,
        ),
        TaskKind.CORRECTION_THEORY: (
            config.correction_theory_prompt,
            config.correction_theory_prompt_sha256,
        ),
        TaskKind.CORRECTION_QUERY: (
            config.correction_query_prompt,
            config.correction_query_prompt_sha256,
        ),
    }
    prompts: dict[TaskKind, tuple[str, str]] = {}
    for kind, (prompt_path, expected_hash) in prompt_specs.items():
        prompt, observed_hash = registry.load(prompt_path)
        if observed_hash != expected_hash:
            raise ValueError(f"{kind.value} prompt hash mismatch")
        prompts[kind] = (prompt, observed_hash)

    schemas = {
        "feedback_schema_sha256": ValidationFeedback.model_json_schema(),
        "critic_theory_schema_sha256": TheoryCriticReport.model_json_schema(),
        "critic_query_schema_sha256": QueryCriticReport.model_json_schema(),
        "correction_theory_schema_sha256": CandidateTheoryOutput.model_json_schema(),
        "correction_query_schema_sha256": CandidateQueryOutput.model_json_schema(),
    }
    for field, schema in schemas.items():
        if sha256_payload(schema) != getattr(config, field):
            raise ValueError(f"{field} mismatch")

    calibration = resolve_repository_path(root, config.calibration_manifest)
    if file_sha256(calibration) != config.calibration_manifest_sha256:
        raise ValueError("Phase 6 calibration manifest hash mismatch")
    return PreparedCorrectionExperiment(
        config=config,
        config_path=path,
        root=root,
        phase5=phase5,
        prompts=prompts,
    )
