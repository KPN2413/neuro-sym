from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldLabel(StrEnum):
    ENTAILED = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


class PredictionLabel(StrEnum):
    ENTAILED = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"
    ABSTAIN = "ABSTAIN"
    ERROR = "ERROR"


class Split(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "dev"
    TEST = "test"


class WorldAssumption(StrEnum):
    OPEN = "OWA"
    CLOSED = "CWA"


class SourceStatement(StrictModel):
    source_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=10000)
    kind: Literal["fact", "rule", "sentence"]
    representation: str | None = Field(default=None, max_length=20000)


class StructuredStatement(StrictModel):
    text: str = Field(min_length=1, max_length=10000)
    representation: str | None = Field(default=None, max_length=20000)


class ExampleProvenance(StrictModel):
    loader_name: str = "proofwriter"
    loader_version: str
    record_line: int = Field(ge=1)
    record_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_manifest_reference: str | None = Field(default=None, max_length=1024)


class BenchmarkExample(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    example_id: str = Field(min_length=1, max_length=512)
    dataset_name: Literal["proofwriter"] = "proofwriter"
    dataset_version: str = Field(min_length=1, max_length=64)
    variant: str = Field(min_length=1, max_length=128)
    split: Split
    theory_id: str | None = Field(default=None, max_length=256)
    question_id: str | None = Field(default=None, max_length=256)
    reasoning_depth: int | None = Field(default=None, ge=0)
    source_statements: list[SourceStatement] = Field(min_length=1)
    context: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=10000)
    gold_label: GoldLabel
    original_raw_label: bool | str
    world_assumption: WorldAssumption
    structured_facts: dict[str, StructuredStatement] = Field(default_factory=dict)
    structured_rules: dict[str, StructuredStatement] = Field(default_factory=dict)
    gold_proofs: dict[str, JsonValue] | None = None
    source_relative_path: str = Field(min_length=1, max_length=2048)
    provenance: ExampleProvenance

    def for_prediction(self) -> PredictionInput:
        """Return a gold-redacted view for predictors."""
        return PredictionInput(
            example_id=self.example_id,
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            variant=self.variant,
            split=self.split,
            theory_id=self.theory_id,
            question_id=self.question_id,
            reasoning_depth=self.reasoning_depth,
            source_statements=self.source_statements,
            context=self.context,
            query=self.query,
            world_assumption=self.world_assumption,
            structured_facts=self.structured_facts,
            structured_rules=self.structured_rules,
            source_relative_path=self.source_relative_path,
        )


class PredictionInput(StrictModel):
    """Gold-free input passed to every genuine predictor."""

    example_id: str
    dataset_name: Literal["proofwriter"]
    dataset_version: str
    variant: str
    split: Split
    theory_id: str | None
    question_id: str | None
    reasoning_depth: int | None
    source_statements: list[SourceStatement]
    context: str
    query: str
    world_assumption: WorldAssumption
    structured_facts: dict[str, StructuredStatement]
    structured_rules: dict[str, StructuredStatement]
    source_relative_path: str


class PredictionRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
    example_id: str = Field(min_length=1, max_length=512)
    predicted_label: PredictionLabel
    confidence: float | None = Field(default=None, ge=0, le=1)
    abstention_reason: str | None = Field(default=None, max_length=1000)
    error_type: str | None = Field(default=None, max_length=256)
    latency_ms: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    raw_output_reference: str | None = Field(default=None, max_length=2048)
    provider_request_id: str | None = Field(default=None, max_length=256)
    configured_model: str | None = Field(default=None, max_length=256)
    returned_model: str | None = Field(default=None, max_length=256)
    provider_version: str | None = Field(default=None, max_length=64)
    model_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    execution_device: Literal["cpu", "gpu", "hybrid"] | None = None
    provider_total_duration_ms: float | None = Field(default=None, ge=0)
    provider_load_duration_ms: float | None = Field(default=None, ge=0)
    provider_prompt_eval_duration_ms: float | None = Field(default=None, ge=0)
    provider_generation_duration_ms: float | None = Field(default=None, ge=0)
    generation_tokens_per_second: float | None = Field(default=None, ge=0)
    request_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    prompt_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    retry_count: int | None = Field(default=None, ge=0)
    cache_hit: bool | None = None
    provider_status: str | None = Field(default=None, max_length=64)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    predictor_name: str = Field(min_length=1, max_length=256)
    predictor_version: str = Field(min_length=1, max_length=64)
    timestamp: datetime

    @model_validator(mode="after")
    def validate_outcome_details(self) -> Self:
        if self.predicted_label is PredictionLabel.ABSTAIN and not self.abstention_reason:
            raise ValueError("ABSTAIN predictions require an abstention_reason")
        if self.predicted_label is PredictionLabel.ERROR and not self.error_type:
            raise ValueError("ERROR predictions require an error_type")
        return self


class PerLabelMetrics(StrictModel):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    support: int = Field(ge=0)
    predicted: int = Field(ge=0)


class PerDepthMetrics(StrictModel):
    total: int = Field(ge=0)
    answered: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)


class MetricReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    total_examples: int = Field(ge=0)
    answered_examples: int = Field(ge=0)
    abstained_examples: int = Field(ge=0)
    errored_examples: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    answered_only_accuracy: float | None = Field(default=None, ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    selective_risk: float | None = Field(default=None, ge=0, le=1)
    macro_precision: float = Field(ge=0, le=1)
    macro_recall: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    confusion_matrix: dict[str, dict[str, int]]
    per_label_metrics: dict[str, PerLabelMetrics]
    per_depth_metrics: dict[str, PerDepthMetrics]
    invalid_prediction_count: int = Field(ge=0)
    refusal_count: int = Field(default=0, ge=0)
    cache_hit_count: int = Field(default=0, ge=0)
    cache_miss_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    non_cache_total_latency_ms: float = Field(default=0, ge=0)
    non_cache_median_latency_ms: float | None = Field(default=None, ge=0)
    provider_total_duration_ms: float = Field(default=0, ge=0)
    provider_load_duration_ms: float = Field(default=0, ge=0)
    provider_prompt_eval_duration_ms: float = Field(default=0, ge=0)
    provider_generation_duration_ms: float = Field(default=0, ge=0)
    generation_tokens_per_second: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)


class RunStatus(StrEnum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    FAILED = "failed"


class RunManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
    status: RunStatus
    dataset_manifest_reference: str
    selected_splits: list[Split]
    configuration: dict[str, JsonValue]
    seed: int
    predictor_name: str
    predictor_version: str
    git_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    git_dirty: bool | None = None
    started_at: datetime
    completed_at: datetime | None = None
    environment: dict[str, JsonValue]
    example_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    abstention_count: int = Field(ge=0)
    error_count: int = Field(ge=0)


class SamplingStrategy(StrEnum):
    RANDOM = "random"
    BALANCED = "balanced"
    STRATIFIED = "stratified"


class SamplingConfig(StrictModel):
    seed: int = 42
    max_examples: Annotated[int | None, Field(gt=0)] = None
    allowed_splits: list[Split] = Field(default_factory=lambda: [Split.TRAIN, Split.DEVELOPMENT])
    allow_test: bool = False
    labels: list[GoldLabel] | None = None
    reasoning_depths: list[Annotated[int, Field(ge=0)]] | None = None
    strategy: SamplingStrategy = SamplingStrategy.RANDOM

    @model_validator(mode="after")
    def validate_unique_filters(self) -> Self:
        for name, values in (
            ("allowed_splits", self.allowed_splits),
            ("labels", self.labels),
            ("reasoning_depths", self.reasoning_depths),
        ):
            if values is not None and len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if not self.allowed_splits:
            raise ValueError("allowed_splits must not be empty")
        return self


class DatasetSelection(StrictModel):
    name: Literal["proofwriter"] = "proofwriter"
    data_source: str
    version: str = "V2020.12.3"
    variant: str
    world_assumption: Literal["OWA"] = "OWA"
    splits: list[Split] = Field(default_factory=lambda: [Split.TRAIN, Split.DEVELOPMENT])
    manifest_reference: str


class PredictorConfig(StrictModel):
    kind: Literal["constant_unknown"] = "constant_unknown"
    version: str = "1.0"


class RunConfig(StrictModel):
    output_directory: str = "results/runs"
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
    run_id_prefix: str = Field(
        default="proofwriter-smoke", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
    )


class EvaluationConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset: DatasetSelection
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    run: RunConfig = Field(default_factory=RunConfig)
