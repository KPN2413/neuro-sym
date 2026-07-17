from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, JsonValue, model_validator

from verilogic_ns_api.research.models import GoldLabel, Split, StrictModel


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


class BaselineCondition(StrEnum):
    DIRECT = "direct"
    FEW_SHOT = "few_shot"


class ProviderStatus(StrEnum):
    SUCCESS = "success"
    REFUSAL = "refusal"


class BaselineOutput(StrictModel):
    label: GoldLabel


class UsageTelemetry(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ProviderTimingTelemetry(StrictModel):
    total_duration_ms: float | None = Field(default=None, ge=0)
    load_duration_ms: float | None = Field(default=None, ge=0)
    prompt_eval_duration_ms: float | None = Field(default=None, ge=0)
    generation_duration_ms: float | None = Field(default=None, ge=0)
    generation_tokens_per_second: float | None = Field(default=None, ge=0)


class LLMRequest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    provider: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    api_family: Literal["responses", "native_chat"] = "responses"
    configured_model: str = Field(min_length=1, max_length=256)
    endpoint_identity: str | None = Field(default=None, max_length=256)
    provider_version: str | None = Field(default=None, min_length=1, max_length=64)
    model_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_options: dict[str, JsonValue] = Field(default_factory=dict)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"]
    max_output_tokens: int = Field(ge=16, le=128000)
    instructions: str = Field(min_length=1)
    input_text: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1, max_length=64)
    prompt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_version: Literal["1.0"] = "1.0"
    output_schema_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    demonstration_manifest_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    selection_manifest_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    example_id: str = Field(min_length=1, max_length=512)
    rendered_request_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    def cache_identity(self) -> dict[str, JsonValue]:
        return {
            "provider": self.provider,
            "api_family": self.api_family,
            "configured_model": self.configured_model,
            "endpoint_identity": self.endpoint_identity,
            "provider_version": self.provider_version,
            "model_digest": self.model_digest,
            "model_options": self.model_options,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "output_schema_version": self.output_schema_version,
            "output_schema_hash": self.output_schema_hash,
            "demonstration_manifest_hash": self.demonstration_manifest_hash,
            "selection_manifest_hash": self.selection_manifest_hash,
            "example_id": self.example_id,
            "rendered_request_hash": self.rendered_request_hash,
        }

    @property
    def cache_key(self) -> str:
        return sha256_json(self.cache_identity())


class LLMResponse(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    request_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: ProviderStatus
    label: GoldLabel | None = None
    refusal_reason: str | None = Field(default=None, max_length=1000)
    provider_request_id: str | None = Field(default=None, max_length=256)
    configured_model: str = Field(min_length=1, max_length=256)
    returned_model: str | None = Field(default=None, max_length=256)
    provider_version: str | None = Field(default=None, min_length=1, max_length=64)
    model_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    execution_device: Literal["cpu", "gpu", "hybrid"] | None = None
    usage: UsageTelemetry = Field(default_factory=UsageTelemetry)
    provider_timing: ProviderTimingTelemetry = Field(default_factory=ProviderTimingTelemetry)
    started_at: datetime
    completed_at: datetime
    latency_ms: float = Field(ge=0)
    retry_count: int = Field(default=0, ge=0)
    cache_hit: bool = False
    estimated_cost_usd: float = Field(default=0, ge=0)
    raw_provider_payload: JsonValue | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status is ProviderStatus.SUCCESS and self.label is None:
            raise ValueError("Successful provider responses require a label")
        if self.status is ProviderStatus.REFUSAL and not self.refusal_reason:
            raise ValueError("Provider refusals require a reason")
        if self.status is ProviderStatus.REFUSAL and self.label is not None:
            raise ValueError("Provider refusals cannot contain a label")
        return self


class CacheEntry(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_identity: dict[str, JsonValue]
    response: LLMResponse


class RetryConfig(StrictModel):
    max_attempts: int = Field(default=3, ge=1, le=6)
    base_delay_seconds: float = Field(default=1.0, ge=0, le=60)
    max_delay_seconds: float = Field(default=8.0, ge=0, le=120)
    jitter_seconds: float = Field(default=0.25, ge=0, le=5)
    circuit_breaker_threshold: int = Field(default=3, ge=1, le=20)


class ProviderConfig(StrictModel):
    name: Literal["openai", "ollama"] = "openai"
    api_family: Literal["responses", "native_chat"] = "responses"
    model: str = Field(default="gpt-5.6-terra", min_length=1, max_length=256)
    endpoint: str | None = Field(default=None, max_length=256)
    model_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    provider_version: str | None = Field(default=None, min_length=1, max_length=64)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"
    temperature: float | None = Field(default=None, ge=0, le=2)
    sampling_seed: int | None = None
    context_tokens: int | None = Field(default=None, ge=512, le=131072)
    think: bool | None = None
    keep_alive: str | None = Field(default=None, pattern=r"^[0-9]+(?:ms|s|m|h)$")
    execution_device: Literal["cpu", "gpu", "hybrid"] | None = None
    max_output_tokens: int = Field(default=512, ge=16, le=128000)
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    concurrency: int = Field(default=3, ge=1, le=16)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    def request_options(self) -> dict[str, JsonValue]:
        if self.name == "openai":
            return {}
        return {
            "temperature": self.temperature,
            "seed": self.sampling_seed,
            "num_ctx": self.context_tokens,
            "num_predict": self.max_output_tokens,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "execution_device": self.execution_device,
        }

    @model_validator(mode="after")
    def validate_provider_contract(self) -> Self:
        if self.name == "openai":
            if self.api_family != "responses":
                raise ValueError("OpenAI provider requires the Responses API family")
            local_only = (
                self.endpoint,
                self.model_digest,
                self.provider_version,
                self.temperature,
                self.sampling_seed,
                self.context_tokens,
                self.think,
                self.keep_alive,
                self.execution_device,
            )
            if any(item is not None for item in local_only):
                raise ValueError("OpenAI provider cannot contain Ollama-only settings")
            return self

        if self.api_family != "native_chat":
            raise ValueError("Ollama provider requires the native_chat API family")
        if self.endpoint not in {"http://127.0.0.1:11434", "http://localhost:11434"}:
            raise ValueError("Ollama endpoint must be loopback HTTP on port 11434")
        if "cloud" in self.model.casefold():
            raise ValueError("Ollama cloud model tags are forbidden")
        required = {
            "model_digest": self.model_digest,
            "provider_version": self.provider_version,
            "temperature": self.temperature,
            "sampling_seed": self.sampling_seed,
            "context_tokens": self.context_tokens,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "execution_device": self.execution_device,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Ollama provider is missing frozen settings: {', '.join(missing)}")
        if self.temperature != 0:
            raise ValueError("Ollama baseline temperature must be zero")
        if self.think is not False:
            raise ValueError("Ollama baseline thinking must be disabled")
        if self.reasoning_effort != "none":
            raise ValueError("Ollama baseline reasoning_effort must be none")
        if self.concurrency != 1:
            raise ValueError("Ollama baseline concurrency must be one")
        return self


class PricingConfig(StrictModel):
    source_url: str = "https://developers.openai.com/api/docs/pricing"
    as_of: date
    model: str
    service_tier: Literal["standard"] = "standard"
    input_usd_per_million: float = Field(ge=0)
    cached_input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    long_context_threshold_tokens: int = Field(default=272000, gt=0)
    long_context_input_multiplier: float = Field(default=2.0, ge=1)
    long_context_output_multiplier: float = Field(default=1.5, ge=1)


class DatasetPilotConfig(StrictModel):
    data_source: str
    dataset_manifest_reference: str
    archive_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    version: Literal["V2020.12.3"] = "V2020.12.3"
    variant: str = "depth-5"
    split: Literal["dev"] = "dev"
    selection_manifest: str


class PromptConfig(StrictModel):
    path: str
    version: Literal["v1"] = "v1"
    expected_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_path: str = "schemas/llm-baseline-output.v1.schema.json"
    output_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    demonstration_manifest: str | None = None


class BaselineRunConfig(StrictModel):
    output_directory: str = "results/runs"
    cache_directory: str = "results/cache/llm-responses"
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
    run_id_prefix: str = Field(
        default="openai-baseline", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
    )


class BaselineConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    condition: BaselineCondition
    provider: ProviderConfig
    dataset: DatasetPilotConfig
    prompt: PromptConfig
    pricing: PricingConfig | None = None
    seed: int = 20260713
    predictor_version: Literal["1.0"] = "1.0"
    run: BaselineRunConfig = Field(default_factory=BaselineRunConfig)

    @model_validator(mode="after")
    def validate_condition(self) -> Self:
        if self.provider.name == "openai":
            if self.pricing is None:
                raise ValueError("OpenAI baselines require pricing metadata")
            if self.pricing.model != self.provider.model:
                raise ValueError("Pricing model must match the configured provider model")
        elif self.pricing is not None:
            raise ValueError("Zero-cost Ollama baselines cannot specify API pricing")
        if self.condition is BaselineCondition.DIRECT:
            if self.prompt.demonstration_manifest is not None:
                raise ValueError("Direct baseline cannot specify demonstrations")
        elif self.prompt.demonstration_manifest is None:
            raise ValueError("Few-shot baseline requires a demonstration manifest")
        return self


class SelectionEntry(StrictModel):
    example_id: str = Field(min_length=1, max_length=512)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reasoning_depth: int = Field(ge=0)
    label: GoldLabel
    split: Split


class SelectionManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    selection_kind: Literal["demonstrations", "pilot"]
    dataset_name: Literal["ProofWriter"] = "ProofWriter"
    dataset_version: Literal["V2020.12.3"] = "V2020.12.3"
    archive_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    world_assumption: Literal["OWA"] = "OWA"
    variant: str
    split: Split
    seed: int
    sampler_version: Literal["phase3-v1"] = "phase3-v1"
    entries: list[SelectionEntry]
    manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    def calculated_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        return sha256_json(payload)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.manifest_hash != self.calculated_hash():
            raise ValueError("Selection manifest hash does not match its contents")
        if any(entry.split is not self.split for entry in self.entries):
            raise ValueError("Selection entry split does not match manifest split")
        ids = [entry.example_id for entry in self.entries]
        hashes = [entry.content_sha256 for entry in self.entries]
        if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
            raise ValueError("Selection manifest contains duplicate IDs or content hashes")
        return self


class PlanReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    condition: BaselineCondition
    provider: str
    api_family: str
    configured_model: str
    reasoning_effort: str
    planned_requests: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    new_provider_requests: int = Field(default=0, ge=0)
    new_billable_requests: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    maximum_output_tokens_per_request: int = Field(ge=0)
    estimated_worst_case_output_tokens: int = Field(ge=0)
    pricing_source: str | None = None
    pricing_as_of: date | None = None
    estimated_worst_case_usd: float = Field(ge=0)
    endpoint_identity: str | None = None
    provider_version: str | None = None
    model_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_options: dict[str, JsonValue] = Field(default_factory=dict)
    dataset_version: str
    dataset_variant: str
    dataset_splits: list[Split]
    external_data_description: str
    prompt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    selection_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    demonstration_manifest_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class PairedComparison(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    direct_run_id: str
    few_shot_run_id: str
    example_count: int = Field(ge=0)
    accuracy_delta: float
    coverage_delta: float
    per_depth_accuracy_delta: dict[str, float]
    per_label_f1_delta: dict[str, float]
    both_correct: int = Field(ge=0)
    direct_only_correct: int = Field(ge=0)
    few_shot_only_correct: int = Field(ge=0)
    both_incorrect: int = Field(ge=0)
    prediction_disagreement_matrix: dict[str, dict[str, int]]
    significance_claimed: Literal[False] = False
