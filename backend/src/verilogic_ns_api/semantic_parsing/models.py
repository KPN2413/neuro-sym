from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from verilogic_ns_api.reasoning.models import EntityTerm, Term, VariableDefinition

PREDICATE_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
NEUTRAL_SOURCE_PATTERN = r"^sent[1-9][0-9]*$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NeutralStatement(StrictModel):
    source_id: str = Field(pattern=NEUTRAL_SOURCE_PATTERN)
    text: str = Field(min_length=1, max_length=10_000)


class TheoryParseInput(StrictModel):
    """The only theory-bearing object a prompt renderer may accept."""

    input_hash: str = Field(pattern=SHA256_PATTERN)
    statements: tuple[NeutralStatement, ...] = Field(min_length=1, max_length=256)


class QueryParseInput(StrictModel):
    """The only query-bearing object a prompt renderer may accept."""

    input_hash: str = Field(pattern=SHA256_PATTERN)
    text: str = Field(min_length=1, max_length=10_000)


class CandidateFactLiteral(StrictModel):
    predicate: str = Field(pattern=PREDICATE_PATTERN)
    arity: Literal[1, 2]
    arguments: tuple[EntityTerm, ...] = Field(min_length=1, max_length=2)
    negated: bool

    @model_validator(mode="after")
    def arity_matches_arguments(self) -> Self:
        if len(self.arguments) != self.arity:
            raise ValueError("literal arity does not match argument count")
        return self


class CandidateRuleLiteral(StrictModel):
    predicate: str = Field(pattern=PREDICATE_PATTERN)
    arity: Literal[1, 2]
    arguments: tuple[Term, ...] = Field(min_length=1, max_length=2)
    negated: bool

    @model_validator(mode="after")
    def arity_matches_arguments(self) -> Self:
        if len(self.arguments) != self.arity:
            raise ValueError("literal arity does not match argument count")
        return self


class CandidateRule(StrictModel):
    variables: tuple[VariableDefinition, ...] = Field(max_length=32)
    body: tuple[CandidateRuleLiteral, ...] = Field(min_length=1, max_length=32)
    head: CandidateRuleLiteral


class CandidateFactStatement(StrictModel):
    source_id: str = Field(pattern=NEUTRAL_SOURCE_PATTERN)
    kind: Literal["fact"]
    fact: CandidateFactLiteral


class CandidateRuleStatement(StrictModel):
    source_id: str = Field(pattern=NEUTRAL_SOURCE_PATTERN)
    kind: Literal["rule"]
    rule: CandidateRule


class CandidateTheoryOutput(StrictModel):
    facts: tuple[CandidateFactStatement, ...] = Field(max_length=256)
    rules: tuple[CandidateRuleStatement, ...] = Field(max_length=256)

    @model_validator(mode="after")
    def require_statement(self) -> Self:
        if not self.facts and not self.rules:
            raise ValueError("a parsed theory must contain at least one statement")
        return self

    @property
    def statements(self) -> tuple[CandidateFactStatement | CandidateRuleStatement, ...]:
        return (*self.facts, *self.rules)


class CandidateQueryOutput(StrictModel):
    query: CandidateFactLiteral


class ParserStatus(StrEnum):
    PARSED = "PARSED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    STRUCTURED_OUTPUT_ERROR = "STRUCTURED_OUTPUT_ERROR"
    STRUCTURAL_INVALID = "STRUCTURAL_INVALID"
    SEMANTIC_INVALID = "SEMANTIC_INVALID"
    SOURCE_COVERAGE_ERROR = "SOURCE_COVERAGE_ERROR"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    TIMEOUT = "TIMEOUT"


class ParserKind(StrEnum):
    THEORY = "theory"
    QUERY = "query"


class ParserUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ParserTiming(StrictModel):
    total_duration_ms: float = Field(ge=0)
    load_duration_ms: float = Field(ge=0)
    prompt_eval_duration_ms: float = Field(ge=0)
    generation_duration_ms: float = Field(ge=0)
    generation_tokens_per_second: float | None = Field(default=None, ge=0)


class ParserResponse(StrictModel):
    request_hash: str = Field(pattern=SHA256_PATTERN)
    configured_model: str
    returned_model: str
    provider_version: str
    model_digest: str = Field(pattern=SHA256_PATTERN)
    content: dict[str, object]
    usage: ParserUsage
    timing: ParserTiming
    started_at: datetime
    completed_at: datetime
    latency_ms: float = Field(ge=0)


class ParserOutcome(StrictModel):
    parser_kind: ParserKind
    input_hash: str = Field(pattern=SHA256_PATTERN)
    request_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    status: ParserStatus
    cache_hit: bool = False
    error_type: str | None = None
    error_message: str | None = None
    candidate: dict[str, object] | None = None
    usage: ParserUsage | None = None
    timing: ParserTiming | None = None


class ParserRuntimeConfig(StrictModel):
    provider: Literal["ollama"] = "ollama"
    endpoint: Literal["http://127.0.0.1:11434", "http://localhost:11434"]
    provider_version: str
    model: str = Field(min_length=1, max_length=128)
    model_digest: str = Field(pattern=SHA256_PATTERN)
    temperature: Literal[0] = 0
    seed: int
    num_ctx: int = Field(ge=4096, le=32768)
    theory_num_predict: int = Field(ge=256, le=8192)
    query_num_predict: int = Field(ge=32, le=1024)
    think: Literal[False] = False
    keep_alive: str = Field(pattern=r"^[1-9][0-9]*[smh]$")
    timeout_seconds: float = Field(gt=0, le=3600)
    max_attempts: int = Field(default=2, ge=1, le=3)

    @model_validator(mode="after")
    def reject_cloud_model(self) -> Self:
        if "cloud" in self.model.lower():
            raise ValueError("cloud Ollama model tags are forbidden")
        return self


class ParserExperimentConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    dataset_version: str
    data_source: str
    variant: str
    archive_sha256: str = Field(pattern=SHA256_PATTERN)
    pilot_manifest: str
    pilot_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    calibration_manifest: str
    calibration_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    theory_prompt: str
    theory_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    query_prompt: str
    query_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    theory_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    query_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    cache_directory: str
    output_directory: str
    runtime: ParserRuntimeConfig
