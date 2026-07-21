from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from verilogic_ns_api.reasoning.models import sha256_payload
from verilogic_ns_api.semantic_parsing.models import (
    ParserRuntimeConfig,
    QueryParseInput,
    TheoryParseInput,
)

SHA256_PATTERN = r"^[a-f0-9]{64}$"
NEUTRAL_SOURCE_PATTERN = r"^sent[1-9][0-9]*$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentType(StrEnum):
    THEORY = "theory"
    QUERY = "query"


class FeedbackIssueCode(StrEnum):
    STRUCTURED_OUTPUT_ERROR = "STRUCTURED_OUTPUT_ERROR"
    MISSING_SOURCE = "MISSING_SOURCE"
    DUPLICATE_SOURCE = "DUPLICATE_SOURCE"
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"
    INVENTED_SOURCE = "INVENTED_SOURCE"
    FACT_RULE_CONFUSION = "FACT_RULE_CONFUSION"
    UNSAFE_VARIABLE = "UNSAFE_VARIABLE"
    UNBOUND_HEAD_VARIABLE = "UNBOUND_HEAD_VARIABLE"
    NON_GROUND_FACT = "NON_GROUND_FACT"
    NON_GROUND_QUERY = "NON_GROUND_QUERY"
    UNSUPPORTED_ARITY = "UNSUPPORTED_ARITY"
    PREDICATE_ARITY_CONFLICT = "PREDICATE_ARITY_CONFLICT"
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    INVALID_POLARITY = "INVALID_POLARITY"
    INVALID_TERM = "INVALID_TERM"
    MALFORMED_RULE_BODY = "MALFORMED_RULE_BODY"
    MALFORMED_RULE_HEAD = "MALFORMED_RULE_HEAD"
    UNSUPPORTED_STRUCTURE = "UNSUPPORTED_STRUCTURE"


class IssueSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class ValidationIssue(StrictModel):
    issue_code: FeedbackIssueCode
    source_id: str | None = Field(default=None, pattern=r"^(sent[1-9][0-9]*|query)$")
    ast_path: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=240)
    severity: IssueSeverity = IssueSeverity.ERROR
    retryable: bool
    related_issue_ids: tuple[str, ...] = Field(default=(), max_length=16)


class ValidationFeedback(StrictModel):
    feedback_schema_version: Literal["1.0"] = "1.0"
    component_type: ComponentType
    candidate_hash: str = Field(pattern=SHA256_PATTERN)
    issues: tuple[ValidationIssue, ...] = Field(max_length=64)

    @property
    def feedback_hash(self) -> str:
        return sha256_payload(self.model_dump(mode="json"))


class CriticDecision(StrEnum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"


class CriticCategory(StrEnum):
    FACT_RULE_CONFUSION = "FACT_RULE_CONFUSION"
    PREDICATE_MISMATCH = "PREDICATE_MISMATCH"
    POLARITY_MISMATCH = "POLARITY_MISMATCH"
    ARITY_MISMATCH = "ARITY_MISMATCH"
    CONSTANT_MISMATCH = "CONSTANT_MISMATCH"
    VARIABLE_MISMATCH = "VARIABLE_MISMATCH"
    PREMISE_MISMATCH = "PREMISE_MISMATCH"
    CONCLUSION_MISMATCH = "CONCLUSION_MISMATCH"
    OMITTED_MEANING = "OMITTED_MEANING"
    INVENTED_MEANING = "INVENTED_MEANING"
    OTHER_MISMATCH = "OTHER_MISMATCH"


class TheoryCriticIssue(StrictModel):
    source_id: str = Field(pattern=NEUTRAL_SOURCE_PATTERN)
    category: CriticCategory
    description: str = Field(min_length=1, max_length=240)


class QueryCriticIssue(StrictModel):
    source_id: Literal["query"] = "query"
    category: CriticCategory
    description: str = Field(min_length=1, max_length=240)


class TheoryCriticReport(StrictModel):
    decision: CriticDecision
    issues: tuple[TheoryCriticIssue, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def decision_matches_issues(self) -> Self:
        if self.decision is CriticDecision.ACCEPT and self.issues:
            raise ValueError("ACCEPT critic reports cannot contain issues")
        if self.decision is CriticDecision.REVISE and not self.issues:
            raise ValueError("REVISE critic reports require at least one issue")
        return self


class QueryCriticReport(StrictModel):
    decision: CriticDecision
    issues: tuple[QueryCriticIssue, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def decision_matches_issues(self) -> Self:
        if self.decision is CriticDecision.ACCEPT and self.issues:
            raise ValueError("ACCEPT critic reports cannot contain issues")
        if self.decision is CriticDecision.REVISE and not self.issues:
            raise ValueError("REVISE critic reports require at least one issue")
        return self


class TheoryCriticInput(StrictModel):
    source: TheoryParseInput
    candidate: dict[str, object]


class QueryCriticInput(StrictModel):
    source: QueryParseInput
    candidate: dict[str, object]


class TheoryCorrectionInput(StrictModel):
    source: TheoryParseInput
    previous_candidate: dict[str, object]
    validator_feedback: ValidationFeedback
    critic_report: TheoryCriticReport | None = None
    attempt: Literal[1] = 1


class QueryCorrectionInput(StrictModel):
    source: QueryParseInput
    previous_candidate: dict[str, object]
    validator_feedback: ValidationFeedback
    critic_report: QueryCriticReport | None = None
    attempt: Literal[1] = 1


class TaskKind(StrEnum):
    CRITIC_THEORY = "critic-theory"
    CRITIC_QUERY = "critic-query"
    CORRECTION_THEORY = "correction-theory"
    CORRECTION_QUERY = "correction-query"


class TaskStatus(StrEnum):
    SUCCESS = "SUCCESS"
    STRUCTURED_OUTPUT_ERROR = "STRUCTURED_OUTPUT_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


class TaskOutcome(StrictModel):
    task_kind: TaskKind
    request_hash: str = Field(pattern=SHA256_PATTERN)
    status: TaskStatus
    cache_hit: bool = False
    output: dict[str, object] | None = None
    error_type: str | None = Field(default=None, max_length=128)
    error_message: str | None = Field(default=None, max_length=500)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0, ge=0)


class ControllerState(StrEnum):
    RAW = "RAW"
    VALIDATING = "VALIDATING"
    NEEDS_CORRECTION = "NEEDS_CORRECTION"
    CRITIQUING = "CRITIQUING"
    CORRECTING = "CORRECTING"
    REVALIDATING = "REVALIDATING"
    FINAL_CRITIQUE = "FINAL_CRITIQUE"
    ACCEPTED = "ACCEPTED"
    ABSTAINED = "ABSTAINED"
    ERROR = "ERROR"


class AbstentionReason(StrEnum):
    INVALID_THEORY = "INVALID_THEORY"
    INCOMPLETE_SOURCE_COVERAGE = "INCOMPLETE_SOURCE_COVERAGE"
    INVALID_QUERY = "INVALID_QUERY"
    CRITIC_REJECTED = "CRITIC_REJECTED"
    CORRECTION_FAILED = "CORRECTION_FAILED"
    NO_CORRECTION_PROGRESS = "NO_CORRECTION_PROGRESS"
    CORRECTION_LIMIT = "CORRECTION_LIMIT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    UNEXPECTED_INCONSISTENCY = "UNEXPECTED_INCONSISTENCY"
    RELIABILITY_GATE_FAILED = "RELIABILITY_GATE_FAILED"


class StateTransition(StrictModel):
    sequence: int = Field(ge=1, le=16)
    from_state: ControllerState
    to_state: ControllerState
    event: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    candidate_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    feedback_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    request_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class ReliabilityEvidence(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    structured_output_valid: bool
    source_coverage_complete: bool
    semantic_validation_passed: bool
    critic_accepted: bool
    correction_required: bool
    correction_succeeded: bool
    reasoning_completed: bool = False
    proof_or_result_verified: bool = False
    unresolved_warning_count: int = Field(default=0, ge=0, le=64)
    attempt_count: int = Field(default=0, ge=0, le=1)
    no_progress_detected: bool = False
    unexpected_inconsistency: bool = False

    @property
    def mandatory_gates_passed(self) -> bool:
        return (
            self.structured_output_valid
            and self.source_coverage_complete
            and self.semantic_validation_passed
            and self.critic_accepted
            and not self.no_progress_detected
            and not self.unexpected_inconsistency
        )


class ComponentDecision(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    component_type: ComponentType
    input_hash: str = Field(pattern=SHA256_PATTERN)
    raw_candidate_hash: str = Field(pattern=SHA256_PATTERN)
    final_candidate_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    final_candidate: dict[str, object] | None = None
    deterministic_accepted: bool
    selective_accepted: bool
    correction_attempts: int = Field(ge=0, le=1)
    critic_decision: CriticDecision | None = None
    abstention_reason: AbstentionReason | None = None
    error_type: str | None = Field(default=None, max_length=128)
    reliability: ReliabilityEvidence
    transitions: tuple[StateTransition, ...] = Field(min_length=2, max_length=16)
    task_outcomes: tuple[TaskOutcome, ...] = Field(default=(), max_length=3)


class CorrectionTaskLimits(StrictModel):
    critic_theory_num_predict: int = Field(ge=128, le=2048)
    critic_query_num_predict: int = Field(ge=64, le=1024)
    correction_theory_num_predict: int = Field(ge=256, le=8192)
    correction_query_num_predict: int = Field(ge=64, le=2048)
    maximum_new_pilot_calls: int = Field(ge=1, le=180)
    maximum_request_characters: int = Field(ge=1_000, le=200_000)
    maximum_feedback_characters: int = Field(ge=256, le=20_000)
    correction_attempt_limit: Literal[1] = 1


class ReliabilityPolicy(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    name: Literal["mandatory-evidence-gates-v1"] = "mandatory-evidence-gates-v1"
    require_structured_output: Literal[True] = True
    require_source_coverage: Literal[True] = True
    require_semantic_validation: Literal[True] = True
    require_critic_acceptance: Literal[True] = True
    require_proof_verification: Literal[True] = True
    correction_attempt_limit: Literal[1] = 1

    @property
    def policy_hash(self) -> str:
        return sha256_payload(self.model_dump(mode="json"))


class CorrectionExperimentConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    phase5_config: str
    phase5_config_sha256: str = Field(pattern=SHA256_PATTERN)
    critic_theory_prompt: str
    critic_theory_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    critic_query_prompt: str
    critic_query_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    correction_theory_prompt: str
    correction_theory_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    correction_query_prompt: str
    correction_query_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    feedback_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    critic_theory_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    critic_query_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    correction_theory_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    correction_query_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    calibration_manifest: str
    calibration_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    cache_directory: str
    output_directory: str
    runtime: ParserRuntimeConfig
    limits: CorrectionTaskLimits
    reliability_policy: ReliabilityPolicy


class ControllerTrace(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    trace_id: str = Field(pattern=SHA256_PATTERN)
    component_type: ComponentType
    input_hash: str = Field(pattern=SHA256_PATTERN)
    transitions: tuple[StateTransition, ...]
    raw_candidate_hash: str = Field(pattern=SHA256_PATTERN)
    final_candidate_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    correction_attempts: int = Field(ge=0, le=1)
    final_state: Literal[ControllerState.ACCEPTED, ControllerState.ABSTAINED, ControllerState.ERROR]
    abstention_reason: AbstentionReason | None = None
    error_type: str | None = None
    created_at: datetime
