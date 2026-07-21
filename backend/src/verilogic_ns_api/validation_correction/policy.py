from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

from verilogic_ns_api.reasoning.engine import ForwardChainingEngine
from verilogic_ns_api.reasoning.models import ReasoningStatus, Theory
from verilogic_ns_api.reasoning.verifier import ProofVerifier
from verilogic_ns_api.research.models import BenchmarkExample, PredictionLabel, PredictionRecord
from verilogic_ns_api.semantic_parsing.views import (
    PreparedTheoryView,
    prepare_query_view,
)
from verilogic_ns_api.validation_correction.feedback import (
    validate_query_candidate,
    validate_theory_candidate,
)
from verilogic_ns_api.validation_correction.models import (
    AbstentionReason,
    ComponentDecision,
)


@dataclass(frozen=True)
class PolicyResult:
    predictions: tuple[PredictionRecord, ...]
    parsed_theories: dict[str, Theory]
    proof_attempted: int
    proof_verified: int
    abstention_reasons: dict[str, int]


def apply_policy(
    *,
    examples: tuple[BenchmarkExample, ...],
    theory_views: dict[str, PreparedTheoryView],
    theory_decisions: dict[str, ComponentDecision],
    query_decisions: dict[str, ComponentDecision],
    selective: bool,
) -> PolicyResult:
    engine = ForwardChainingEngine()
    verifier = ProofVerifier()
    predictions: list[PredictionRecord] = []
    parsed: dict[str, Theory] = {}
    abstentions: Counter[str] = Counter()
    attempted = verified = 0
    for example in examples:
        key = example.theory_id or example.example_id
        theory_decision = theory_decisions[key]
        query_decision = query_decisions[example.example_id]
        error_type = theory_decision.error_type or query_decision.error_type
        if error_type:
            predictions.append(_prediction(example, PredictionLabel.ERROR, error_type=error_type))
            continue
        accepted_field = "selective_accepted" if selective else "deterministic_accepted"
        theory_accepted = bool(getattr(theory_decision, accepted_field))
        query_accepted = bool(getattr(query_decision, accepted_field))
        if not theory_accepted or not query_accepted:
            reason = (
                theory_decision.abstention_reason
                if not theory_accepted
                else query_decision.abstention_reason
            ) or AbstentionReason.RELIABILITY_GATE_FAILED
            abstentions[reason.value] += 1
            predictions.append(
                _prediction(example, PredictionLabel.ABSTAIN, abstention_reason=reason.value)
            )
            continue
        if theory_decision.final_candidate is None or query_decision.final_candidate is None:
            abstentions[AbstentionReason.RELIABILITY_GATE_FAILED.value] += 1
            predictions.append(
                _prediction(
                    example,
                    PredictionLabel.ABSTAIN,
                    abstention_reason=AbstentionReason.RELIABILITY_GATE_FAILED.value,
                )
            )
            continue
        theory_validation = validate_theory_candidate(
            theory_decision.final_candidate,
            theory_views[key],
            theory_id=example.theory_id or "phase6_theory",
        )
        query_validation = validate_query_candidate(
            query_decision.final_candidate,
            prepare_query_view(example),
            body=theory_validation.converted,
        )
        if (
            not theory_validation.valid
            or not query_validation.valid
            or query_validation.theory is None
        ):
            reason = AbstentionReason.RELIABILITY_GATE_FAILED
            abstentions[reason.value] += 1
            predictions.append(
                _prediction(example, PredictionLabel.ABSTAIN, abstention_reason=reason.value)
            )
            continue
        theory = query_validation.theory
        reasoning = engine.reason(theory)
        if reasoning.result.status is ReasoningStatus.INCONSISTENT:
            reason = AbstentionReason.UNEXPECTED_INCONSISTENCY
            abstentions[reason.value] += 1
            predictions.append(
                _prediction(example, PredictionLabel.ABSTAIN, abstention_reason=reason.value)
            )
            continue
        attempted += 1
        try:
            verifier.verify_result(theory, reasoning.result)
            verified += 1
        except Exception as error:  # independent verification fails closed as ERROR
            predictions.append(
                _prediction(
                    example,
                    PredictionLabel.ERROR,
                    error_type=f"PROOF_VERIFICATION_ERROR:{type(error).__name__}",
                )
            )
            continue
        label = {
            ReasoningStatus.ENTAILED: PredictionLabel.ENTAILED,
            ReasoningStatus.CONTRADICTED: PredictionLabel.CONTRADICTED,
            ReasoningStatus.UNKNOWN: PredictionLabel.UNKNOWN,
        }[reasoning.result.status]
        parsed[example.example_id] = theory
        predictions.append(_prediction(example, label))
    return PolicyResult(
        predictions=tuple(predictions),
        parsed_theories=parsed,
        proof_attempted=attempted,
        proof_verified=verified,
        abstention_reasons=dict(sorted(abstentions.items())),
    )


def _prediction(
    example: BenchmarkExample,
    label: PredictionLabel,
    *,
    abstention_reason: str | None = None,
    error_type: str | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        run_id="phase6-corrected-selective",
        example_id=example.example_id,
        predicted_label=label,
        abstention_reason=abstention_reason,
        error_type=error_type,
        latency_ms=0,
        cache_hit=True,
        configured_model="qwen3.5:4b-q4_K_M",
        returned_model="qwen3.5:4b-q4_K_M",
        provider_version="0.32.1",
        model_digest="2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd",
        execution_device="cpu",
        estimated_cost_usd=0,
        predictor_name=(
            "validation-correction-selective"
            if label is PredictionLabel.ABSTAIN
            else "validation-correction"
        ),
        predictor_version="1.0",
        timestamp=datetime.now(UTC),
    )
