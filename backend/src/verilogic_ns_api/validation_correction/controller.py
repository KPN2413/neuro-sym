from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from verilogic_ns_api.semantic_parsing.converter import ConvertedTheoryBody
from verilogic_ns_api.semantic_parsing.models import (
    CandidateQueryOutput,
    CandidateTheoryOutput,
)
from verilogic_ns_api.semantic_parsing.views import PreparedQueryView, PreparedTheoryView
from verilogic_ns_api.validation_correction.feedback import (
    candidate_hash,
    validate_query_candidate,
    validate_theory_candidate,
)
from verilogic_ns_api.validation_correction.models import (
    AbstentionReason,
    ComponentDecision,
    ComponentType,
    ControllerState,
    CriticDecision,
    QueryCorrectionInput,
    QueryCriticInput,
    QueryCriticReport,
    ReliabilityEvidence,
    StateTransition,
    TaskOutcome,
    TaskStatus,
    TheoryCorrectionInput,
    TheoryCriticInput,
    TheoryCriticReport,
    ValidationFeedback,
)
from verilogic_ns_api.validation_correction.service import CorrectionTaskService, TaskExecution


class ControllerTransitionError(RuntimeError):
    pass


_ALLOWED_TRANSITIONS: dict[ControllerState, frozenset[ControllerState]] = {
    ControllerState.RAW: frozenset({ControllerState.VALIDATING}),
    ControllerState.VALIDATING: frozenset(
        {ControllerState.NEEDS_CORRECTION, ControllerState.CRITIQUING}
    ),
    ControllerState.NEEDS_CORRECTION: frozenset({ControllerState.CORRECTING}),
    ControllerState.CRITIQUING: frozenset(
        {
            ControllerState.ACCEPTED,
            ControllerState.CORRECTING,
            ControllerState.ABSTAINED,
            ControllerState.ERROR,
        }
    ),
    ControllerState.CORRECTING: frozenset(
        {ControllerState.REVALIDATING, ControllerState.ABSTAINED, ControllerState.ERROR}
    ),
    ControllerState.REVALIDATING: frozenset(
        {ControllerState.FINAL_CRITIQUE, ControllerState.ABSTAINED}
    ),
    ControllerState.FINAL_CRITIQUE: frozenset(
        {ControllerState.ACCEPTED, ControllerState.ABSTAINED, ControllerState.ERROR}
    ),
    ControllerState.ACCEPTED: frozenset(),
    ControllerState.ABSTAINED: frozenset(),
    ControllerState.ERROR: frozenset(),
}


class CorrectionStateMachine:
    def __init__(self) -> None:
        self.state = ControllerState.RAW
        self.transitions: list[StateTransition] = []
        self.correction_attempts = 0
        self.seen_candidate_hashes: set[str] = set()

    def transition(
        self,
        target: ControllerState,
        event: str,
        *,
        candidate_hash_value: str | None = None,
        feedback_hash: str | None = None,
        request_hash: str | None = None,
    ) -> None:
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise ControllerTransitionError(f"invalid transition {self.state} -> {target}")
        self.transitions.append(
            StateTransition(
                sequence=len(self.transitions) + 1,
                from_state=self.state,
                to_state=target,
                event=event,
                candidate_hash=candidate_hash_value,
                feedback_hash=feedback_hash,
                request_hash=request_hash,
            )
        )
        self.state = target

    def start_correction(self) -> None:
        if self.correction_attempts >= 1:
            raise ControllerTransitionError("semantic correction attempt limit reached")
        self.correction_attempts += 1


@dataclass(frozen=True)
class ValidationSnapshot:
    valid: bool
    candidate: BaseModel | None
    feedback: ValidationFeedback


class ValidationCorrectionController:
    def __init__(self, service: CorrectionTaskService) -> None:
        self.service = service

    def run_theory(
        self,
        *,
        view: PreparedTheoryView,
        raw_candidate: object,
        theory_id: str,
    ) -> ComponentDecision:
        def validate(raw: object) -> ValidationSnapshot:
            result = validate_theory_candidate(raw, view, theory_id=theory_id)
            return ValidationSnapshot(result.valid, result.candidate, result.feedback)

        def critique(candidate: dict[str, object]) -> TaskExecution[TheoryCriticReport]:
            return self.service.critique_theory(
                TheoryCriticInput(source=view.public, candidate=candidate)
            )

        def correct(
            previous: dict[str, object],
            feedback: ValidationFeedback,
            critic: BaseModel | None,
        ) -> TaskExecution[CandidateTheoryOutput]:
            report = critic if isinstance(critic, TheoryCriticReport) else None
            return self.service.correct_theory(
                TheoryCorrectionInput(
                    source=view.public,
                    previous_candidate=previous,
                    validator_feedback=feedback,
                    critic_report=report,
                )
            )

        allowed_sources = {item.source_id for item in view.public.statements}
        return self._run(
            component=ComponentType.THEORY,
            input_hash=view.public.input_hash,
            raw_candidate=raw_candidate,
            validate=validate,
            critique=critique,
            correct=correct,
            allowed_critic_sources=allowed_sources,
        )

    def run_query(
        self,
        *,
        view: PreparedQueryView,
        raw_candidate: object,
        body: ConvertedTheoryBody | None,
    ) -> ComponentDecision:
        def validate(raw: object) -> ValidationSnapshot:
            result = validate_query_candidate(raw, view, body=body)
            return ValidationSnapshot(result.valid, result.candidate, result.feedback)

        def critique(candidate: dict[str, object]) -> TaskExecution[QueryCriticReport]:
            return self.service.critique_query(
                QueryCriticInput(source=view.public, candidate=candidate)
            )

        def correct(
            previous: dict[str, object],
            feedback: ValidationFeedback,
            critic: BaseModel | None,
        ) -> TaskExecution[CandidateQueryOutput]:
            report = critic if isinstance(critic, QueryCriticReport) else None
            return self.service.correct_query(
                QueryCorrectionInput(
                    source=view.public,
                    previous_candidate=previous,
                    validator_feedback=feedback,
                    critic_report=report,
                )
            )

        return self._run(
            component=ComponentType.QUERY,
            input_hash=view.public.input_hash,
            raw_candidate=raw_candidate,
            validate=validate,
            critique=critique,
            correct=correct,
            allowed_critic_sources={"query"},
        )

    def _run(
        self,
        *,
        component: ComponentType,
        input_hash: str,
        raw_candidate: object,
        validate: Callable[[object], ValidationSnapshot],
        critique: Callable[[dict[str, object]], TaskExecution],
        correct: Callable[[dict[str, object], ValidationFeedback, BaseModel | None], TaskExecution],
        allowed_critic_sources: set[str],
    ) -> ComponentDecision:
        machine = CorrectionStateMachine()
        raw_hash = candidate_hash(raw_candidate)
        machine.seen_candidate_hashes.add(raw_hash)
        machine.transition(
            ControllerState.VALIDATING, "raw_candidate_received", candidate_hash_value=raw_hash
        )
        initial = validate(raw_candidate)
        tasks: list[TaskOutcome] = []
        if initial.valid and initial.candidate is not None:
            machine.transition(
                ControllerState.CRITIQUING,
                "raw_validation_passed",
                candidate_hash_value=raw_hash,
                feedback_hash=initial.feedback.feedback_hash,
            )
            first_critic = critique(initial.candidate.model_dump(mode="json"))
            tasks.append(first_critic.outcome)
            if _provider_failed(first_critic.outcome):
                machine.transition(
                    ControllerState.ERROR,
                    "critic_provider_failed",
                    request_hash=first_critic.outcome.request_hash,
                )
                return _decision(
                    component,
                    input_hash,
                    raw_hash,
                    machine,
                    tasks,
                    final_candidate=initial.candidate,
                    error_type=first_critic.outcome.error_type or "CRITIC_PROVIDER_ERROR",
                )
            if first_critic.outcome.status is TaskStatus.RESOURCE_LIMIT:
                machine.transition(
                    ControllerState.ABSTAINED,
                    "critic_resource_limit",
                    request_hash=first_critic.outcome.request_hash,
                )
                return _decision(
                    component,
                    input_hash,
                    raw_hash,
                    machine,
                    tasks,
                    final_candidate=initial.candidate,
                    deterministic_accepted=True,
                    abstention_reason=AbstentionReason.RESOURCE_LIMIT,
                )
            if first_critic.value is None or not _critic_sources_valid(
                first_critic.value, allowed_critic_sources
            ):
                machine.transition(
                    ControllerState.ABSTAINED,
                    "critic_output_invalid",
                    request_hash=first_critic.outcome.request_hash,
                )
                return _decision(
                    component,
                    input_hash,
                    raw_hash,
                    machine,
                    tasks,
                    final_candidate=initial.candidate,
                    deterministic_accepted=True,
                    abstention_reason=AbstentionReason.RELIABILITY_GATE_FAILED,
                )
            if first_critic.value.decision is CriticDecision.ACCEPT:
                machine.transition(
                    ControllerState.ACCEPTED,
                    "critic_accepted_raw",
                    candidate_hash_value=raw_hash,
                    request_hash=first_critic.outcome.request_hash,
                )
                return _decision(
                    component,
                    input_hash,
                    raw_hash,
                    machine,
                    tasks,
                    final_candidate=initial.candidate,
                    deterministic_accepted=True,
                    selective_accepted=True,
                    critic_decision=CriticDecision.ACCEPT,
                )
            machine.transition(
                ControllerState.CORRECTING,
                "critic_requested_revision",
                candidate_hash_value=raw_hash,
                request_hash=first_critic.outcome.request_hash,
            )
            critic_report: BaseModel | None = first_critic.value
        else:
            machine.transition(
                ControllerState.NEEDS_CORRECTION,
                "raw_validation_failed",
                candidate_hash_value=raw_hash,
                feedback_hash=initial.feedback.feedback_hash,
            )
            machine.transition(
                ControllerState.CORRECTING,
                "bounded_correction_started",
                candidate_hash_value=raw_hash,
                feedback_hash=initial.feedback.feedback_hash,
            )
            critic_report = None

        machine.start_correction()
        previous = raw_candidate if isinstance(raw_candidate, dict) else {}
        correction = correct(previous, initial.feedback, critic_report)
        tasks.append(correction.outcome)
        if _provider_failed(correction.outcome):
            machine.transition(
                ControllerState.ERROR,
                "correction_provider_failed",
                request_hash=correction.outcome.request_hash,
            )
            return _decision(
                component,
                input_hash,
                raw_hash,
                machine,
                tasks,
                error_type=correction.outcome.error_type or "CORRECTION_PROVIDER_ERROR",
            )
        if correction.outcome.status is TaskStatus.RESOURCE_LIMIT:
            machine.transition(
                ControllerState.ABSTAINED,
                "correction_resource_limit",
                request_hash=correction.outcome.request_hash,
            )
            return _decision(
                component,
                input_hash,
                raw_hash,
                machine,
                tasks,
                abstention_reason=AbstentionReason.RESOURCE_LIMIT,
            )
        if correction.value is None:
            machine.transition(
                ControllerState.ABSTAINED,
                "correction_output_invalid",
                request_hash=correction.outcome.request_hash,
            )
            return _decision(
                component,
                input_hash,
                raw_hash,
                machine,
                tasks,
                abstention_reason=AbstentionReason.CORRECTION_FAILED,
            )
        corrected_payload = correction.value.model_dump(mode="json")
        corrected_hash = candidate_hash(corrected_payload)
        if corrected_hash in machine.seen_candidate_hashes:
            machine.transition(
                ControllerState.ABSTAINED,
                "no_correction_progress",
                candidate_hash_value=corrected_hash,
                request_hash=correction.outcome.request_hash,
            )
            return _decision(
                component,
                input_hash,
                raw_hash,
                machine,
                tasks,
                final_candidate=correction.value,
                abstention_reason=AbstentionReason.NO_CORRECTION_PROGRESS,
                no_progress=True,
            )
        machine.seen_candidate_hashes.add(corrected_hash)
        machine.transition(
            ControllerState.REVALIDATING,
            "correction_received",
            candidate_hash_value=corrected_hash,
            request_hash=correction.outcome.request_hash,
        )
        final_validation = validate(corrected_payload)
        if not final_validation.valid or final_validation.candidate is None:
            reason = _invalid_reason(component, final_validation.feedback)
            machine.transition(
                ControllerState.ABSTAINED,
                "corrected_validation_failed",
                candidate_hash_value=corrected_hash,
                feedback_hash=final_validation.feedback.feedback_hash,
            )
            return _decision(
                component,
                input_hash,
                raw_hash,
                machine,
                tasks,
                final_candidate=final_validation.candidate or correction.value,
                abstention_reason=reason,
            )
        machine.transition(
            ControllerState.FINAL_CRITIQUE,
            "corrected_validation_passed",
            candidate_hash_value=corrected_hash,
            feedback_hash=final_validation.feedback.feedback_hash,
        )
        final_critic = critique(final_validation.candidate.model_dump(mode="json"))
        tasks.append(final_critic.outcome)
        if _provider_failed(final_critic.outcome):
            machine.transition(
                ControllerState.ERROR,
                "final_critic_provider_failed",
                request_hash=final_critic.outcome.request_hash,
            )
            return _decision(
                component,
                input_hash,
                raw_hash,
                machine,
                tasks,
                final_candidate=final_validation.candidate,
                error_type=final_critic.outcome.error_type or "CRITIC_PROVIDER_ERROR",
            )
        if final_critic.outcome.status is TaskStatus.RESOURCE_LIMIT:
            machine.transition(
                ControllerState.ABSTAINED,
                "final_critic_resource_limit",
                request_hash=final_critic.outcome.request_hash,
            )
            return _decision(
                component,
                input_hash,
                raw_hash,
                machine,
                tasks,
                final_candidate=final_validation.candidate,
                deterministic_accepted=True,
                abstention_reason=AbstentionReason.RESOURCE_LIMIT,
            )
        if final_critic.value is None or not _critic_sources_valid(
            final_critic.value, allowed_critic_sources
        ):
            machine.transition(
                ControllerState.ABSTAINED,
                "final_critic_output_invalid",
                request_hash=final_critic.outcome.request_hash,
            )
            return _decision(
                component,
                input_hash,
                raw_hash,
                machine,
                tasks,
                final_candidate=final_validation.candidate,
                deterministic_accepted=True,
                abstention_reason=AbstentionReason.RELIABILITY_GATE_FAILED,
            )
        accepted = final_critic.value.decision is CriticDecision.ACCEPT
        machine.transition(
            ControllerState.ACCEPTED if accepted else ControllerState.ABSTAINED,
            "final_critic_accepted" if accepted else "final_critic_rejected",
            candidate_hash_value=corrected_hash,
            request_hash=final_critic.outcome.request_hash,
        )
        return _decision(
            component,
            input_hash,
            raw_hash,
            machine,
            tasks,
            final_candidate=final_validation.candidate,
            deterministic_accepted=True,
            selective_accepted=accepted,
            critic_decision=final_critic.value.decision,
            abstention_reason=None if accepted else AbstentionReason.CRITIC_REJECTED,
        )


def _decision(
    component: ComponentType,
    input_hash: str,
    raw_hash: str,
    machine: CorrectionStateMachine,
    tasks: list[TaskOutcome],
    *,
    final_candidate: BaseModel | None = None,
    deterministic_accepted: bool = False,
    selective_accepted: bool = False,
    critic_decision: CriticDecision | None = None,
    abstention_reason: AbstentionReason | None = None,
    error_type: str | None = None,
    no_progress: bool = False,
) -> ComponentDecision:
    correction_required = machine.correction_attempts > 0
    reliability = ReliabilityEvidence(
        structured_output_valid=final_candidate is not None,
        source_coverage_complete=deterministic_accepted,
        semantic_validation_passed=deterministic_accepted,
        critic_accepted=selective_accepted,
        correction_required=correction_required,
        correction_succeeded=correction_required and deterministic_accepted,
        unresolved_warning_count=0 if selective_accepted else 1,
        attempt_count=machine.correction_attempts,
        no_progress_detected=no_progress,
    )
    payload = final_candidate.model_dump(mode="json") if final_candidate is not None else None
    return ComponentDecision(
        component_type=component,
        input_hash=input_hash,
        raw_candidate_hash=raw_hash,
        final_candidate_hash=candidate_hash(payload) if payload is not None else None,
        final_candidate=payload,
        deterministic_accepted=deterministic_accepted,
        selective_accepted=selective_accepted,
        correction_attempts=machine.correction_attempts,
        critic_decision=critic_decision,
        abstention_reason=abstention_reason,
        error_type=error_type,
        reliability=reliability,
        transitions=tuple(machine.transitions),
        task_outcomes=tuple(tasks),
    )


def _critic_sources_valid(report: BaseModel, allowed: set[str]) -> bool:
    issues = getattr(report, "issues", ())
    return all(getattr(item, "source_id", None) in allowed for item in issues)


def _provider_failed(outcome: TaskOutcome) -> bool:
    return outcome.status in {TaskStatus.PROVIDER_ERROR, TaskStatus.TIMEOUT}


def _invalid_reason(component: ComponentType, feedback: ValidationFeedback) -> AbstentionReason:
    coverage_codes = {
        "MISSING_SOURCE",
        "DUPLICATE_SOURCE",
        "UNKNOWN_SOURCE",
        "INVENTED_SOURCE",
        "FACT_RULE_CONFUSION",
    }
    if any(item.issue_code.value in coverage_codes for item in feedback.issues):
        return AbstentionReason.INCOMPLETE_SOURCE_COVERAGE
    return (
        AbstentionReason.INVALID_THEORY
        if component is ComponentType.THEORY
        else AbstentionReason.INVALID_QUERY
    )
