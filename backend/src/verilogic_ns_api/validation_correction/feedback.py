from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from verilogic_ns_api.reasoning.models import Theory, sha256_payload
from verilogic_ns_api.semantic_parsing.converter import (
    ConvertedTheoryBody,
    ParserSemanticError,
    combine_theory_and_query,
    convert_theory_candidate,
)
from verilogic_ns_api.semantic_parsing.models import CandidateQueryOutput, CandidateTheoryOutput
from verilogic_ns_api.semantic_parsing.views import PreparedQueryView, PreparedTheoryView
from verilogic_ns_api.validation_correction.models import (
    ComponentType,
    FeedbackIssueCode,
    ValidationFeedback,
    ValidationIssue,
)


@dataclass(frozen=True)
class TheoryValidation:
    valid: bool
    candidate: CandidateTheoryOutput | None
    converted: ConvertedTheoryBody | None
    feedback: ValidationFeedback


@dataclass(frozen=True)
class QueryValidation:
    valid: bool
    candidate: CandidateQueryOutput | None
    theory: Theory | None
    feedback: ValidationFeedback


def candidate_hash(candidate: object) -> str:
    return sha256_payload({"candidate": candidate})


def validate_theory_candidate(
    raw: object,
    view: PreparedTheoryView,
    *,
    theory_id: str,
) -> TheoryValidation:
    digest = candidate_hash(raw)
    if not isinstance(raw, dict):
        feedback = _feedback(
            ComponentType.THEORY,
            digest,
            [_issue(FeedbackIssueCode.STRUCTURED_OUTPUT_ERROR, "$", retryable=True)],
        )
        return TheoryValidation(False, None, None, feedback)
    try:
        candidate = CandidateTheoryOutput.model_validate(raw)
    except ValidationError as error:
        return TheoryValidation(
            False,
            None,
            None,
            _feedback(
                ComponentType.THEORY,
                digest,
                [_validation_issue(item) for item in error.errors(include_url=False)],
            ),
        )

    expected = {item.neutral_id: item for item in view.bindings}
    observed = [item.source_id for item in candidate.statements]
    issues: list[ValidationIssue] = []
    for source_id in sorted(set(expected) - set(observed)):
        issues.append(
            _issue(
                FeedbackIssueCode.MISSING_SOURCE,
                "statements",
                source_id=source_id,
                retryable=True,
            )
        )
    for source_id in sorted({value for value in observed if observed.count(value) > 1}):
        issues.append(
            _issue(
                FeedbackIssueCode.DUPLICATE_SOURCE,
                "statements",
                source_id=source_id,
                retryable=True,
            )
        )
    for _source_id in sorted(set(observed) - set(expected)):
        issues.append(
            _issue(
                FeedbackIssueCode.UNKNOWN_SOURCE,
                "statements",
                source_id=None,
                retryable=True,
                description="A candidate statement uses a source ID not present in the neutral input.",
            )
        )
    for statement in candidate.statements:
        binding = expected.get(statement.source_id)
        if binding is not None and statement.kind != binding.expected_kind:
            issues.append(
                _issue(
                    FeedbackIssueCode.FACT_RULE_CONFUSION,
                    f"statements.{statement.source_id}.kind",
                    source_id=statement.source_id,
                    retryable=True,
                )
            )
    if issues:
        return TheoryValidation(
            False, candidate, None, _feedback(ComponentType.THEORY, digest, issues)
        )
    try:
        converted = convert_theory_candidate(candidate, view, theory_id=theory_id)
    except ParserSemanticError as error:
        issue = _semantic_issue(str(error), ComponentType.THEORY)
        return TheoryValidation(
            False,
            candidate,
            None,
            _feedback(ComponentType.THEORY, digest, [issue]),
        )
    return TheoryValidation(True, candidate, converted, _feedback(ComponentType.THEORY, digest, []))


def validate_query_candidate(
    raw: object,
    view: PreparedQueryView,
    *,
    body: ConvertedTheoryBody | None,
) -> QueryValidation:
    digest = candidate_hash(raw)
    if not isinstance(raw, dict):
        feedback = _feedback(
            ComponentType.QUERY,
            digest,
            [
                _issue(
                    FeedbackIssueCode.STRUCTURED_OUTPUT_ERROR,
                    "$",
                    source_id="query",
                    retryable=True,
                )
            ],
        )
        return QueryValidation(False, None, None, feedback)
    try:
        candidate = CandidateQueryOutput.model_validate(raw)
    except ValidationError as error:
        return QueryValidation(
            False,
            None,
            None,
            _feedback(
                ComponentType.QUERY,
                digest,
                [
                    _validation_issue(item, source_id="query")
                    for item in error.errors(include_url=False)
                ],
            ),
        )
    if body is None:
        return QueryValidation(True, candidate, None, _feedback(ComponentType.QUERY, digest, []))
    try:
        theory = combine_theory_and_query(body, candidate, view)
    except ParserSemanticError as error:
        return QueryValidation(
            False,
            candidate,
            None,
            _feedback(
                ComponentType.QUERY,
                digest,
                [_semantic_issue(str(error), ComponentType.QUERY, source_id="query")],
            ),
        )
    return QueryValidation(True, candidate, theory, _feedback(ComponentType.QUERY, digest, []))


def _feedback(
    component: ComponentType, digest: str, issues: list[ValidationIssue]
) -> ValidationFeedback:
    unique = {sha256_payload(item.model_dump(mode="json")): item for item in issues}
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.issue_code.value,
                item.source_id or "",
                item.ast_path,
                item.description,
            ),
        )[:64]
    )
    return ValidationFeedback(
        component_type=component,
        candidate_hash=digest,
        issues=ordered,
    )


def _validation_issue(error: dict[str, object], *, source_id: str | None = None) -> ValidationIssue:
    location = error.get("loc", ())
    path = ".".join(str(item) for item in location)[:160] or "$"
    error_type = str(error.get("type", ""))
    message = str(error.get("msg", "")).lower()
    code = FeedbackIssueCode.UNSUPPORTED_STRUCTURE
    if "missing" in error_type or "at least one statement" in message:
        code = FeedbackIssueCode.STRUCTURED_OUTPUT_ERROR
    elif "pattern" in error_type or "identifier" in message:
        code = FeedbackIssueCode.INVALID_IDENTIFIER
    elif "arity" in message or "too_long" in error_type or "too_short" in error_type:
        code = FeedbackIssueCode.UNSUPPORTED_ARITY
    elif "literal" in path and "negated" in path:
        code = FeedbackIssueCode.INVALID_POLARITY
    elif "arguments" in path:
        code = FeedbackIssueCode.INVALID_TERM
    elif "body" in path:
        code = FeedbackIssueCode.MALFORMED_RULE_BODY
    elif "head" in path:
        code = FeedbackIssueCode.MALFORMED_RULE_HEAD
    return _issue(code, path, source_id=source_id, retryable=True)


def _semantic_issue(
    message: str,
    component: ComponentType,
    *,
    source_id: str | None = None,
) -> ValidationIssue:
    lowered = message.lower()
    code = FeedbackIssueCode.UNSUPPORTED_STRUCTURE
    path = "query" if component is ComponentType.QUERY else "theory"
    if "conflicting arities" in lowered or "arity conflicts" in lowered:
        code = FeedbackIssueCode.PREDICATE_ARITY_CONFLICT
    elif "unsafe" in lowered or "variable" in lowered:
        code = FeedbackIssueCode.UNSAFE_VARIABLE
    return _issue(code, path, source_id=source_id, retryable=True)


def _issue(
    code: FeedbackIssueCode,
    path: str,
    *,
    source_id: str | None = None,
    retryable: bool,
    description: str | None = None,
) -> ValidationIssue:
    descriptions = {
        FeedbackIssueCode.STRUCTURED_OUTPUT_ERROR: "The candidate does not match the required structured-output schema.",
        FeedbackIssueCode.MISSING_SOURCE: "A neutral source sentence is missing from the candidate.",
        FeedbackIssueCode.DUPLICATE_SOURCE: "A neutral source sentence appears more than once in the candidate.",
        FeedbackIssueCode.UNKNOWN_SOURCE: "The candidate references a neutral source ID that was not supplied.",
        FeedbackIssueCode.FACT_RULE_CONFUSION: "The candidate changed a fact into a rule or a rule into a fact.",
        FeedbackIssueCode.UNSAFE_VARIABLE: "The candidate rule contains an unsafe or improperly bound variable.",
        FeedbackIssueCode.UNSUPPORTED_ARITY: "A literal has an unsupported or inconsistent argument count.",
        FeedbackIssueCode.PREDICATE_ARITY_CONFLICT: "The same predicate is used with conflicting arities.",
        FeedbackIssueCode.INVALID_IDENTIFIER: "A candidate identifier does not satisfy the restricted AST grammar.",
        FeedbackIssueCode.INVALID_POLARITY: "A candidate literal has an invalid polarity representation.",
        FeedbackIssueCode.INVALID_TERM: "A candidate literal contains an invalid term.",
        FeedbackIssueCode.MALFORMED_RULE_BODY: "A rule body does not match the restricted conjunctive-rule schema.",
        FeedbackIssueCode.MALFORMED_RULE_HEAD: "A rule head does not match the restricted rule schema.",
        FeedbackIssueCode.UNSUPPORTED_STRUCTURE: "The candidate cannot be represented by the restricted typed AST.",
    }
    return ValidationIssue(
        issue_code=code,
        source_id=source_id,
        ast_path=path[:160] or "$",
        description=(description or descriptions.get(code, "The candidate is invalid."))[:240],
        retryable=retryable,
    )
