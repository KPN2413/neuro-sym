from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from verilogic_ns_api.baselines.configuration import resolve_repository_path
from verilogic_ns_api.evaluation.metrics import compute_metrics
from verilogic_ns_api.reasoning.models import Theory, sha256_payload
from verilogic_ns_api.reasoning.proofwriter import select_conformance_examples
from verilogic_ns_api.research.models import (
    BenchmarkExample,
    PredictionLabel,
    PredictionRecord,
    Split,
)
from verilogic_ns_api.semantic_parsing.canonicalization import (
    canonical_literal_key,
    canonical_query,
    canonical_rule_key,
    canonical_statement_sets,
    closure_keys,
)
from verilogic_ns_api.semantic_parsing.cli import _service as phase5_service
from verilogic_ns_api.semantic_parsing.evaluation import run_parser_evaluation
from verilogic_ns_api.semantic_parsing.views import prepare_query_view
from verilogic_ns_api.validation_correction.configuration import PreparedCorrectionExperiment
from verilogic_ns_api.validation_correction.controller import ValidationCorrectionController
from verilogic_ns_api.validation_correction.feedback import validate_theory_candidate
from verilogic_ns_api.validation_correction.models import (
    ComponentDecision,
    ControllerTrace,
    CriticDecision,
    TaskKind,
)
from verilogic_ns_api.validation_correction.policy import PolicyResult, apply_policy
from verilogic_ns_api.validation_correction.raw import load_raw_phase5_candidates
from verilogic_ns_api.validation_correction.service import CorrectionTaskService


class Phase6EvaluationError(RuntimeError):
    pass


def run_correction_evaluation(
    *,
    prepared: PreparedCorrectionExperiment,
    service: CorrectionTaskService,
    output_directory: Path,
    run_id: str,
    calibration: bool,
) -> dict[str, object]:
    if output_directory.exists():
        raise Phase6EvaluationError(f"run directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    _atomic_json(output_directory / "run-state.json", {"status": "incomplete", "run_id": run_id})
    examples = (
        prepared.phase5.calibration_examples if calibration else prepared.phase5.pilot_examples
    )
    split = Split.TRAIN if calibration else Split.DEVELOPMENT
    started = perf_counter()

    p0_service = phase5_service(prepared.phase5, provider=None, replay_only=True)
    p0_report = run_parser_evaluation(
        examples=examples,
        data_source=resolve_repository_path(prepared.root, prepared.phase5.config.data_source),
        variant=prepared.phase5.config.variant,
        split=split,
        parser=p0_service,
        output_directory=output_directory / "p0-raw",
        run_id=f"{run_id}-p0",
    )
    if not calibration:
        _assert_frozen_p0(p0_report)

    raw = load_raw_phase5_candidates(prepared.phase5, calibration=calibration)
    controller = ValidationCorrectionController(service)
    theory_decisions: dict[str, ComponentDecision] = {}
    final_bodies = {}
    for key, view in sorted(raw.theory_views.items()):
        decision = controller.run_theory(
            view=view,
            raw_candidate=raw.theories[key],
            theory_id=key,
        )
        theory_decisions[key] = decision
        if decision.final_candidate is not None:
            validated = validate_theory_candidate(decision.final_candidate, view, theory_id=key)
            if validated.valid and validated.converted is not None:
                final_bodies[key] = validated.converted

    query_decisions: dict[str, ComponentDecision] = {}
    for example in examples:
        key = example.theory_id or example.example_id
        query_decisions[example.example_id] = controller.run_query(
            view=prepare_query_view(example),
            raw_candidate=raw.queries[example.example_id],
            body=final_bodies.get(key),
        )

    if service.new_call_count > prepared.config.limits.maximum_new_pilot_calls:
        raise Phase6EvaluationError("local-call budget was exceeded")

    p1 = apply_policy(
        examples=examples,
        theory_views=raw.theory_views,
        theory_decisions=theory_decisions,
        query_decisions=query_decisions,
        selective=False,
    )
    p2 = apply_policy(
        examples=examples,
        theory_views=raw.theory_views,
        theory_decisions=theory_decisions,
        query_decisions=query_decisions,
        selective=True,
    )
    p1_metrics = compute_metrics(examples, p1.predictions).model_dump(mode="json")
    p2_metrics = compute_metrics(examples, p2.predictions).model_dump(mode="json")
    p0_predictions = tuple(
        PredictionRecord.model_validate(item)
        for item in json.loads(
            (output_directory / "p0-raw" / "predictions.json").read_text(encoding="utf-8")
        )
    )
    validation_only_predictions = _validation_only_predictions(p0_predictions)
    validation_only_metrics = compute_metrics(examples, validation_only_predictions).model_dump(
        mode="json"
    )

    formal_examples = select_conformance_examples(
        resolve_repository_path(prepared.root, prepared.phase5.config.data_source),
        variant=prepared.phase5.config.variant,
        split=split,
        example_ids={item.example_id for item in examples},
    )
    gold = {item.example_id: item.theory for item in formal_examples}
    correction_metrics = _correction_metrics(theory_decisions, query_decisions)
    critic_metrics = _critic_metrics(
        examples,
        gold,
        raw.theory_views,
        theory_decisions,
        query_decisions,
    )
    ast_metrics = _ast_metrics(examples, gold, p1.parsed_theories)
    efficiency = _efficiency(theory_decisions, query_decisions, raw.cache_hits)
    comparison = _comparison_table(
        prepared.root,
        p0_report["end_to_end"],
        p1_metrics,
        p2_metrics,
        p1,
        p2,
    )
    ablation = [
        _policy_row("Raw only", p0_report["end_to_end"], proof=p0_report["proof_verification"]),
        _policy_row("Validator feedback only", validation_only_metrics),
        _policy_row("Critic plus correction", p1_metrics, proof=_proof_payload(p1)),
        _policy_row("Critic plus correction plus abstention", p2_metrics, proof=_proof_payload(p2)),
    ]
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "run_id": run_id,
        "dataset": "calibration" if calibration else "pilot",
        "p0_raw": p0_report,
        "p1_corrected_valid": {
            "metrics": p1_metrics,
            "proof_verification": _proof_payload(p1),
            "abstention_reasons": p1.abstention_reasons,
        },
        "p2_corrected_selective": {
            "metrics": p2_metrics,
            "proof_verification": _proof_payload(p2),
            "abstention_reasons": p2.abstention_reasons,
        },
        "validation_only": validation_only_metrics,
        "correction_metrics": correction_metrics,
        "critic_metrics": critic_metrics,
        "ast_metrics": ast_metrics,
        "efficiency": efficiency,
        "comparison_table": comparison,
        "correction_ablation": ablation,
        "replay_fingerprint": sha256_payload(
            {
                "p0": p0_report["end_to_end"],
                "p1": p1_metrics,
                "p2": p2_metrics,
                "correction": correction_metrics,
                "critic": critic_metrics,
                "ast": ast_metrics,
            }
        ),
        "wall_seconds": perf_counter() - started,
        "api_cost_usd": 0.0,
        "hosted_provider_calls": 0,
        "test_split": False,
    }
    _write_traces(output_directory, theory_decisions, query_decisions)
    _atomic_json(
        output_directory / "p1-predictions.json",
        [item.model_dump(mode="json") for item in p1.predictions],
    )
    _atomic_json(
        output_directory / "p2-predictions.json",
        [item.model_dump(mode="json") for item in p2.predictions],
    )
    _atomic_json(output_directory / "report.json", report)
    _atomic_json(output_directory / "run-state.json", {"status": "complete", "run_id": run_id})
    return report


def _assert_frozen_p0(report: dict[str, object]) -> None:
    metrics = report["end_to_end"]
    proof = report["proof_verification"]
    expected = {
        "accuracy": 0.1,
        "coverage": 4 / 30,
        "answered_only_accuracy": 0.75,
        "macro_f1": 0.16317016317016317,
        "errored_examples": 26,
    }
    if not isinstance(metrics, dict) or any(
        metrics.get(key) != value for key, value in expected.items()
    ):
        raise Phase6EvaluationError("P0 does not reproduce the frozen Phase 5 metrics")
    if not isinstance(proof, dict) or proof.get("verified") != 4 or proof.get("failed") != 0:
        raise Phase6EvaluationError("P0 proof verification differs from Phase 5")


def _validation_only_predictions(
    predictions: tuple[PredictionRecord, ...],
) -> tuple[PredictionRecord, ...]:
    converted = []
    for item in predictions:
        if item.predicted_label is PredictionLabel.ERROR:
            converted.append(
                item.model_copy(
                    update={
                        "predicted_label": PredictionLabel.ABSTAIN,
                        "abstention_reason": "RAW_VALIDATION_FAILED",
                        "error_type": None,
                    }
                )
            )
        else:
            converted.append(item)
    return tuple(converted)


def _proof_payload(result: PolicyResult) -> dict[str, object]:
    return {
        "attempted": result.proof_attempted,
        "verified": result.proof_verified,
        "failed": result.proof_attempted - result.proof_verified,
        "rate": result.proof_verified / result.proof_attempted if result.proof_attempted else 0,
    }


def _correction_metrics(
    theories: dict[str, ComponentDecision], queries: dict[str, ComponentDecision]
) -> dict[str, object]:
    decisions = [*theories.values(), *queries.values()]
    attempted = [item for item in decisions if item.correction_attempts]
    recovered = [item for item in attempted if item.deterministic_accepted]
    raw_invalid = sum(
        any(transition.event == "raw_validation_failed" for transition in item.transitions)
        for item in decisions
    )
    no_progress = sum(item.reliability.no_progress_detected for item in decisions)
    regressions = sum(
        any(transition.event == "raw_validation_passed" for transition in item.transitions)
        and item.correction_attempts
        and not item.deterministic_accepted
        for item in decisions
    )
    return {
        "components": len(decisions),
        "raw_invalid_components": raw_invalid,
        "correction_attempts": len(attempted),
        "structurally_recovered": sum(item.final_candidate is not None for item in attempted),
        "source_coverage_recovered": len(recovered),
        "semantically_recovered": len(recovered),
        "critic_accepted_after_correction": sum(item.selective_accepted for item in attempted),
        "correction_success_rate": len(recovered) / len(attempted) if attempted else 0,
        "no_progress_count": no_progress,
        "correction_regression_count": regressions,
    }


def _critic_metrics(
    examples: tuple[BenchmarkExample, ...],
    gold: dict[str, Theory],
    theory_views,
    theory_decisions: dict[str, ComponentDecision],
    query_decisions: dict[str, ComponentDecision],
) -> dict[str, object]:
    observations: list[tuple[CriticDecision, bool]] = []
    first_by_theory: dict[str, BenchmarkExample] = {}
    for example in examples:
        first_by_theory.setdefault(example.theory_id or example.example_id, example)
    for key, decision in theory_decisions.items():
        if decision.critic_decision is None or decision.final_candidate is None:
            continue
        reference = gold[first_by_theory[key].example_id]
        validated = validate_theory_candidate(
            decision.final_candidate, theory_views[key], theory_id=key
        )
        exact = False
        if validated.valid and validated.converted is not None:
            predicted_sets = canonical_statement_sets(
                _body_with_reference_query(validated.converted, reference)
            )
            exact = predicted_sets == canonical_statement_sets(reference)
        observations.append((decision.critic_decision, not exact))
    for example in examples:
        decision = query_decisions[example.example_id]
        if decision.critic_decision is None or decision.final_candidate is None:
            continue
        reference = gold[example.example_id]
        from verilogic_ns_api.semantic_parsing.models import CandidateQueryOutput

        try:
            candidate = CandidateQueryOutput.model_validate(decision.final_candidate)
            exact = (
                candidate.query.predicate,
                tuple(item.id for item in candidate.query.arguments),
                candidate.query.negated,
            ) == (
                reference.query.predicate,
                tuple(item.id for item in reference.query.arguments),
                reference.query.negated,
            )
        except ValueError:
            exact = False
        observations.append((decision.critic_decision, not exact))
    true_detection = sum(
        decision is CriticDecision.REVISE and error for decision, error in observations
    )
    false_rejection = sum(
        decision is CriticDecision.REVISE and not error for decision, error in observations
    )
    false_acceptance = sum(
        decision is CriticDecision.ACCEPT and error for decision, error in observations
    )
    precision = _ratio(true_detection, true_detection + false_rejection)
    recall = _ratio(true_detection, true_detection + false_acceptance)
    return {
        "evaluated_reports": len(observations),
        "true_semantic_error_detections": true_detection,
        "false_acceptances": false_acceptance,
        "false_rejections": false_rejection,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
    }


def _body_with_reference_query(body, reference: Theory) -> Theory:
    from verilogic_ns_api.semantic_parsing.converter import combine_theory_and_query
    from verilogic_ns_api.semantic_parsing.models import CandidateQueryOutput
    from verilogic_ns_api.semantic_parsing.views import PreparedQueryView

    candidate = CandidateQueryOutput.model_validate(
        {
            "query": {
                "predicate": reference.query.predicate,
                "arity": len(reference.query.arguments),
                "arguments": [item.model_dump(mode="json") for item in reference.query.arguments],
                "negated": reference.query.negated,
            }
        }
    )
    return combine_theory_and_query(
        body,
        candidate,
        PreparedQueryView(
            public=prepare_query_view_from_reference(reference),
            original_source_id=reference.query.source_id,
            text="Post-hoc comparison query.",
        ),
    )


def prepare_query_view_from_reference(reference: Theory):
    from verilogic_ns_api.semantic_parsing.models import QueryParseInput

    return QueryParseInput(
        input_hash=sha256_payload({"reference": reference.theory_id}), text="Reference query."
    )


def _ast_metrics(
    examples: tuple[BenchmarkExample, ...],
    gold: dict[str, Theory],
    parsed: dict[str, Theory],
) -> dict[str, object]:
    statement_tp = statement_fp = statement_fn = 0
    predicate_correct = polarity_correct = arity_correct = term_correct = 0
    statement_total = 0
    rule_body_correct = rule_head_correct = rule_total = 0
    exact_theories = 0
    exact_queries = 0
    closure_tp = closure_fp = closure_fn = 0
    first_by_theory: dict[str, BenchmarkExample] = {}
    predicted_by_theory: dict[str, Theory] = {}
    for example in examples:
        key = example.theory_id or example.example_id
        first_by_theory.setdefault(key, example)
        if example.example_id in parsed:
            predicted_by_theory.setdefault(key, parsed[example.example_id])
    for key, example in first_by_theory.items():
        reference = gold[example.example_id]
        prediction = predicted_by_theory.get(key)
        gold_facts, gold_rules = canonical_statement_sets(reference)
        pred_facts, pred_rules = (
            canonical_statement_sets(prediction) if prediction else (set(), set())
        )
        gold_statements = gold_facts | gold_rules
        pred_statements = pred_facts | pred_rules
        statement_tp += len(gold_statements & pred_statements)
        statement_fp += len(pred_statements - gold_statements)
        statement_fn += len(gold_statements - pred_statements)
        exact_theories += int(prediction is not None and gold_statements == pred_statements)
        pred_fact_sources = (
            {item.source_id: item for item in prediction.facts} if prediction else {}
        )
        pred_rule_sources = (
            {item.source_id: item for item in prediction.rules} if prediction else {}
        )
        for fact in reference.facts:
            candidate = pred_fact_sources.get(fact.source_id)
            statement_total += 1
            predicate_correct += int(
                candidate is not None and candidate.predicate == fact.predicate
            )
            polarity_correct += int(candidate is not None and candidate.negated == fact.negated)
            arity_correct += int(
                candidate is not None and len(candidate.arguments) == len(fact.arguments)
            )
            term_correct += int(
                candidate is not None
                and tuple(item.id for item in candidate.arguments)
                == tuple(item.id for item in fact.arguments)
            )
        for rule in reference.rules:
            candidate = pred_rule_sources.get(rule.source_id)
            statement_total += 1
            rule_total += 1
            exact_rule = candidate is not None and canonical_rule_key(
                candidate
            ) == canonical_rule_key(rule)
            predicate_correct += int(exact_rule)
            polarity_correct += int(exact_rule)
            arity_correct += int(exact_rule)
            term_correct += int(exact_rule)
            rule_body_correct += int(
                candidate is not None
                and {canonical_literal_key(item) for item in candidate.body}
                == {canonical_literal_key(item) for item in rule.body}
            )
            rule_head_correct += int(
                candidate is not None
                and canonical_literal_key(candidate.head) == canonical_literal_key(rule.head)
            )
    for example in examples:
        reference = gold[example.example_id]
        prediction = parsed.get(example.example_id)
        exact_queries += int(
            prediction is not None and canonical_query(prediction) == canonical_query(reference)
        )
        gold_closure = closure_keys(reference)
        pred_closure = closure_keys(prediction) if prediction else set()
        closure_tp += len(gold_closure & pred_closure)
        closure_fp += len(pred_closure - gold_closure)
        closure_fn += len(gold_closure - pred_closure)
    statement_precision = _ratio(statement_tp, statement_tp + statement_fp)
    statement_recall = _ratio(statement_tp, statement_tp + statement_fn)
    closure_precision = _ratio(closure_tp, closure_tp + closure_fp)
    closure_recall = _ratio(closure_tp, closure_tp + closure_fn)
    return {
        "statement_exact": {
            "true_positive": statement_tp,
            "false_positive": statement_fp,
            "false_negative": statement_fn,
            "precision": statement_precision,
            "recall": statement_recall,
            "f1": _f1(statement_precision, statement_recall),
        },
        "predicate_accuracy": _ratio(predicate_correct, statement_total),
        "polarity_accuracy": _ratio(polarity_correct, statement_total),
        "arity_accuracy": _ratio(arity_correct, statement_total),
        "term_accuracy": _ratio(term_correct, statement_total),
        "rule_body_accuracy": _ratio(rule_body_correct, rule_total),
        "rule_head_accuracy": _ratio(rule_head_correct, rule_total),
        "complete_theory_exact_match": _ratio(exact_theories, len(first_by_theory)),
        "query_exact_match": _ratio(exact_queries, len(examples)),
        "closure": {
            "true_positive": closure_tp,
            "false_positive": closure_fp,
            "false_negative": closure_fn,
            "precision": closure_precision,
            "recall": closure_recall,
            "f1": _f1(closure_precision, closure_recall),
        },
    }


def _efficiency(
    theories: dict[str, ComponentDecision],
    queries: dict[str, ComponentDecision],
    raw_cache_hits: int,
) -> dict[str, object]:
    outcomes = [
        outcome
        for decision in [*theories.values(), *queries.values()]
        for outcome in decision.task_outcomes
    ]
    new = [item for item in outcomes if not item.cache_hit]
    recovered = sum(
        item.correction_attempts and item.deterministic_accepted
        for item in [*theories.values(), *queries.values()]
    )
    return {
        "raw_phase5_cache_hits": raw_cache_hits,
        "logical_task_requests": len(outcomes),
        "new_local_calls": len(new),
        "cache_hits": sum(item.cache_hit for item in outcomes),
        "critic_calls": sum(
            item.task_kind in {TaskKind.CRITIC_THEORY, TaskKind.CRITIC_QUERY} for item in new
        ),
        "correction_calls": sum(
            item.task_kind in {TaskKind.CORRECTION_THEORY, TaskKind.CORRECTION_QUERY}
            for item in new
        ),
        "input_tokens": sum(item.input_tokens for item in new),
        "output_tokens": sum(item.output_tokens for item in new),
        "local_inference_ms": sum(item.duration_ms for item in new),
        "mean_new_calls_per_recovered_component": _ratio(len(new), recovered),
        "api_cost_usd": 0.0,
        "hosted_provider_calls": 0,
    }


def _comparison_table(
    root: Path,
    p0_metrics: dict[str, object],
    p1_metrics: dict[str, object],
    p2_metrics: dict[str, object],
    p1: PolicyResult,
    p2: PolicyResult,
) -> list[dict[str, object]]:
    direct = _latest_metrics(root / "results" / "runs", "ollama-direct-pilot-live-*")
    few = _latest_metrics(root / "results" / "runs", "ollama-few-shot-pilot-live-*")
    return [
        _policy_row("Direct local LLM", direct or {}),
        _policy_row("Few-shot local LLM", few or {}),
        _policy_row("P0 raw neural-symbolic", p0_metrics, proof={"rate": 1.0}),
        _policy_row("P1 corrected-valid", p1_metrics, proof=_proof_payload(p1)),
        _policy_row("P2 corrected-selective", p2_metrics, proof=_proof_payload(p2)),
        {
            "system": "Oracle-structure symbolic ceiling",
            "correct": 30,
            "accuracy": 1.0,
            "coverage": 1.0,
            "answered_only_accuracy": 1.0,
            "macro_f1": 1.0,
            "abstain": 0,
            "error": 0,
            "proof_verification_rate": 1.0,
        },
    ]


def _latest_metrics(directory: Path, pattern: str) -> dict[str, object] | None:
    candidates = sorted(directory.glob(f"{pattern}/metrics.json"))
    return json.loads(candidates[-1].read_text(encoding="utf-8")) if candidates else None


def _policy_row(
    name: str,
    metrics: dict[str, object],
    *,
    proof: dict[str, object] | None = None,
) -> dict[str, object]:
    total = int(metrics.get("total_examples", 0) or 0)
    accuracy = float(metrics.get("accuracy", 0) or 0)
    return {
        "system": name,
        "correct": round(total * accuracy),
        "accuracy": accuracy,
        "coverage": metrics.get("coverage"),
        "answered_only_accuracy": metrics.get("answered_only_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "abstain": metrics.get("abstained_examples", 0),
        "error": metrics.get("errored_examples", 0),
        "proof_verification_rate": proof.get("rate") if proof else None,
    }


def _write_traces(
    output_directory: Path,
    theories: dict[str, ComponentDecision],
    queries: dict[str, ComponentDecision],
) -> None:
    payload: dict[str, object] = {"theories": {}, "queries": {}}
    for group, decisions in (("theories", theories), ("queries", queries)):
        target = payload[group]
        assert isinstance(target, dict)
        for key, decision in decisions.items():
            final_state = decision.transitions[-1].to_state
            trace = ControllerTrace(
                trace_id=sha256_payload(
                    {
                        "component": decision.component_type.value,
                        "input_hash": decision.input_hash,
                        "transitions": [
                            item.model_dump(mode="json") for item in decision.transitions
                        ],
                    }
                ),
                component_type=decision.component_type,
                input_hash=decision.input_hash,
                transitions=decision.transitions,
                raw_candidate_hash=decision.raw_candidate_hash,
                final_candidate_hash=decision.final_candidate_hash,
                correction_attempts=decision.correction_attempts,
                final_state=final_state,
                abstention_reason=decision.abstention_reason,
                error_type=decision.error_type,
                created_at=datetime.now(UTC),
            )
            target[sha256_payload({"key": key})] = trace.model_dump(mode="json")
    _atomic_json(output_directory / "controller-traces.json", payload)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
