from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from verilogic_ns_api.evaluation.metrics import compute_metrics
from verilogic_ns_api.reasoning.engine import ForwardChainingEngine
from verilogic_ns_api.reasoning.models import ReasoningStatus, Theory
from verilogic_ns_api.reasoning.proofwriter import FormalExample, select_conformance_examples
from verilogic_ns_api.reasoning.verifier import ProofVerifier
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
from verilogic_ns_api.semantic_parsing.converter import (
    ParserSemanticError,
    SourceCoverageError,
    combine_theory_and_query,
    convert_theory_candidate,
)
from verilogic_ns_api.semantic_parsing.models import ParserOutcome, ParserStatus
from verilogic_ns_api.semantic_parsing.service import SemanticParser
from verilogic_ns_api.semantic_parsing.views import (
    PreparedTheoryView,
    assert_same_theory,
    prepare_query_view,
    prepare_theory_view,
)


class ParserEvaluationError(RuntimeError):
    pass


def run_parser_evaluation(
    *,
    examples: tuple[BenchmarkExample, ...],
    data_source: Path,
    variant: str,
    split: Split,
    parser: SemanticParser,
    output_directory: Path,
    run_id: str,
) -> dict[str, object]:
    if split is Split.TEST:
        raise ParserEvaluationError("semantic-parser evaluation refuses the test split")
    if output_directory.exists():
        raise ParserEvaluationError(f"run directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    _atomic_json(output_directory / "run-state.json", {"status": "incomplete", "run_id": run_id})

    formal_examples = select_conformance_examples(
        data_source,
        variant=variant,
        split=split,
        example_ids={item.example_id for item in examples},
    )
    gold = {item.example_id: item for item in formal_examples}
    if set(gold) != {item.example_id for item in examples}:
        raise ParserEvaluationError("formal-reference selection differs from parser selection")

    theory_views: dict[str, PreparedTheoryView] = {}
    for example in examples:
        key = example.theory_id or example.example_id
        view = prepare_theory_view(example)
        if key in theory_views:
            assert_same_theory(theory_views[key], view)
        else:
            theory_views[key] = view

    started = perf_counter()
    theory_outcomes: dict[str, ParserOutcome] = {}
    theory_candidates: dict[str, object] = {}
    for key, view in sorted(theory_views.items()):
        execution = parser.parse_theory(view.public)
        theory_outcomes[key] = execution.outcome
        if execution.candidate is not None:
            theory_candidates[key] = execution.candidate

    query_outcomes: dict[str, ParserOutcome] = {}
    query_candidates: dict[str, object] = {}
    for example in examples:
        view = prepare_query_view(example)
        execution = parser.parse_query(view.public)
        query_outcomes[example.example_id] = execution.outcome
        if execution.candidate is not None:
            query_candidates[example.example_id] = execution.candidate

    parsed_theories: dict[str, Theory] = {}
    predictions: list[PredictionRecord] = []
    parser_errors: Counter[str] = Counter()
    engine = ForwardChainingEngine()
    verifier = ProofVerifier()
    proofs_attempted = 0
    proofs_verified = 0
    for example in examples:
        key = example.theory_id or example.example_id
        theory_outcome = theory_outcomes[key]
        query_outcome = query_outcomes[example.example_id]
        error = next(
            (
                item
                for item in (theory_outcome, query_outcome)
                if item.status is not ParserStatus.PARSED
            ),
            None,
        )
        parsed: Theory | None = None
        if error is None:
            try:
                formal = gold[example.example_id]
                body = convert_theory_candidate(
                    theory_candidates[key],  # type: ignore[arg-type]
                    theory_views[key],
                    theory_id=formal.theory.theory_id,
                )
                parsed = combine_theory_and_query(
                    body,
                    query_candidates[example.example_id],  # type: ignore[arg-type]
                    prepare_query_view(example),
                )
                parsed_theories[example.example_id] = parsed
            except SourceCoverageError as exc:
                error = theory_outcome.model_copy(
                    update={"status": ParserStatus.SOURCE_COVERAGE_ERROR, "error_message": str(exc)}
                )
            except ParserSemanticError as exc:
                error = theory_outcome.model_copy(
                    update={"status": ParserStatus.SEMANTIC_INVALID, "error_message": str(exc)}
                )
            except Exception as exc:  # fail closed at the neural/symbolic boundary
                error = theory_outcome.model_copy(
                    update={
                        "status": ParserStatus.STRUCTURAL_INVALID,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                    }
                )
        if error is not None or parsed is None:
            error_name = error.status.value if error is not None else "STRUCTURAL_INVALID"
            parser_errors[error_name] += 1
            predictions.append(
                _prediction(
                    example, PredictionLabel.ERROR, error_name, theory_outcome, query_outcome
                )
            )
            continue
        reasoning = engine.reason(parsed)
        proofs_attempted += 1
        try:
            verifier.verify_result(parsed, reasoning.result)
            proofs_verified += 1
        except Exception as exc:  # a proof is never trusted merely because this engine emitted it
            parser_errors["PROOF_VERIFICATION_ERROR"] += 1
            predictions.append(
                _prediction(
                    example,
                    PredictionLabel.ERROR,
                    f"PROOF_VERIFICATION_ERROR:{type(exc).__name__}",
                    theory_outcome,
                    query_outcome,
                )
            )
            continue
        label = {
            ReasoningStatus.ENTAILED: PredictionLabel.ENTAILED,
            ReasoningStatus.CONTRADICTED: PredictionLabel.CONTRADICTED,
            ReasoningStatus.UNKNOWN: PredictionLabel.UNKNOWN,
            ReasoningStatus.INCONSISTENT: PredictionLabel.ERROR,
        }[reasoning.result.status]
        error_type = "INCONSISTENT" if label is PredictionLabel.ERROR else None
        predictions.append(_prediction(example, label, error_type, theory_outcome, query_outcome))

    report = _build_report(
        examples=examples,
        gold=gold,
        parsed=parsed_theories,
        predictions=predictions,
        theory_outcomes=theory_outcomes,
        query_outcomes=query_outcomes,
        parser_errors=parser_errors,
        proofs_attempted=proofs_attempted,
        proofs_verified=proofs_verified,
        wall_seconds=perf_counter() - started,
    )
    _atomic_json(
        output_directory / "parser-outcomes.json",
        {
            "theories": {
                key: value.model_dump(mode="json") for key, value in theory_outcomes.items()
            },
            "queries": {
                key: value.model_dump(mode="json") for key, value in query_outcomes.items()
            },
        },
    )
    _atomic_json(
        output_directory / "predictions.json",
        [item.model_dump(mode="json") for item in predictions],
    )
    _atomic_json(output_directory / "metrics.json", report)
    _atomic_json(output_directory / "run-state.json", {"status": "complete", "run_id": run_id})
    return report


def _prediction(
    example: BenchmarkExample,
    label: PredictionLabel,
    error_type: str | None,
    theory: ParserOutcome,
    query: ParserOutcome,
) -> PredictionRecord:
    usages = [
        item.usage for item in (theory, query) if item.usage is not None and not item.cache_hit
    ]
    timings = [
        item.timing for item in (theory, query) if item.timing is not None and not item.cache_hit
    ]
    request_hash = query.request_hash or theory.request_hash
    return PredictionRecord(
        run_id="semantic-parser",
        example_id=example.example_id,
        predicted_label=label,
        error_type=error_type,
        latency_ms=sum(item.total_duration_ms for item in timings),
        prompt_tokens=sum(item.input_tokens for item in usages),
        completion_tokens=sum(item.output_tokens for item in usages),
        total_tokens=sum(item.total_tokens for item in usages),
        configured_model="qwen3.5:4b-q4_K_M",
        returned_model="qwen3.5:4b-q4_K_M",
        provider_version="0.32.1",
        model_digest="2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd",
        execution_device="cpu",
        provider_total_duration_ms=sum(item.total_duration_ms for item in timings),
        provider_load_duration_ms=sum(item.load_duration_ms for item in timings),
        provider_prompt_eval_duration_ms=sum(item.prompt_eval_duration_ms for item in timings),
        provider_generation_duration_ms=sum(item.generation_duration_ms for item in timings),
        request_hash=request_hash,
        cache_hit=theory.cache_hit and query.cache_hit,
        provider_status="SUCCESS" if label is not PredictionLabel.ERROR else "ERROR",
        estimated_cost_usd=0,
        predictor_name="local-neural-semantic-parser+forward-chainer",
        predictor_version="1.0",
        timestamp=datetime.now(UTC),
    )


def _build_report(
    *,
    examples: tuple[BenchmarkExample, ...],
    gold: dict[str, FormalExample],
    parsed: dict[str, Theory],
    predictions: list[PredictionRecord],
    theory_outcomes: dict[str, ParserOutcome],
    query_outcomes: dict[str, ParserOutcome],
    parser_errors: Counter[str],
    proofs_attempted: int,
    proofs_verified: int,
    wall_seconds: float,
) -> dict[str, object]:
    statement_tp = statement_fp = statement_fn = 0
    exact_theories: set[str] = set()
    seen_theories: set[str] = set()
    exact_queries = 0
    closure_tp = closure_fp = closure_fn = 0
    predicate_tp = predicate_fp = predicate_fn = 0
    entity_tp = entity_fp = entity_fn = 0
    component_correct: Counter[str] = Counter()
    component_total: Counter[str] = Counter()
    construction_correct: Counter[str] = Counter()
    construction_total: Counter[str] = Counter()
    parsed_by_theory: dict[str, Theory] = {}
    for example in examples:
        predicted = parsed.get(example.example_id)
        if predicted is not None:
            parsed_by_theory.setdefault(example.theory_id or example.example_id, predicted)
    for example in examples:
        formal = gold[example.example_id].theory
        predicted = parsed.get(example.example_id)
        key = example.theory_id or example.example_id
        theory_prediction = parsed_by_theory.get(key)
        if key not in seen_theories:
            if theory_prediction is None:
                gold_facts, gold_rules = canonical_statement_sets(formal)
                statement_fn += len(gold_facts) + len(gold_rules)
                _record_theory_components(
                    formal,
                    None,
                    component_correct,
                    component_total,
                    construction_correct,
                    construction_total,
                )
            else:
                gold_facts, gold_rules = canonical_statement_sets(formal)
                pred_facts, pred_rules = canonical_statement_sets(theory_prediction)
                gold_statements = gold_facts | gold_rules
                pred_statements = pred_facts | pred_rules
                statement_tp += len(gold_statements & pred_statements)
                statement_fp += len(pred_statements - gold_statements)
                statement_fn += len(gold_statements - pred_statements)
                if gold_statements == pred_statements:
                    exact_theories.add(key)
                gold_predicates = {(item.name, item.arity) for item in formal.predicates}
                pred_predicates = {(item.name, item.arity) for item in theory_prediction.predicates}
                predicate_tp += len(gold_predicates & pred_predicates)
                predicate_fp += len(pred_predicates - gold_predicates)
                predicate_fn += len(gold_predicates - pred_predicates)
                gold_entities = {item.id for item in formal.entities}
                pred_entities = {item.id for item in theory_prediction.entities}
                entity_tp += len(gold_entities & pred_entities)
                entity_fp += len(pred_entities - gold_entities)
                entity_fn += len(gold_entities - pred_entities)
                _record_theory_components(
                    formal,
                    theory_prediction,
                    component_correct,
                    component_total,
                    construction_correct,
                    construction_total,
                )
            seen_theories.add(key)
        if predicted is None:
            _record_query_components(formal, None, component_correct, component_total)
            continue
        _record_query_components(formal, predicted, component_correct, component_total)
        if canonical_query(formal) == canonical_query(predicted):
            exact_queries += 1
        gold_closure = closure_keys(formal)
        pred_closure = closure_keys(predicted)
        closure_tp += len(gold_closure & pred_closure)
        closure_fp += len(pred_closure - gold_closure)
        closure_fn += len(gold_closure - pred_closure)

    metric_report = compute_metrics(examples, predictions).model_dump(mode="json")
    all_outcomes = list(theory_outcomes.values()) + list(query_outcomes.values())
    provider_usages = [item.usage for item in all_outcomes if item.usage and not item.cache_hit]
    provider_timings = [item.timing for item in all_outcomes if item.timing and not item.cache_hit]
    structured_pairs = sum(
        theory_outcomes[example.theory_id or example.example_id].status is ParserStatus.PARSED
        and query_outcomes[example.example_id].status is ParserStatus.PARSED
        for example in examples
    )
    source_coverage_passed = (
        structured_pairs - parser_errors[ParserStatus.SOURCE_COVERAGE_ERROR.value]
    )
    return {
        "schema_version": "1.0",
        "status": "complete",
        "counts": {
            "examples": len(examples),
            "unique_theories": len(theory_outcomes),
            "theory_parse_success": sum(
                item.status is ParserStatus.PARSED for item in theory_outcomes.values()
            ),
            "query_parse_success": sum(
                item.status is ParserStatus.PARSED for item in query_outcomes.values()
            ),
            "complete_valid_theories": len(parsed),
        },
        "structural_validity_rate": _ratio(len(parsed), len(examples)),
        "source_coverage_rate": _ratio(source_coverage_passed, structured_pairs),
        "semantic_validation_rate": _ratio(len(parsed), source_coverage_passed),
        "validation_funnel": {
            "structured_output": {
                "attempted": len(examples),
                "passed": structured_pairs,
                "rate": _ratio(structured_pairs, len(examples)),
            },
            "source_coverage": {
                "attempted": structured_pairs,
                "passed": source_coverage_passed,
                "rate": _ratio(source_coverage_passed, structured_pairs),
            },
            "semantic_validation": {
                "attempted": source_coverage_passed,
                "passed": len(parsed),
                "rate": _ratio(len(parsed), source_coverage_passed),
            },
        },
        "statement_semantics": _prf(statement_tp, statement_fp, statement_fn),
        "predicate_semantics": _prf(predicate_tp, predicate_fp, predicate_fn),
        "entity_semantics": _prf(entity_tp, entity_fp, entity_fn),
        "exact_theory_rate": _ratio(len(exact_theories), len(theory_outcomes)),
        "exact_query_rate": _ratio(exact_queries, len(examples)),
        "closure_semantics": _prf(closure_tp, closure_fp, closure_fn),
        "component_accuracy": {
            name: {
                "correct": component_correct[name],
                "total": total,
                "accuracy": _ratio(component_correct[name], total),
            }
            for name, total in sorted(component_total.items())
        },
        "construction_accuracy": {
            name: {
                "correct": construction_correct[name],
                "total": total,
                "accuracy": _ratio(construction_correct[name], total),
            }
            for name, total in sorted(construction_total.items())
        },
        "end_to_end": metric_report,
        "error_taxonomy": dict(sorted(parser_errors.items())),
        "proof_verification": {
            "attempted": proofs_attempted,
            "verified": proofs_verified,
            "failed": proofs_attempted - proofs_verified,
            "rate": _ratio(proofs_verified, proofs_attempted),
        },
        "efficiency": {
            "wall_seconds": wall_seconds,
            "provider_requests": sum(not item.cache_hit for item in all_outcomes),
            "cache_hits": sum(item.cache_hit for item in all_outcomes),
            "input_tokens": sum(item.input_tokens for item in provider_usages),
            "output_tokens": sum(item.output_tokens for item in provider_usages),
            "provider_duration_ms": sum(item.total_duration_ms for item in provider_timings),
            "api_cost_usd": 0.0,
            "hosted_provider_calls": 0,
        },
    }


def _record_theory_components(
    gold: Theory,
    predicted: Theory | None,
    correct: Counter[str],
    total: Counter[str],
    construction_correct: Counter[str],
    construction_total: Counter[str],
) -> None:
    predicted_facts = {item.source_id: item for item in predicted.facts} if predicted else {}
    predicted_rules = {item.source_id: item for item in predicted.rules} if predicted else {}
    for fact in gold.facts:
        candidate = predicted_facts.get(fact.source_id)
        category = f"fact_{'binary' if len(fact.arguments) == 2 else 'unary'}_"
        category += "negative" if fact.negated else "positive"
        construction_total[category] += 1
        exact = candidate is not None and canonical_literal_key(candidate) == canonical_literal_key(
            fact
        )
        construction_correct[category] += int(exact)
        _component(correct, total, "source_aligned_statement", exact)
        _component(
            correct,
            total,
            "literal_predicate",
            candidate is not None and candidate.predicate == fact.predicate,
        )
        _component(
            correct,
            total,
            "literal_arity",
            candidate is not None and len(candidate.arguments) == len(fact.arguments),
        )
        _component(
            correct,
            total,
            "literal_polarity",
            candidate is not None and candidate.negated == fact.negated,
        )
        _component(
            correct,
            total,
            "literal_arguments_ordered",
            candidate is not None
            and tuple(item.id for item in candidate.arguments)
            == tuple(item.id for item in fact.arguments),
        )
    for rule in gold.rules:
        candidate = predicted_rules.get(rule.source_id)
        categories = ["rule_conjunctive" if len(rule.body) > 1 else "rule_single_premise"]
        if any(item.negated for item in rule.body):
            categories.append("rule_negative_premise")
        if rule.head.negated:
            categories.append("rule_negative_head")
        if any(len(item.arguments) == 2 for item in (*rule.body, rule.head)):
            categories.append("rule_binary_relation")
        if len(rule.variables) > 1:
            categories.append("rule_variable_join")
        exact = candidate is not None and canonical_rule_key(candidate) == canonical_rule_key(rule)
        for category in categories:
            construction_total[category] += 1
            construction_correct[category] += int(exact)
        _component(correct, total, "source_aligned_statement", exact)
        _component(correct, total, "rule_direction", exact)
        _component(
            correct,
            total,
            "rule_body",
            candidate is not None
            and {canonical_literal_key(item) for item in candidate.body}
            == {canonical_literal_key(item) for item in rule.body},
        )
        _component(
            correct,
            total,
            "rule_head",
            candidate is not None
            and canonical_literal_key(candidate.head) == canonical_literal_key(rule.head),
        )


def _record_query_components(
    gold: Theory,
    predicted: Theory | None,
    correct: Counter[str],
    total: Counter[str],
) -> None:
    candidate = predicted.query if predicted else None
    query = gold.query
    _component(
        correct,
        total,
        "query_exact",
        candidate is not None and canonical_query(predicted) == canonical_query(gold),
    )
    _component(
        correct,
        total,
        "query_predicate",
        candidate is not None and candidate.predicate == query.predicate,
    )
    _component(
        correct,
        total,
        "query_arity",
        candidate is not None and len(candidate.arguments) == len(query.arguments),
    )
    _component(
        correct,
        total,
        "query_polarity",
        candidate is not None and candidate.negated == query.negated,
    )
    _component(
        correct,
        total,
        "query_arguments_ordered",
        candidate is not None
        and tuple(item.id for item in candidate.arguments)
        == tuple(item.id for item in query.arguments),
    )


def _component(correct: Counter[str], total: Counter[str], name: str, matched: bool) -> None:
    total[name] += 1
    correct[name] += int(matched)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


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
