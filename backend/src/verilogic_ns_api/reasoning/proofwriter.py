from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from verilogic_ns_api.datasets.acquisition import PROOFWRITER_VERSION
from verilogic_ns_api.datasets.proofwriter import (
    ProofWriterSource,
    map_gold_label,
    stable_example_id,
)
from verilogic_ns_api.reasoning.configuration import FormalRepresentationError
from verilogic_ns_api.reasoning.engine import ForwardChainingEngine
from verilogic_ns_api.reasoning.models import (
    Entity,
    EntityTerm,
    GroundLiteral,
    PredicateDefinition,
    ReasoningStatus,
    Rule,
    RuleLiteral,
    SourceStatement,
    Theory,
    VariableDefinition,
    VariableTerm,
    sha256_payload,
)
from verilogic_ns_api.reasoning.verifier import ProofVerifier
from verilogic_ns_api.research.models import GoldLabel, Split, WorldAssumption

_VARIABLES = {"someone": "Someone", "something": "Something"}
_SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_SAFE_PREDICATE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class _Token:
    kind: Literal["lparen", "rparen", "arrow", "string"]
    value: str
    position: int


@dataclass(frozen=True)
class FormalExample:
    example_id: str
    theory: Theory
    gold_label: GoldLabel
    reasoning_depth: int


@dataclass(frozen=True)
class _Reference:
    rank: str
    theory_id: str
    question_id: str
    label: GoldLabel
    depth: int


class ProofWriterFormalParser:
    """Parse ProofWriter's formal S-expression representation, never its prose."""

    def parse_literal(self, representation: str) -> tuple[str, str, str, bool]:
        expression = _parse_representation(representation)
        if not isinstance(expression, list) or len(expression) != 4:
            raise FormalRepresentationError("Formal literal must contain exactly four strings")
        if not all(isinstance(item, str) for item in expression):
            raise FormalRepresentationError("Formal literal fields must be strings")
        subject, relation, obj, polarity = expression
        if polarity not in {"+", "-", "~"}:
            raise FormalRepresentationError(f"Unsupported literal polarity {polarity!r}")
        return subject, relation, obj, polarity != "+"

    def parse_rule(
        self, representation: str
    ) -> tuple[tuple[tuple[str, str, str, bool], ...], tuple[str, str, str, bool]]:
        expression = _parse_representation(representation)
        if (
            not isinstance(expression, list)
            or len(expression) != 3
            or expression[1] != "->"
            or not isinstance(expression[0], list)
            or not isinstance(expression[2], list)
        ):
            raise FormalRepresentationError("Formal rule must have the shape ((body) -> head)")
        body_raw = expression[0]
        if not body_raw:
            raise FormalRepresentationError("Formal rule body must not be empty")
        body = tuple(_literal_from_expression(item) for item in body_raw)
        head = _literal_from_expression(expression[2])
        return body, head

    def convert_record_question(
        self,
        record: Mapping[str, Any],
        question_id: str,
        *,
        variant: str,
        split: Split,
    ) -> FormalExample:
        theory_id_raw = _required_string(record, "id")
        theory_id = _safe_identifier(theory_id_raw, prefix="theory")
        triples = _required_mapping(record, "triples")
        rules_raw = _required_mapping(record, "rules")
        questions = _required_mapping(record, "questions")
        question = questions.get(question_id)
        if not isinstance(question, Mapping):
            raise FormalRepresentationError(f"Unknown ProofWriter question {question_id!r}")

        parsed_facts: list[tuple[str, str, str, bool, str, str]] = []
        parsed_rules: list[
            tuple[
                str,
                str,
                tuple[tuple[str, str, str, bool], ...],
                tuple[str, str, str, bool],
            ]
        ] = []
        raw_entity_names: set[str] = set()
        predicate_arities: dict[str, int] = {}

        for raw_id, value in sorted(triples.items()):
            if not isinstance(raw_id, str) or not isinstance(value, Mapping):
                raise FormalRepresentationError("Invalid ProofWriter triple entry")
            parsed = self.parse_literal(_required_string(value, "representation"))
            source_id = _safe_identifier(raw_id, prefix="fact")
            source_text = _required_string(value, "text")
            parsed_facts.append((*parsed, source_id, source_text))
            _collect_signature(parsed, raw_entity_names, predicate_arities, rule_context=False)

        for raw_id, value in sorted(rules_raw.items()):
            if not isinstance(raw_id, str) or not isinstance(value, Mapping):
                raise FormalRepresentationError("Invalid ProofWriter rule entry")
            body, head = self.parse_rule(_required_string(value, "representation"))
            source_id = _safe_identifier(raw_id, prefix="rule")
            source_text = _required_string(value, "text")
            parsed_rules.append((source_id, source_text, body, head))
            for item in (*body, head):
                _collect_signature(item, raw_entity_names, predicate_arities, rule_context=True)

        query_representation = _required_string(question, "representation")
        parsed_query = self.parse_literal(query_representation)
        _collect_signature(parsed_query, raw_entity_names, predicate_arities, rule_context=False)
        entity_ids = _make_identifier_map(raw_entity_names, prefix="entity")
        predicate_ids = _make_predicate_map(predicate_arities)

        source_statements: list[SourceStatement] = []
        facts: list[GroundLiteral] = []
        for subject, relation, obj, negated, source_id, source_text in parsed_facts:
            source_statements.append(SourceStatement(id=source_id, text=source_text))
            predicate, arguments = _convert_literal_parts(
                subject,
                relation,
                obj,
                entity_ids=entity_ids,
                predicate_ids=predicate_ids,
                variables={},
            )
            facts.append(
                GroundLiteral(
                    predicate=predicate,
                    arguments=tuple(
                        EntityTerm(kind="entity", id=entity_ids[item]) for item in arguments
                    ),
                    negated=negated,
                    source_id=source_id,
                )
            )

        rules: list[Rule] = []
        for rule_id, source_text, body_raw, head_raw in parsed_rules:
            source_statements.append(SourceStatement(id=rule_id, text=source_text))
            raw_variables = sorted(
                {
                    value
                    for literal in (*body_raw, head_raw)
                    for value in _literal_argument_names(literal)
                    if value in _VARIABLES
                }
            )
            variables = {value: _VARIABLES[value] for value in raw_variables}
            body = tuple(
                _convert_rule_literal(
                    literal,
                    source_id=rule_id,
                    entity_ids=entity_ids,
                    predicate_ids=predicate_ids,
                    variables=variables,
                )
                for literal in body_raw
            )
            head = _convert_rule_literal(
                head_raw,
                source_id=rule_id,
                entity_ids=entity_ids,
                predicate_ids=predicate_ids,
                variables=variables,
            )
            rules.append(
                Rule(
                    id=rule_id,
                    variables=tuple(
                        VariableDefinition(name=name) for name in sorted(variables.values())
                    ),
                    body=body,
                    head=head,
                    source_id=rule_id,
                )
            )

        query_source_id = _safe_identifier(question_id, prefix="query")
        source_statements.append(
            SourceStatement(id=query_source_id, text=_required_string(question, "question"))
        )
        query_predicate, query_arguments = _convert_literal_parts(
            parsed_query[0],
            parsed_query[1],
            parsed_query[2],
            entity_ids=entity_ids,
            predicate_ids=predicate_ids,
            variables={},
        )
        theory = Theory(
            schema_version="1.0",
            theory_id=theory_id,
            source_statements=tuple(source_statements),
            entities=tuple(
                Entity(id=entity_id, label=raw_name)
                for raw_name, entity_id in sorted(entity_ids.items(), key=lambda item: item[1])
            ),
            predicates=tuple(
                PredicateDefinition(name=predicate_ids[key], arity=arity)
                for key, arity in sorted(
                    predicate_arities.items(), key=lambda item: predicate_ids[item[0]]
                )
            ),
            facts=tuple(facts),
            rules=tuple(rules),
            query=GroundLiteral(
                predicate=query_predicate,
                arguments=tuple(
                    EntityTerm(kind="entity", id=entity_ids[item]) for item in query_arguments
                ),
                negated=parsed_query[3],
                source_id=query_source_id,
            ),
            parser_metadata={
                "parser_name": "proofwriter.formal",
                "parser_version": "1.0",
            },
        )
        raw_label = question.get("answer")
        gold = map_gold_label(
            raw_label,
            world_assumption=WorldAssumption.OPEN,
            question=question,
        )
        depth = _question_depth(question)
        return FormalExample(
            example_id=stable_example_id(
                dataset_version=PROOFWRITER_VERSION,
                world_assumption=WorldAssumption.OPEN,
                variant=variant,
                theory_id=theory_id_raw,
                question_id=question_id,
            ),
            theory=theory,
            gold_label=gold,
            reasoning_depth=depth,
        )


def select_conformance_examples(
    data_source: Path,
    *,
    variant: str = "depth-5",
    split: Split = Split.DEVELOPMENT,
    depths: tuple[int, ...] = (0, 1, 2, 3, 5),
    per_cell: int = 20,
    seed: int = 20260713,
    example_ids: set[str] | None = None,
) -> tuple[FormalExample, ...]:
    if split is Split.TEST:
        raise FormalRepresentationError("Formal conformance refuses the test split")
    source = ProofWriterSource(data_source)
    member = source.member_for(WorldAssumption.OPEN, variant, split)
    references: defaultdict[tuple[int, GoldLabel], list[_Reference]] = defaultdict(list)
    wanted_by_record: defaultdict[str, set[str]] = defaultdict(set)
    selected_ids = example_ids or set()

    with source.open_text(member) as stream:
        for raw_line in stream:
            record = json.loads(raw_line)
            if not isinstance(record, Mapping):
                raise FormalRepresentationError("ProofWriter record must be an object")
            theory_id = _required_string(record, "id")
            questions = _required_mapping(record, "questions")
            for question_id, question in questions.items():
                if not isinstance(question_id, str) or not isinstance(question, Mapping):
                    raise FormalRepresentationError("Invalid ProofWriter question entry")
                example_id = stable_example_id(
                    dataset_version=PROOFWRITER_VERSION,
                    world_assumption=WorldAssumption.OPEN,
                    variant=variant,
                    theory_id=theory_id,
                    question_id=question_id,
                )
                if selected_ids:
                    if example_id in selected_ids:
                        wanted_by_record[theory_id].add(question_id)
                    continue
                depth = _question_depth(question)
                if depth not in depths:
                    continue
                label = map_gold_label(
                    question.get("answer"),
                    world_assumption=WorldAssumption.OPEN,
                    question=question,
                )
                rank = hashlib.sha256(f"{seed}|{example_id}".encode()).hexdigest()
                references[(depth, label)].append(
                    _Reference(rank, theory_id, question_id, label, depth)
                )

    if selected_ids:
        found = {
            stable_example_id(
                dataset_version=PROOFWRITER_VERSION,
                world_assumption=WorldAssumption.OPEN,
                variant=variant,
                theory_id=theory_id,
                question_id=question_id,
            )
            for theory_id, question_ids in wanted_by_record.items()
            for question_id in question_ids
        }
        missing = selected_ids - found
        if missing:
            raise FormalRepresentationError(
                f"Selection contains {len(missing)} IDs not present in the formal split"
            )
    else:
        for depth in depths:
            for label in GoldLabel:
                cell = sorted(references[(depth, label)], key=lambda item: item.rank)
                if len(cell) < per_cell:
                    raise FormalRepresentationError(
                        f"Conformance cell depth={depth}, label={label.value} has "
                        f"{len(cell)} examples; {per_cell} required"
                    )
                for reference in cell[:per_cell]:
                    wanted_by_record[reference.theory_id].add(reference.question_id)

    parser = ProofWriterFormalParser()
    examples: list[FormalExample] = []
    with source.open_text(member) as stream:
        for raw_line in stream:
            record = json.loads(raw_line)
            theory_id = _required_string(record, "id")
            for question_id in sorted(wanted_by_record.get(theory_id, ())):
                examples.append(
                    parser.convert_record_question(
                        record,
                        question_id,
                        variant=variant,
                        split=split,
                    )
                )
    return tuple(sorted(examples, key=lambda item: item.example_id))


def run_conformance(
    examples: tuple[FormalExample, ...],
    *,
    engine: ForwardChainingEngine | None = None,
    verifier: ProofVerifier | None = None,
) -> dict[str, object]:
    engine = engine or ForwardChainingEngine()
    verifier = verifier or ProofVerifier()
    correct = 0
    verified = 0
    mismatches: list[dict[str, object]] = []
    distribution: Counter[str] = Counter()
    label_totals: Counter[str] = Counter()
    label_correct: Counter[str] = Counter()
    depth_totals: Counter[int] = Counter()
    depth_correct: Counter[int] = Counter()
    status_map = {
        GoldLabel.ENTAILED: ReasoningStatus.ENTAILED,
        GoldLabel.CONTRADICTED: ReasoningStatus.CONTRADICTED,
        GoldLabel.UNKNOWN: ReasoningStatus.UNKNOWN,
    }
    total_rule_firings = 0
    total_rule_instances = 0
    total_initial_facts = 0
    total_derived_facts = 0
    total_closure_size = 0
    total_conflicts = 0
    total_duplicate_conclusions = 0
    total_rounds = 0
    maximum_depth = 0
    total_duration_ms = 0.0
    for example in examples:
        outcome = engine.reason(example.theory)
        expected = status_map[example.gold_label]
        distribution[f"depth={example.reasoning_depth}|label={example.gold_label.value}"] += 1
        label_totals[example.gold_label.value] += 1
        depth_totals[example.reasoning_depth] += 1
        if outcome.result.status is expected:
            correct += 1
            label_correct[example.gold_label.value] += 1
            depth_correct[example.reasoning_depth] += 1
        else:
            mismatches.append(
                {
                    "example_hash": hashlib.sha256(example.example_id.encode()).hexdigest(),
                    "depth": example.reasoning_depth,
                    "expected": expected.value,
                    "observed": outcome.result.status.value,
                }
            )
        verifier.verify_result(example.theory, outcome.result)
        verified += 1
        total_initial_facts += outcome.telemetry.initial_fact_count
        total_derived_facts += outcome.telemetry.derived_fact_count
        total_closure_size += outcome.telemetry.total_closure_size
        total_conflicts += outcome.telemetry.conflict_count
        total_rule_instances += outcome.telemetry.rule_instances_considered
        total_rule_firings += outcome.telemetry.successful_rule_firings
        total_duplicate_conclusions += outcome.telemetry.duplicate_conclusions
        total_rounds += outcome.telemetry.rounds
        maximum_depth = max(maximum_depth, outcome.telemetry.maximum_proof_depth)
        total_duration_ms += outcome.telemetry.execution_duration_ms

    count = len(examples)
    return {
        "schema_version": "1.0",
        "experiment": "oracle-structure-symbolic-ceiling",
        "example_count": count,
        "selection_hash": sha256_payload(sorted(item.example_id for item in examples)),
        "distribution": dict(sorted(distribution.items())),
        "per_label": {
            label: {
                "correct": label_correct[label],
                "total": total,
                "accuracy": label_correct[label] / total,
            }
            for label, total in sorted(label_totals.items())
        },
        "per_depth": {
            str(depth): {
                "correct": depth_correct[depth],
                "total": total,
                "accuracy": depth_correct[depth] / total,
            }
            for depth, total in sorted(depth_totals.items())
        },
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
        "invalid_prediction_count": 0,
        "proofs_verified": verified,
        "proof_verification_rate": verified / count if count else 0.0,
        "proof_verification_failures": count - verified,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "total_initial_fact_count": total_initial_facts,
        "total_derived_fact_count": total_derived_facts,
        "total_closure_size": total_closure_size,
        "total_conflict_count": total_conflicts,
        "total_rule_instances_considered": total_rule_instances,
        "total_successful_rule_firings": total_rule_firings,
        "total_duplicate_conclusions": total_duplicate_conclusions,
        "total_rounds": total_rounds,
        "maximum_proof_depth": maximum_depth,
        "total_execution_duration_ms": total_duration_ms,
        "gold_labels_available_only_to_evaluator": True,
        "natural_language_parsed": False,
    }


def _tokenize(text: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == "(":
            tokens.append(_Token("lparen", char, index))
            index += 1
            continue
        if char == ")":
            tokens.append(_Token("rparen", char, index))
            index += 1
            continue
        if text.startswith("->", index):
            tokens.append(_Token("arrow", "->", index))
            index += 2
            continue
        if char == '"':
            end = index + 1
            escaped = False
            while end < len(text):
                current = text[end]
                if current == '"' and not escaped:
                    break
                escaped = current == "\\" and not escaped
                if current != "\\":
                    escaped = False
                end += 1
            if end >= len(text):
                raise FormalRepresentationError(f"Unterminated string at position {index}")
            raw = text[index : end + 1]
            value = json.loads(raw)
            if not isinstance(value, str):
                raise FormalRepresentationError("Formal string token did not decode to text")
            tokens.append(_Token("string", value, index))
            index = end + 1
            continue
        raise FormalRepresentationError(f"Unexpected token at position {index}")
    return tuple(tokens)


def _parse_representation(text: str) -> list[object]:
    tokens = _tokenize(text)
    position = 0

    def parse_expression() -> object:
        nonlocal position
        if position >= len(tokens):
            raise FormalRepresentationError("Unexpected end of formal representation")
        token = tokens[position]
        position += 1
        if token.kind == "string":
            return token.value
        if token.kind == "arrow":
            return "->"
        if token.kind != "lparen":
            raise FormalRepresentationError(f"Unexpected token at position {token.position}")
        values: list[object] = []
        while position < len(tokens) and tokens[position].kind != "rparen":
            values.append(parse_expression())
        if position >= len(tokens):
            raise FormalRepresentationError("Unclosed parenthesis in formal representation")
        position += 1
        return values

    expression = parse_expression()
    if position != len(tokens):
        raise FormalRepresentationError("Trailing tokens in formal representation")
    if not isinstance(expression, list):
        raise FormalRepresentationError("Formal representation must be parenthesized")
    return expression


def _literal_from_expression(expression: object) -> tuple[str, str, str, bool]:
    if not isinstance(expression, list) or len(expression) != 4:
        raise FormalRepresentationError("Rule literal must contain four string fields")
    if not all(isinstance(item, str) for item in expression):
        raise FormalRepresentationError("Rule literal fields must be strings")
    subject, relation, obj, polarity = expression
    if polarity not in {"+", "-", "~"}:
        raise FormalRepresentationError(f"Unsupported literal polarity {polarity!r}")
    return subject, relation, obj, polarity != "+"


def _collect_signature(
    literal: tuple[str, str, str, bool],
    entities: set[str],
    predicate_arities: dict[str, int],
    *,
    rule_context: bool,
) -> None:
    subject, relation, obj, _ = literal
    predicate_raw = obj if relation == "is" else relation
    arity = 1 if relation == "is" else 2
    previous = predicate_arities.setdefault(predicate_raw, arity)
    if previous != arity:
        raise FormalRepresentationError(
            f"ProofWriter predicate {predicate_raw!r} appears with conflicting arities"
        )
    if not (rule_context and subject in _VARIABLES):
        entities.add(subject)
    if arity == 2 and not (rule_context and obj in _VARIABLES):
        entities.add(obj)


def _literal_argument_names(literal: tuple[str, str, str, bool]) -> tuple[str, ...]:
    subject, relation, obj, _ = literal
    return (subject,) if relation == "is" else (subject, obj)


def _convert_rule_literal(
    literal: tuple[str, str, str, bool],
    *,
    source_id: str,
    entity_ids: Mapping[str, str],
    predicate_ids: Mapping[str, str],
    variables: Mapping[str, str],
) -> RuleLiteral:
    subject, relation, obj, negated = literal
    predicate, arguments = _convert_literal_parts(
        subject,
        relation,
        obj,
        entity_ids=entity_ids,
        predicate_ids=predicate_ids,
        variables=variables,
    )
    terms = tuple(
        VariableTerm(kind="variable", name=variables[value])
        if value in variables
        else EntityTerm(kind="entity", id=entity_ids[value])
        for value in arguments
    )
    return RuleLiteral(
        predicate=predicate,
        arguments=terms,
        negated=negated,
        source_id=source_id,
    )


def _convert_literal_parts(
    subject: str,
    relation: str,
    obj: str,
    *,
    entity_ids: Mapping[str, str],
    predicate_ids: Mapping[str, str],
    variables: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
    del variables
    predicate_raw = obj if relation == "is" else relation
    arguments = (subject,) if relation == "is" else (subject, obj)
    for value in arguments:
        if value not in entity_ids and value not in _VARIABLES:
            raise FormalRepresentationError(f"Unmapped formal argument {value!r}")
    return predicate_ids[predicate_raw], arguments


def _make_identifier_map(values: set[str], *, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for value in sorted(values):
        candidate = _normalized_identifier(value, prefix=prefix, predicate=False)
        if candidate in used:
            suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
            candidate = f"{candidate[:55]}_{suffix}"
        used.add(candidate)
        result[value] = candidate
    return result


def _make_predicate_map(arities: Mapping[str, int]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for value in sorted(arities):
        candidate = _normalized_identifier(value, prefix="predicate", predicate=True)
        if candidate in used:
            suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
            candidate = f"{candidate[:55]}_{suffix}"
        used.add(candidate)
        result[value] = candidate
    return result


def _safe_identifier(value: str, *, prefix: str) -> str:
    if _SAFE_ID.fullmatch(value):
        return value
    return _normalized_identifier(value, prefix=prefix, predicate=False)


def _normalized_identifier(value: str, *, prefix: str, predicate: bool) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", value).strip("_-")
    if predicate:
        normalized = normalized.lower()
        if not normalized or not normalized[0].isalpha():
            normalized = f"p_{normalized}"
        normalized = normalized[:64]
        if not _SAFE_PREDICATE.fullmatch(normalized):
            raise FormalRepresentationError(f"Cannot normalize predicate {value!r} safely")
        return normalized
    if not normalized or not normalized[0].isalpha():
        normalized = f"{prefix}_{normalized}"
    if len(normalized) > 64:
        suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
        normalized = f"{normalized[:55]}_{suffix}"
    if not _SAFE_ID.fullmatch(normalized):
        raise FormalRepresentationError(f"Cannot normalize identifier {value!r} safely")
    return normalized


def _required_mapping(record: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = record.get(field)
    if not isinstance(value, Mapping):
        raise FormalRepresentationError(f"Missing formal mapping field {field!r}")
    return value


def _required_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise FormalRepresentationError(f"Missing non-empty formal field {field!r}")
    return value


def _question_depth(question: Mapping[str, Any]) -> int:
    value = question.get("QDep")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise FormalRepresentationError(f"Invalid formal question depth {value!r}")
