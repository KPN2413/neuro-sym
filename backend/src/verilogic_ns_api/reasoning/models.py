from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"
QUALIFIED_IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$"
PREDICATE_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
VARIABLE_PATTERN = r"^[A-Z][A-Za-z0-9_]{0,63}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
QualifiedIdentifier = Annotated[str, Field(pattern=QUALIFIED_IDENTIFIER_PATTERN)]
EntityIdentifier = Annotated[str, Field(pattern=IDENTIFIER_PATTERN)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceStatement(StrictModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    text: str = Field(min_length=1, max_length=10000)


class Entity(StrictModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    label: str = Field(min_length=1, max_length=256)
    type: str | None = Field(default=None, pattern=QUALIFIED_IDENTIFIER_PATTERN)


class PredicateDefinition(StrictModel):
    name: str = Field(pattern=PREDICATE_PATTERN)
    arity: Literal[1, 2]
    argument_types: tuple[QualifiedIdentifier, ...] | None = None

    @model_validator(mode="after")
    def validate_argument_types(self) -> Self:
        if self.argument_types is not None and len(self.argument_types) != self.arity:
            raise ValueError("argument_types length must equal predicate arity")
        return self


class EntityTerm(StrictModel):
    kind: Literal["entity"]
    id: str = Field(pattern=IDENTIFIER_PATTERN)


class VariableTerm(StrictModel):
    kind: Literal["variable"]
    name: str = Field(pattern=VARIABLE_PATTERN)


Term = Annotated[EntityTerm | VariableTerm, Field(discriminator="kind")]


class GroundLiteral(StrictModel):
    predicate: str = Field(pattern=PREDICATE_PATTERN)
    arguments: tuple[EntityTerm, ...] = Field(min_length=1, max_length=2)
    negated: bool
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)


class RuleLiteral(StrictModel):
    predicate: str = Field(pattern=PREDICATE_PATTERN)
    arguments: tuple[Term, ...] = Field(min_length=1, max_length=2)
    negated: bool
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)


class VariableDefinition(StrictModel):
    name: str = Field(pattern=VARIABLE_PATTERN)
    type: str | None = Field(default=None, pattern=QUALIFIED_IDENTIFIER_PATTERN)


class Rule(StrictModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    variables: tuple[VariableDefinition, ...] = Field(max_length=64)
    body: tuple[RuleLiteral, ...] = Field(min_length=1, max_length=64)
    head: RuleLiteral
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)


class ParserMetadata(StrictModel):
    parser_name: str = Field(pattern=QUALIFIED_IDENTIFIER_PATTERN)
    parser_version: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$"
    )
    model_id: str | None = Field(default=None, min_length=1, max_length=256)
    confidence: float | None = Field(default=None, ge=0, le=1)
    warnings: tuple[str, ...] = Field(default=(), max_length=100)


class Theory(StrictModel):
    schema_version: Literal["1.0"]
    theory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_statements: tuple[SourceStatement, ...] = Field(min_length=1, max_length=10000)
    entities: tuple[Entity, ...] = Field(min_length=1, max_length=10000)
    predicates: tuple[PredicateDefinition, ...] = Field(min_length=1, max_length=1000)
    facts: tuple[GroundLiteral, ...] = Field(max_length=100000)
    rules: tuple[Rule, ...] = Field(max_length=10000)
    query: GroundLiteral
    parser_metadata: ParserMetadata | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        sources = _unique_by(self.source_statements, "id", "source statement")
        entities = _unique_by(self.entities, "id", "entity")
        predicates = _unique_by(self.predicates, "name", "predicate")
        _unique_by(self.rules, "id", "rule")

        for fact in self.facts:
            _validate_literal(
                fact,
                predicates=predicates,
                entities=entities,
                variables={},
                sources=sources,
            )
        _validate_literal(
            self.query,
            predicates=predicates,
            entities=entities,
            variables={},
            sources=sources,
        )

        for rule in self.rules:
            if rule.source_id not in sources:
                raise ValueError(f"rule {rule.id!r} references unknown source {rule.source_id!r}")
            variables = _unique_by(rule.variables, "name", f"variable in rule {rule.id}")
            body_variables: set[str] = set()
            for literal in rule.body:
                _validate_literal(
                    literal,
                    predicates=predicates,
                    entities=entities,
                    variables=variables,
                    sources=sources,
                )
                body_variables.update(_literal_variables(literal))
            _validate_literal(
                rule.head,
                predicates=predicates,
                entities=entities,
                variables=variables,
                sources=sources,
            )
            unbound = _literal_variables(rule.head) - body_variables
            if unbound:
                names = ", ".join(sorted(unbound))
                raise ValueError(f"rule {rule.id!r} has unbound head variables: {names}")
        return self


def _unique_by(
    items: tuple[BaseModel, ...], field_name: str, description: str
) -> dict[str, BaseModel]:
    result: dict[str, BaseModel] = {}
    for item in items:
        key = str(getattr(item, field_name))
        if key in result:
            raise ValueError(f"duplicate {description} identifier {key!r}")
        result[key] = item
    return result


def _literal_variables(literal: RuleLiteral) -> set[str]:
    return {term.name for term in literal.arguments if isinstance(term, VariableTerm)}


def _validate_literal(
    literal: GroundLiteral | RuleLiteral,
    *,
    predicates: dict[str, BaseModel],
    entities: dict[str, BaseModel],
    variables: dict[str, BaseModel],
    sources: dict[str, BaseModel],
) -> None:
    predicate = predicates.get(literal.predicate)
    if not isinstance(predicate, PredicateDefinition):
        raise ValueError(f"literal references undeclared predicate {literal.predicate!r}")
    if len(literal.arguments) != predicate.arity:
        raise ValueError(
            f"predicate {literal.predicate!r} expects arity {predicate.arity}, "
            f"received {len(literal.arguments)}"
        )
    if literal.source_id not in sources:
        raise ValueError(f"literal references unknown source {literal.source_id!r}")

    for index, term in enumerate(literal.arguments):
        actual_type: str | None
        if isinstance(term, EntityTerm):
            entity = entities.get(term.id)
            if not isinstance(entity, Entity):
                raise ValueError(f"literal references undeclared entity {term.id!r}")
            actual_type = entity.type
        else:
            variable = variables.get(term.name)
            if not isinstance(variable, VariableDefinition):
                raise ValueError(f"literal references undeclared variable {term.name!r}")
            actual_type = variable.type
        if predicate.argument_types is not None:
            expected_type = predicate.argument_types[index]
            if actual_type != expected_type:
                raise ValueError(
                    f"type mismatch for {literal.predicate!r} argument {index + 1}: "
                    f"expected {expected_type!r}, received {actual_type!r}"
                )


class ReasoningStatus(StrEnum):
    ENTAILED = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"
    INCONSISTENT = "INCONSISTENT"


class CanonicalLiteral(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predicate: str = Field(pattern=PREDICATE_PATTERN)
    arguments: tuple[EntityIdentifier, ...] = Field(min_length=1, max_length=2)
    negated: bool

    def sort_key(self) -> tuple[str, int, bool, tuple[str, ...]]:
        return (self.predicate, len(self.arguments), self.negated, self.arguments)

    def opposite(self) -> CanonicalLiteral:
        return self.model_copy(update={"negated": not self.negated})


class SubstitutionBinding(StrictModel):
    variable: str = Field(pattern=VARIABLE_PATTERN)
    entity: str = Field(pattern=IDENTIFIER_PATTERN)


class SourceFactNode(StrictModel):
    node_type: Literal["source_fact"] = "source_fact"
    node_id: str = Field(pattern=SHA256_PATTERN)
    literal: CanonicalLiteral
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_text: str = Field(min_length=1, max_length=10000)
    depth: Literal[0] = 0


class RuleApplicationNode(StrictModel):
    node_type: Literal["rule_application"] = "rule_application"
    node_id: str = Field(pattern=SHA256_PATTERN)
    rule_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_text: str = Field(min_length=1, max_length=10000)
    substitution: tuple[SubstitutionBinding, ...] = Field(max_length=64)
    premise_node_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    conclusion: CanonicalLiteral
    depth: int = Field(ge=1)


class DerivedLiteralNode(StrictModel):
    node_type: Literal["derived_literal"] = "derived_literal"
    node_id: str = Field(pattern=SHA256_PATTERN)
    literal: CanonicalLiteral
    rule_application_node_id: str = Field(pattern=SHA256_PATTERN)
    depth: int = Field(ge=1)


ProofNode = Annotated[
    SourceFactNode | RuleApplicationNode | DerivedLiteralNode, Field(discriminator="node_type")
]


class ProofDAG(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    theory_hash: str = Field(pattern=SHA256_PATTERN)
    query: CanonicalLiteral
    status: ReasoningStatus
    support_root_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    opposition_root_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    nodes: tuple[ProofNode, ...] = Field(max_length=100_000)
    proof_hash: str = Field(pattern=SHA256_PATTERN)


class ReasoningResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    theory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: ReasoningStatus
    query: CanonicalLiteral
    closure_contains_conflicts: bool
    conflict_count: int = Field(ge=0)
    proof: ProofDAG


class ReasoningLimitSnapshot(StrictModel):
    max_derived_literals: int = Field(gt=0)
    max_rule_firings: int = Field(gt=0)
    max_rounds: int = Field(gt=0)
    max_proof_nodes: int = Field(gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)


class ReasoningTelemetry(StrictModel):
    initial_fact_count: int = Field(ge=0)
    derived_fact_count: int = Field(ge=0)
    total_closure_size: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    rounds: int = Field(ge=0)
    rule_instances_considered: int = Field(ge=0)
    successful_rule_firings: int = Field(ge=0)
    duplicate_conclusions: int = Field(ge=0)
    maximum_proof_depth: int = Field(ge=0)
    execution_duration_ms: float = Field(ge=0)
    resource_limits: ReasoningLimitSnapshot


class ReasoningOutput(StrictModel):
    result: ReasoningResult
    telemetry: ReasoningTelemetry


class ClosureEntry(StrictModel):
    literal: CanonicalLiteral
    depth: int = Field(ge=0)


class SaturationOutput(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    theory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    theory_hash: str = Field(pattern=SHA256_PATTERN)
    closure: tuple[ClosureEntry, ...]
    conflicts: tuple[CanonicalLiteral, ...]
    telemetry: ReasoningTelemetry


class ProofVerificationResult(StrictModel):
    valid: Literal[True] = True
    status: ReasoningStatus
    proof_hash: str = Field(pattern=SHA256_PATTERN)
    node_count: int = Field(ge=0)


def canonical_literal(literal: GroundLiteral) -> CanonicalLiteral:
    return CanonicalLiteral(
        predicate=literal.predicate,
        arguments=tuple(argument.id for argument in literal.arguments),
        negated=literal.negated,
    )


def canonical_json(value: BaseModel | dict[str, object] | list[object] | tuple[object, ...]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_payload(value: BaseModel | dict[str, object] | list[object] | tuple[object, ...]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_theory_payload(theory: Theory) -> dict[str, object]:
    payload = theory.model_dump(mode="json")
    payload["source_statements"] = sorted(payload["source_statements"], key=lambda item: item["id"])
    payload["entities"] = sorted(payload["entities"], key=lambda item: item["id"])
    payload["predicates"] = sorted(payload["predicates"], key=lambda item: item["name"])
    payload["facts"] = sorted(
        payload["facts"],
        key=lambda item: (
            item["predicate"],
            item["negated"],
            tuple(term["id"] for term in item["arguments"]),
            item["source_id"],
        ),
    )
    for rule in payload["rules"]:
        rule["variables"] = sorted(rule["variables"], key=lambda item: item["name"])
        rule["body"] = sorted(rule["body"], key=_serialized_rule_literal_key)
    payload["rules"] = sorted(payload["rules"], key=lambda item: item["id"])
    return payload


def _serialized_rule_literal_key(item: dict[str, object]) -> tuple[object, ...]:
    arguments = tuple(
        (term["kind"], term.get("id", term.get("name"))) for term in item["arguments"]
    )
    return (item["predicate"], item["negated"], arguments, item["source_id"])


def theory_hash(theory: Theory) -> str:
    return sha256_payload(canonical_theory_payload(theory))
