from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from verilogic_ns_api.reasoning.models import (
    Entity,
    EntityTerm,
    GroundLiteral,
    ParserMetadata,
    PredicateDefinition,
    Rule,
    RuleLiteral,
    SourceStatement,
    Theory,
)
from verilogic_ns_api.semantic_parsing.models import (
    CandidateFactStatement,
    CandidateQueryOutput,
    CandidateRuleLiteral,
    CandidateRuleStatement,
    CandidateTheoryOutput,
)
from verilogic_ns_api.semantic_parsing.views import PreparedQueryView, PreparedTheoryView


class SourceCoverageError(ValueError):
    pass


class ParserSemanticError(ValueError):
    pass


@dataclass(frozen=True)
class ConvertedTheoryBody:
    theory_id: str
    sources: tuple[SourceStatement, ...]
    entities: tuple[Entity, ...]
    predicates: tuple[PredicateDefinition, ...]
    facts: tuple[GroundLiteral, ...]
    rules: tuple[Rule, ...]


def convert_theory_candidate(
    candidate: CandidateTheoryOutput,
    view: PreparedTheoryView,
    *,
    theory_id: str,
) -> ConvertedTheoryBody:
    expected = {item.neutral_id: item for item in view.bindings}
    observed = [item.source_id for item in candidate.statements]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise SourceCoverageError("candidate must cover every neutral source exactly once")

    facts: list[GroundLiteral] = []
    rules: list[Rule] = []
    entity_ids: set[str] = set()
    predicate_arities: dict[str, int] = {}

    def register(predicate: str, arity: int) -> None:
        previous = predicate_arities.setdefault(predicate, arity)
        if previous != arity:
            raise ParserSemanticError(f"predicate {predicate!r} has conflicting arities")

    for statement in candidate.statements:
        binding = expected[statement.source_id]
        if statement.kind != binding.expected_kind:
            raise SourceCoverageError(
                f"{statement.source_id} changed from {binding.expected_kind} to {statement.kind}"
            )
        if isinstance(statement, CandidateFactStatement):
            register(statement.fact.predicate, statement.fact.arity)
            entity_ids.update(term.id for term in statement.fact.arguments)
            facts.append(
                GroundLiteral(
                    predicate=statement.fact.predicate,
                    arguments=statement.fact.arguments,
                    negated=statement.fact.negated,
                    source_id=binding.original_id,
                )
            )
            continue
        if not isinstance(statement, CandidateRuleStatement):
            raise ParserSemanticError("unknown candidate statement type")
        for literal in (*statement.rule.body, statement.rule.head):
            register(literal.predicate, literal.arity)
            entity_ids.update(term.id for term in literal.arguments if isinstance(term, EntityTerm))
        rules.append(
            Rule(
                id=binding.original_id,
                variables=statement.rule.variables,
                body=tuple(
                    _rule_literal(value, binding.original_id) for value in statement.rule.body
                ),
                head=_rule_literal(statement.rule.head, binding.original_id),
                source_id=binding.original_id,
            )
        )

    sources = tuple(SourceStatement(id=item.original_id, text=item.text) for item in view.bindings)
    body = ConvertedTheoryBody(
        theory_id=theory_id,
        sources=sources,
        entities=tuple(Entity(id=value, label=value) for value in sorted(entity_ids)),
        predicates=tuple(
            PredicateDefinition(name=name, arity=arity)
            for name, arity in sorted(predicate_arities.items())
        ),
        facts=tuple(facts),
        rules=tuple(rules),
    )
    _validate_body(body)
    return body


def combine_theory_and_query(
    body: ConvertedTheoryBody,
    candidate: CandidateQueryOutput,
    view: PreparedQueryView,
) -> Theory:
    query = candidate.query
    predicate_arities = {item.name: item.arity for item in body.predicates}
    previous = predicate_arities.setdefault(query.predicate, query.arity)
    if previous != query.arity:
        raise ParserSemanticError("query predicate arity conflicts with the parsed theory")
    entity_ids = {item.id for item in body.entities}
    entity_ids.update(term.id for term in query.arguments)
    source_id = view.original_source_id
    try:
        return Theory(
            schema_version="1.0",
            theory_id=body.theory_id,
            source_statements=(*body.sources, SourceStatement(id=source_id, text=view.text)),
            entities=tuple(Entity(id=value, label=value) for value in sorted(entity_ids)),
            predicates=tuple(
                PredicateDefinition(name=name, arity=arity)
                for name, arity in sorted(predicate_arities.items())
            ),
            facts=body.facts,
            rules=body.rules,
            query=GroundLiteral(
                predicate=query.predicate,
                arguments=query.arguments,
                negated=query.negated,
                source_id=source_id,
            ),
            parser_metadata=ParserMetadata(
                parser_name="local.ollama.semantic_parser",
                parser_version="1.0",
                model_id="qwen3.5:4b-q4_K_M",
            ),
        )
    except ValidationError as error:
        raise ParserSemanticError("parsed theory/query failed semantic validation") from error


def _rule_literal(value: CandidateRuleLiteral, source_id: str) -> RuleLiteral:
    return RuleLiteral(
        predicate=value.predicate,
        arguments=value.arguments,
        negated=value.negated,
        source_id=source_id,
    )


def _validate_body(body: ConvertedTheoryBody) -> None:
    source_id = "validation_query"
    entity_id = "validation_entity"
    predicate = "validation_probe"
    try:
        Theory(
            schema_version="1.0",
            theory_id=body.theory_id,
            source_statements=(
                *body.sources,
                SourceStatement(id=source_id, text="Validation probe."),
            ),
            entities=(*body.entities, Entity(id=entity_id, label=entity_id)),
            predicates=(
                *body.predicates,
                PredicateDefinition(name=predicate, arity=1),
            ),
            facts=body.facts,
            rules=body.rules,
            query=GroundLiteral(
                predicate=predicate,
                arguments=(EntityTerm(kind="entity", id=entity_id),),
                negated=False,
                source_id=source_id,
            ),
        )
    except ValidationError as error:
        raise ParserSemanticError("parsed theory failed AST semantic validation") from error
