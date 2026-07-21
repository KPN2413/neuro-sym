from __future__ import annotations

from collections.abc import Iterator, Mapping

from verilogic_ns_api.reasoning.models import (
    CanonicalLiteral,
    EntityTerm,
    RuleLiteral,
    VariableTerm,
)

IndexKey = tuple[str, int, bool]
Substitution = dict[str, str]


def literal_index_key(literal: CanonicalLiteral | RuleLiteral) -> IndexKey:
    return (literal.predicate, len(literal.arguments), literal.negated)


def rule_literal_sort_key(literal: RuleLiteral) -> tuple[object, ...]:
    arguments = tuple(
        ("entity", term.id) if isinstance(term, EntityTerm) else ("variable", term.name)
        for term in literal.arguments
    )
    return (literal.predicate, literal.negated, arguments, literal.source_id)


def match_literal(
    pattern: RuleLiteral,
    candidate: CanonicalLiteral,
    substitution: Mapping[str, str] | None = None,
) -> Substitution | None:
    if literal_index_key(pattern) != literal_index_key(candidate):
        return None
    result = dict(substitution or {})
    for term, entity in zip(pattern.arguments, candidate.arguments, strict=True):
        if isinstance(term, EntityTerm):
            if term.id != entity:
                return None
            continue
        previous = result.get(term.name)
        if previous is not None and previous != entity:
            return None
        result[term.name] = entity
    return result


def ground_literal(pattern: RuleLiteral, substitution: Mapping[str, str]) -> CanonicalLiteral:
    arguments: list[str] = []
    for term in pattern.arguments:
        if isinstance(term, EntityTerm):
            arguments.append(term.id)
        elif isinstance(term, VariableTerm):
            try:
                arguments.append(substitution[term.name])
            except KeyError as error:
                raise ValueError(f"Missing binding for variable {term.name!r}") from error
    return CanonicalLiteral(
        predicate=pattern.predicate,
        arguments=tuple(arguments),
        negated=pattern.negated,
    )


def iter_body_matches(
    body: tuple[RuleLiteral, ...],
    index: Mapping[IndexKey, tuple[CanonicalLiteral, ...]],
    delta: set[CanonicalLiteral] | None = None,
) -> Iterator[tuple[Substitution, tuple[CanonicalLiteral, ...]]]:
    ordered_body = tuple(sorted(body, key=rule_literal_sort_key))
    states: list[tuple[Substitution, tuple[CanonicalLiteral, ...]]] = [({}, ())]
    for pattern in ordered_body:
        candidates = index.get(literal_index_key(pattern), ())
        next_states: list[tuple[Substitution, tuple[CanonicalLiteral, ...]]] = []
        for substitution, premises in states:
            for candidate in candidates:
                matched = match_literal(pattern, candidate, substitution)
                if matched is not None:
                    next_states.append((matched, (*premises, candidate)))
        states = _deduplicate_states(next_states)
        if not states:
            return

    for substitution, premises in states:
        if delta is None or any(premise in delta for premise in premises):
            yield substitution, premises


def _deduplicate_states(
    states: list[tuple[Substitution, tuple[CanonicalLiteral, ...]]],
) -> list[tuple[Substitution, tuple[CanonicalLiteral, ...]]]:
    unique: dict[
        tuple[tuple[tuple[str, str], ...], tuple[tuple[str, int, bool, tuple[str, ...]], ...]],
        tuple[Substitution, tuple[CanonicalLiteral, ...]],
    ] = {}
    for substitution, premises in states:
        key = (
            tuple(sorted(substitution.items())),
            tuple(premise.sort_key() for premise in premises),
        )
        unique[key] = (substitution, premises)
    return [unique[key] for key in sorted(unique)]
