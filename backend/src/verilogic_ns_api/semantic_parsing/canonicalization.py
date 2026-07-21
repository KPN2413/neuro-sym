from __future__ import annotations

import itertools
import json

from verilogic_ns_api.reasoning.engine import ForwardChainingEngine
from verilogic_ns_api.reasoning.models import (
    EntityTerm,
    Rule,
    RuleLiteral,
    Theory,
    VariableTerm,
)


def canonical_literal_key(literal: object) -> tuple[object, ...]:
    arguments = literal.arguments
    return (
        literal.predicate,
        bool(literal.negated),
        tuple(
            ("entity", term.id) if isinstance(term, EntityTerm) else ("variable", term.name)
            for term in arguments
        ),
    )


def canonical_rule_key(rule: Rule) -> str:
    names = sorted(variable.name for variable in rule.variables)
    canonical_names = [f"V{index}" for index in range(len(names))]
    mappings = (
        (
            dict(zip(names, permutation, strict=True))
            for permutation in itertools.permutations(canonical_names)
        )
        if len(names) <= 7
        else (dict(zip(names, canonical_names, strict=True)),)
    )
    candidates: list[str] = []
    for mapping in mappings:
        body = sorted(_mapped_literal(item, mapping) for item in rule.body)
        head = _mapped_literal(rule.head, mapping)
        candidates.append(
            json.dumps({"body": body, "head": head}, sort_keys=True, separators=(",", ":"))
        )
    return min(candidates)


def canonical_statement_sets(theory: Theory) -> tuple[set[str], set[str]]:
    facts = {
        json.dumps(canonical_literal_key(item), sort_keys=True, separators=(",", ":"))
        for item in theory.facts
    }
    rules = {canonical_rule_key(item) for item in theory.rules}
    return facts, rules


def canonical_query(theory: Theory) -> tuple[object, ...]:
    return canonical_literal_key(theory.query)


def closure_keys(theory: Theory) -> set[str]:
    result = ForwardChainingEngine().saturate(theory)
    return {
        json.dumps(item.literal.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for item in result.closure
    }


def _mapped_literal(literal: RuleLiteral, mapping: dict[str, str]) -> tuple[object, ...]:
    arguments: list[tuple[str, str]] = []
    for term in literal.arguments:
        if isinstance(term, VariableTerm):
            arguments.append(("variable", mapping[term.name]))
        else:
            arguments.append(("entity", term.id))
    return (literal.predicate, literal.negated, tuple(arguments))
