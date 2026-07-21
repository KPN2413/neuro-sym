import copy
import json
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from verilogic_ns_api.reasoning.engine import ForwardChainingEngine
from verilogic_ns_api.reasoning.models import (
    CanonicalLiteral,
    EntityTerm,
    RuleLiteral,
    Theory,
    canonical_literal,
)
from verilogic_ns_api.reasoning.verifier import ProofVerifier

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
THEORY_FIXTURES = REPOSITORY_ROOT / "examples" / "theories"


def load_payload(name: str) -> dict[str, object]:
    return json.loads((THEORY_FIXTURES / name).read_text(encoding="utf-8"))


def naive_test_closure(theory: Theory) -> set[CanonicalLiteral]:
    """Deliberately simple exhaustive oracle kept outside production code."""
    closure = {canonical_literal(fact) for fact in theory.facts}
    while True:
        additions: set[CanonicalLiteral] = set()
        for rule in theory.rules:
            states: list[dict[str, str]] = [{}]
            for pattern in rule.body:
                next_states: list[dict[str, str]] = []
                for state in states:
                    for candidate in closure:
                        matched = match_for_test(pattern, candidate, state)
                        if matched is not None:
                            next_states.append(matched)
                states = next_states
            for state in states:
                additions.add(ground_for_test(rule.head, state))
        new = additions - closure
        if not new:
            return closure
        closure.update(new)


def match_for_test(
    pattern: RuleLiteral,
    candidate: CanonicalLiteral,
    state: dict[str, str],
) -> dict[str, str] | None:
    if (
        pattern.predicate != candidate.predicate
        or pattern.negated != candidate.negated
        or len(pattern.arguments) != len(candidate.arguments)
    ):
        return None
    result = dict(state)
    for term, entity in zip(pattern.arguments, candidate.arguments, strict=True):
        if isinstance(term, EntityTerm):
            if term.id != entity:
                return None
        else:
            existing = result.get(term.name)
            if existing is not None and existing != entity:
                return None
            result[term.name] = entity
    return result


def ground_for_test(pattern: RuleLiteral, state: dict[str, str]) -> CanonicalLiteral:
    return CanonicalLiteral(
        predicate=pattern.predicate,
        arguments=tuple(
            term.id if isinstance(term, EntityTerm) else state[term.name]
            for term in pattern.arguments
        ),
        negated=pattern.negated,
    )


def generated_theory(
    entity_count: int,
    selected_facts: set[tuple[str, str, str, bool]],
) -> Theory:
    names = ["a", "b", "c"][:entity_count]
    facts: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    for index, (predicate, first, second, negated) in enumerate(sorted(selected_facts), 1):
        arguments = [first] if predicate == "marked" else [first, second]
        source_id = f"f{index}"
        sources.append({"id": source_id, "text": f"Generated fact {index}."})
        facts.append(
            {
                "predicate": predicate,
                "arguments": [{"kind": "entity", "id": item} for item in arguments],
                "negated": negated,
                "source_id": source_id,
            }
        )
    sources.extend(
        [
            {"id": "r1s", "text": "Marked and linked implies marked."},
            {"id": "r2s", "text": "A link implies reachability."},
            {"id": "r3s", "text": "A constant rule."},
            {"id": "q1", "text": "Is a marked?"},
        ]
    )
    payload = {
        "schema_version": "1.0",
        "theory_id": "generated_property",
        "source_statements": sources,
        "entities": [{"id": item, "label": item.upper()} for item in names],
        "predicates": [
            {"name": "marked", "arity": 1},
            {"name": "linked", "arity": 2},
            {"name": "reachable", "arity": 2},
        ],
        "facts": facts,
        "rules": [
            {
                "id": "r1",
                "variables": [{"name": "X"}, {"name": "Y"}],
                "body": [
                    {
                        "predicate": "marked",
                        "arguments": [{"kind": "variable", "name": "X"}],
                        "negated": False,
                        "source_id": "r1s",
                    },
                    {
                        "predicate": "linked",
                        "arguments": [
                            {"kind": "variable", "name": "X"},
                            {"kind": "variable", "name": "Y"},
                        ],
                        "negated": False,
                        "source_id": "r1s",
                    },
                ],
                "head": {
                    "predicate": "marked",
                    "arguments": [{"kind": "variable", "name": "Y"}],
                    "negated": False,
                    "source_id": "r1s",
                },
                "source_id": "r1s",
            },
            {
                "id": "r2",
                "variables": [{"name": "X"}, {"name": "Y"}],
                "body": [
                    {
                        "predicate": "linked",
                        "arguments": [
                            {"kind": "variable", "name": "X"},
                            {"kind": "variable", "name": "Y"},
                        ],
                        "negated": False,
                        "source_id": "r2s",
                    }
                ],
                "head": {
                    "predicate": "reachable",
                    "arguments": [
                        {"kind": "variable", "name": "X"},
                        {"kind": "variable", "name": "Y"},
                    ],
                    "negated": False,
                    "source_id": "r2s",
                },
                "source_id": "r2s",
            },
            {
                "id": "r3",
                "variables": [{"name": "X"}],
                "body": [
                    {
                        "predicate": "marked",
                        "arguments": [{"kind": "variable", "name": "X"}],
                        "negated": True,
                        "source_id": "r3s",
                    }
                ],
                "head": {
                    "predicate": "marked",
                    "arguments": [{"kind": "entity", "id": names[0]}],
                    "negated": True,
                    "source_id": "r3s",
                },
                "source_id": "r3s",
            },
        ],
        "query": {
            "predicate": "marked",
            "arguments": [{"kind": "entity", "id": names[0]}],
            "negated": False,
            "source_id": "q1",
        },
    }
    return Theory.model_validate(payload)


@st.composite
def generated_theories(draw) -> Theory:
    entity_count = draw(st.integers(min_value=1, max_value=3))
    names = ["a", "b", "c"][:entity_count]
    candidates = [
        ("marked", first, "", negated) for first in names for negated in (False, True)
    ] + [
        ("linked", first, second, negated)
        for first in names
        for second in names
        for negated in (False, True)
    ]
    selected = draw(
        st.sets(st.sampled_from(candidates), min_size=1, max_size=min(8, len(candidates)))
    )
    return generated_theory(entity_count, selected)


@given(generated_theories())
def test_generated_mixed_theories_match_test_only_naive_oracle(theory: Theory) -> None:
    production = {entry.literal for entry in ForwardChainingEngine().saturate(theory).closure}

    assert production == naive_test_closure(theory)


@given(st.permutations([0, 1]), st.permutations([0, 1]))
def test_fact_and_rule_permutations_preserve_reasoning(
    fact_order: list[int], rule_order: list[int]
) -> None:
    payload = load_payload("multiple-derivations.json")
    original = ForwardChainingEngine().reason(Theory.model_validate(payload))
    payload["facts"] = [payload["facts"][index] for index in fact_order]
    payload["rules"] = [payload["rules"][index] for index in rule_order]

    permuted = ForwardChainingEngine().reason(Theory.model_validate(payload))

    assert permuted.result.status is original.result.status
    assert permuted.result.proof.proof_hash == original.result.proof.proof_hash


@given(st.integers(min_value=1, max_value=8))
def test_duplicate_fact_multiplicity_does_not_change_closure(multiplicity: int) -> None:
    payload = load_payload("unary-multistep.json")
    original_fact = copy.deepcopy(payload["facts"][0])
    payload["facts"] = [copy.deepcopy(original_fact) for _ in range(multiplicity)]

    closure = ForwardChainingEngine().saturate(Theory.model_validate(payload))

    assert len(closure.closure) == 3


@given(st.sampled_from(["Bob", "Carol", "Delta", "Echo"]))
def test_irrelevant_facts_do_not_change_query_status(label: str) -> None:
    payload = load_payload("unary-multistep.json")
    baseline = ForwardChainingEngine().reason(Theory.model_validate(payload))
    identifier = label.lower()
    payload["entities"].append({"id": identifier, "label": label})
    payload["predicates"].append({"name": "irrelevant", "arity": 1})
    payload["source_statements"].append({"id": "s9", "text": f"{label} is irrelevant."})
    payload["facts"].append(
        {
            "predicate": "irrelevant",
            "arguments": [{"kind": "entity", "id": identifier}],
            "negated": False,
            "source_id": "s9",
        }
    )

    extended = ForwardChainingEngine().reason(Theory.model_validate(payload))

    assert extended.result.status is baseline.result.status


@given(st.sampled_from(["A", "Person", "Subject", "EntityVar"]))
def test_alpha_renaming_variables_preserves_closure(variable_name: str) -> None:
    payload = load_payload("unary-multistep.json")
    baseline = ForwardChainingEngine().saturate(Theory.model_validate(payload))
    for rule in payload["rules"]:
        old_name = rule["variables"][0]["name"]
        rule["variables"][0]["name"] = variable_name
        for literal in [*rule["body"], rule["head"]]:
            for term in literal["arguments"]:
                if term["kind"] == "variable" and term["name"] == old_name:
                    term["name"] = variable_name

    renamed = ForwardChainingEngine().saturate(Theory.model_validate(payload))

    assert [item.literal for item in renamed.closure] == [item.literal for item in baseline.closure]


@given(st.booleans())
def test_adding_positive_facts_is_monotonic(negated: bool) -> None:
    payload = load_payload("unary-multistep.json")
    original = Theory.model_validate(payload)
    original_closure = {item.literal for item in ForwardChainingEngine().saturate(original).closure}
    payload["entities"].append({"id": "bob", "label": "Bob"})
    payload["source_statements"].append({"id": "s9", "text": "A new explicit fact."})
    payload["facts"].append(
        {
            "predicate": "kind",
            "arguments": [{"kind": "entity", "id": "bob"}],
            "negated": negated,
            "source_id": "s9",
        }
    )
    extended_closure = {
        item.literal
        for item in ForwardChainingEngine().saturate(Theory.model_validate(payload)).closure
    }

    assert original_closure <= extended_closure


@given(
    st.sampled_from(
        [
            "entailed.json",
            "contradicted.json",
            "unknown.json",
            "inconsistent.json",
            "unary-multistep.json",
            "binary-join.json",
            "explicit-negative-premise.json",
            "recursive-cycle.json",
            "multiple-derivations.json",
        ]
    )
)
def test_production_and_naive_reference_closures_agree(fixture_name: str) -> None:
    theory = Theory.model_validate(load_payload(fixture_name))
    production = {entry.literal for entry in ForwardChainingEngine().saturate(theory).closure}

    reference = naive_test_closure(theory)

    assert production == reference


@given(st.sampled_from(["unary-multistep.json", "binary-join.json", "recursive-cycle.json"]))
def test_saturation_and_proof_output_are_repeatable(fixture_name: str) -> None:
    theory = Theory.model_validate(load_payload(fixture_name))
    engine = ForwardChainingEngine()

    first_closure = engine.saturate(theory)
    second_closure = engine.saturate(theory)
    first_result = engine.reason(theory)
    second_result = engine.reason(theory)

    assert first_closure.closure == second_closure.closure
    assert first_result.result.proof == second_result.result.proof
    assert ProofVerifier().verify_result(theory, first_result.result).valid is True


@given(st.sampled_from(["unary-multistep.json", "multiple-derivations.json"]))
def test_semantically_duplicate_rules_do_not_change_closure(fixture_name: str) -> None:
    payload = load_payload(fixture_name)
    baseline = {
        item.literal
        for item in ForwardChainingEngine().saturate(Theory.model_validate(payload)).closure
    }
    duplicate = copy.deepcopy(payload["rules"][0])
    duplicate["id"] = "duplicate_rule"
    duplicate["source_id"] = "duplicate_source"
    for literal in [*duplicate["body"], duplicate["head"]]:
        literal["source_id"] = "duplicate_source"
    payload["source_statements"].append(
        {"id": "duplicate_source", "text": "A duplicate logical rule."}
    )
    payload["rules"].append(duplicate)

    extended = {
        item.literal
        for item in ForwardChainingEngine().saturate(Theory.model_validate(payload)).closure
    }

    assert extended == baseline


@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
    st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
        min_size=1,
        max_size=2,
        unique=True,
    ),
    st.booleans(),
)
def test_query_complement_is_an_involution(
    predicate: str, arguments: list[str], negated: bool
) -> None:
    literal = CanonicalLiteral(
        predicate=predicate,
        arguments=tuple(arguments),
        negated=negated,
    )

    assert literal.opposite().opposite() == literal


@given(st.integers(min_value=1, max_value=3))
def test_unrelated_inconsistency_does_not_explode_and_positive_recursion_terminates(
    entity_count: int,
) -> None:
    theory = generated_theory(
        entity_count,
        {
            ("marked", "a", "", False),
            ("marked", "a", "", True),
            ("linked", "a", "a", False),
        },
    )

    outcome = ForwardChainingEngine().saturate(theory)

    assert len(outcome.closure) <= 3 * entity_count + 2 * entity_count * entity_count
    assert all(
        item.literal.predicate in {"marked", "linked", "reachable"} for item in outcome.closure
    )
