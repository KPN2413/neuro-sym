import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from verilogic_ns_api.reasoning.configuration import ReasoningLimits, ResourceLimitError
from verilogic_ns_api.reasoning.engine import ForwardChainingEngine
from verilogic_ns_api.reasoning.models import ReasoningStatus, Theory
from verilogic_ns_api.reasoning.verifier import ProofVerifier

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
THEORY_FIXTURES = REPOSITORY_ROOT / "examples" / "theories"


def load_payload(name: str) -> dict[str, object]:
    return json.loads((THEORY_FIXTURES / name).read_text(encoding="utf-8"))


def load_theory(name: str) -> Theory:
    return Theory.model_validate(load_payload(name))


def chain_theory(depth: int) -> Theory:
    sources = [{"id": "s0", "text": "Alice has property zero."}]
    sources.extend(
        {"id": f"s{index}", "text": f"Property {index - 1} implies {index}."}
        for index in range(1, depth + 1)
    )
    sources.append({"id": "q1", "text": "Is the final property known?"})
    return Theory.model_validate(
        {
            "schema_version": "1.0",
            "theory_id": f"chain_{depth}",
            "source_statements": sources,
            "entities": [{"id": "alice", "label": "Alice"}],
            "predicates": [{"name": f"p{index}", "arity": 1} for index in range(depth + 1)],
            "facts": [
                {
                    "predicate": "p0",
                    "arguments": [{"kind": "entity", "id": "alice"}],
                    "negated": False,
                    "source_id": "s0",
                }
            ],
            "rules": [
                {
                    "id": f"r{index}",
                    "variables": [{"name": "X"}],
                    "body": [
                        {
                            "predicate": f"p{index - 1}",
                            "arguments": [{"kind": "variable", "name": "X"}],
                            "negated": False,
                            "source_id": f"s{index}",
                        }
                    ],
                    "head": {
                        "predicate": f"p{index}",
                        "arguments": [{"kind": "variable", "name": "X"}],
                        "negated": False,
                        "source_id": f"s{index}",
                    },
                    "source_id": f"s{index}",
                }
                for index in range(1, depth + 1)
            ],
            "query": {
                "predicate": f"p{depth}",
                "arguments": [{"kind": "entity", "id": "alice"}],
                "negated": False,
                "source_id": "q1",
            },
        }
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        ("entailed.json", ReasoningStatus.ENTAILED),
        ("contradicted.json", ReasoningStatus.CONTRADICTED),
        ("unknown.json", ReasoningStatus.UNKNOWN),
        ("inconsistent.json", ReasoningStatus.INCONSISTENT),
    ],
)
def test_four_way_query_classification(fixture_name: str, expected: ReasoningStatus) -> None:
    theory = load_theory(fixture_name)

    outcome = ForwardChainingEngine().reason(theory)

    assert outcome.result.status is expected
    verified = ProofVerifier().verify_result(theory, outcome.result)
    assert verified.status is expected


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        ("entailed.json", ReasoningStatus.CONTRADICTED),
        ("contradicted.json", ReasoningStatus.ENTAILED),
    ],
)
def test_negative_queries_are_classified_symmetrically(
    fixture_name: str, expected: ReasoningStatus
) -> None:
    payload = load_payload(fixture_name)
    payload["query"]["negated"] = not payload["query"]["negated"]
    theory = Theory.model_validate(payload)

    assert ForwardChainingEngine().reason(theory).result.status is expected


def test_unrelated_conflict_does_not_make_an_unknown_query_inconsistent() -> None:
    payload = load_payload("inconsistent.json")
    payload["predicates"].append({"name": "green", "arity": 1})
    payload["query"] = {
        "predicate": "green",
        "arguments": [{"kind": "entity", "id": "alice"}],
        "negated": False,
        "source_id": "q1",
    }
    theory = Theory.model_validate(payload)

    outcome = ForwardChainingEngine().reason(theory)

    assert outcome.result.status is ReasoningStatus.UNKNOWN
    assert outcome.result.closure_contains_conflicts is True
    assert outcome.result.conflict_count == 1


def test_multistep_chain_has_a_machine_verified_depth_two_proof() -> None:
    theory = load_theory("unary-multistep.json")

    outcome = ForwardChainingEngine().reason(theory)
    verification = ProofVerifier().verify_result(theory, outcome.result)

    assert outcome.result.status is ReasoningStatus.ENTAILED
    assert outcome.telemetry.maximum_proof_depth == 2
    assert verification.node_count == 5


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_exact_reasoning_depth_chains(depth: int) -> None:
    theory = chain_theory(depth)

    outcome = ForwardChainingEngine().reason(theory)

    assert outcome.result.status is ReasoningStatus.ENTAILED
    assert outcome.telemetry.maximum_proof_depth == depth
    assert ProofVerifier().verify_result(theory, outcome.result).valid is True


def test_both_query_polarities_can_be_derived_by_rules() -> None:
    payload = load_payload("multiple-derivations.json")
    payload["rules"][1]["head"]["negated"] = True
    theory = Theory.model_validate(payload)

    outcome = ForwardChainingEngine().reason(theory)

    assert outcome.result.status is ReasoningStatus.INCONSISTENT
    assert outcome.result.proof.support_root_id is not None
    assert outcome.result.proof.opposition_root_id is not None
    assert ProofVerifier().verify_result(theory, outcome.result).valid is True


def test_conjunction_requires_every_premise() -> None:
    payload = load_payload("binary-join.json")
    payload["facts"] = payload["facts"][:1]
    theory = Theory.model_validate(payload)

    assert ForwardChainingEngine().reason(theory).result.status is ReasoningStatus.UNKNOWN


def test_explicit_negative_premise_can_fire_a_rule() -> None:
    theory = load_theory("explicit-negative-premise.json")

    assert ForwardChainingEngine().reason(theory).result.status is ReasoningStatus.ENTAILED


def test_negative_rule_head_proves_the_query_complement() -> None:
    payload = load_payload("unary-multistep.json")
    payload["rules"][1]["head"]["negated"] = True
    theory = Theory.model_validate(payload)

    assert ForwardChainingEngine().reason(theory).result.status is ReasoningStatus.CONTRADICTED


def test_constants_in_rule_bodies_and_binary_joins_are_supported() -> None:
    payload = load_payload("binary-join.json")
    payload["rules"][0]["body"][0]["arguments"][0] = {
        "kind": "entity",
        "id": "alice",
    }
    payload["rules"][0]["head"]["arguments"][0] = {
        "kind": "entity",
        "id": "alice",
    }
    payload["rules"][0]["variables"] = [
        variable for variable in payload["rules"][0]["variables"] if variable["name"] != "X"
    ]
    theory = Theory.model_validate(payload)

    assert ForwardChainingEngine().reason(theory).result.status is ReasoningStatus.ENTAILED


def test_constant_in_a_safe_rule_head_is_supported() -> None:
    payload = load_payload("unary-multistep.json")
    payload["rules"][0]["head"]["arguments"] = [{"kind": "entity", "id": "alice"}]
    payload["rules"][0]["variables"] = [{"name": "X"}]
    payload["query"] = {
        "predicate": "calm",
        "arguments": [{"kind": "entity", "id": "alice"}],
        "negated": False,
        "source_id": "q1",
    }
    theory = Theory.model_validate(payload)

    assert ForwardChainingEngine().reason(theory).result.status is ReasoningStatus.ENTAILED


def test_binary_source_fact_is_directly_entailed() -> None:
    payload = load_payload("binary-join.json")
    payload["query"] = copy.deepcopy(payload["facts"][0])
    payload["query"]["source_id"] = "q1"
    theory = Theory.model_validate(payload)

    assert ForwardChainingEngine().reason(theory).result.status is ReasoningStatus.ENTAILED


def test_repeated_variable_requires_equal_arguments() -> None:
    payload = load_payload("binary-join.json")
    payload["rules"][0]["variables"] = [{"name": "X"}]
    payload["rules"][0]["body"] = [
        {
            "predicate": "likes",
            "arguments": [
                {"kind": "variable", "name": "X"},
                {"kind": "variable", "name": "X"},
            ],
            "negated": False,
            "source_id": "s3",
        }
    ]
    payload["rules"][0]["head"] = {
        "predicate": "connected",
        "arguments": [
            {"kind": "variable", "name": "X"},
            {"kind": "variable", "name": "X"},
        ],
        "negated": False,
        "source_id": "s3",
    }
    payload["query"]["arguments"] = [
        {"kind": "entity", "id": "alice"},
        {"kind": "entity", "id": "alice"},
    ]
    theory = Theory.model_validate(payload)
    assert ForwardChainingEngine().reason(theory).result.status is ReasoningStatus.UNKNOWN

    payload["facts"][0]["arguments"][1]["id"] = "alice"
    matching = Theory.model_validate(payload)
    assert ForwardChainingEngine().reason(matching).result.status is ReasoningStatus.ENTAILED


def test_binary_join_does_not_create_cartesian_bindings() -> None:
    payload = load_payload("binary-join.json")
    payload["facts"][1]["arguments"] = [
        {"kind": "entity", "id": "carol"},
        {"kind": "entity", "id": "alice"},
    ]
    theory = Theory.model_validate(payload)

    assert ForwardChainingEngine().reason(theory).result.status is ReasoningStatus.UNKNOWN


def test_multiple_substitutions_produce_distinct_ground_conclusions() -> None:
    payload = load_payload("unary-multistep.json")
    payload["entities"].append({"id": "bob", "label": "Bob"})
    payload["source_statements"].append({"id": "s4", "text": "Bob is kind."})
    payload["facts"].append(
        {
            "predicate": "kind",
            "arguments": [{"kind": "entity", "id": "bob"}],
            "negated": False,
            "source_id": "s4",
        }
    )
    theory = Theory.model_validate(payload)

    closure = ForwardChainingEngine().saturate(theory).closure
    trusted_entities = {
        entry.literal.arguments[0]
        for entry in closure
        if entry.literal.predicate == "trusted" and not entry.literal.negated
    }

    assert trusted_entities == {"alice", "bob"}


def test_duplicate_facts_are_idempotent_and_duplicate_rule_ids_are_rejected() -> None:
    payload = load_payload("unary-multistep.json")
    payload["facts"].append(copy.deepcopy(payload["facts"][0]))
    theory = Theory.model_validate(payload)

    closure = ForwardChainingEngine().saturate(theory)
    assert len(closure.closure) == 3
    assert closure.telemetry.duplicate_conclusions >= 1

    payload = load_payload("unary-multistep.json")
    payload["rules"].append(copy.deepcopy(payload["rules"][0]))
    with pytest.raises(ValidationError, match="duplicate rule identifier"):
        Theory.model_validate(payload)


def test_positive_recursion_reaches_a_finite_fixpoint() -> None:
    theory = load_theory("recursive-cycle.json")

    outcome = ForwardChainingEngine().reason(theory)

    assert outcome.result.status is ReasoningStatus.ENTAILED
    assert outcome.telemetry.total_closure_size == 2
    assert outcome.telemetry.rounds == 1


@pytest.mark.parametrize("seeded", [False, True])
def test_self_cycle_terminates_with_or_without_a_seed(seeded: bool) -> None:
    payload = load_payload("recursive-cycle.json")
    payload["rules"] = [payload["rules"][0]]
    payload["rules"][0]["head"]["predicate"] = "ready"
    payload["query"]["predicate"] = "ready"
    if not seeded:
        payload["facts"] = []
    theory = Theory.model_validate(payload)

    outcome = ForwardChainingEngine().reason(theory)

    expected = ReasoningStatus.ENTAILED if seeded else ReasoningStatus.UNKNOWN
    assert outcome.result.status is expected
    assert outcome.telemetry.rounds == 0


def test_irrelevant_rule_does_not_change_a_query_result() -> None:
    payload = load_payload("unary-multistep.json")
    payload["source_statements"].append({"id": "s9", "text": "Bright implies bright."})
    payload["predicates"].append({"name": "bright", "arity": 1})
    payload["rules"].append(
        {
            "id": "r9",
            "variables": [{"name": "X"}],
            "body": [
                {
                    "predicate": "bright",
                    "arguments": [{"kind": "variable", "name": "X"}],
                    "negated": False,
                    "source_id": "s9",
                }
            ],
            "head": {
                "predicate": "bright",
                "arguments": [{"kind": "variable", "name": "X"}],
                "negated": False,
                "source_id": "s9",
            },
            "source_id": "s9",
        }
    )

    assert (
        ForwardChainingEngine().reason(Theory.model_validate(payload)).result.status
        is ReasoningStatus.ENTAILED
    )


def test_engine_does_not_apply_contraposition() -> None:
    payload = load_payload("unary-multistep.json")
    payload["facts"][0] = {
        "predicate": "calm",
        "arguments": [{"kind": "entity", "id": "alice"}],
        "negated": True,
        "source_id": "s1",
    }
    payload["rules"] = payload["rules"][:1]
    payload["query"] = {
        "predicate": "kind",
        "arguments": [{"kind": "entity", "id": "alice"}],
        "negated": True,
        "source_id": "q1",
    }
    theory = Theory.model_validate(payload)

    assert ForwardChainingEngine().reason(theory).result.status is ReasoningStatus.UNKNOWN


def test_input_order_does_not_change_the_closure_or_proof_hash() -> None:
    payload = load_payload("multiple-derivations.json")
    original = Theory.model_validate(payload)
    reordered_payload = copy.deepcopy(payload)
    for field in ("source_statements", "predicates", "facts", "rules"):
        reordered_payload[field].reverse()
    reordered = Theory.model_validate(reordered_payload)

    first = ForwardChainingEngine().reason(original)
    second = ForwardChainingEngine().reason(reordered)

    assert first.result.status is second.result.status
    assert first.result.proof.proof_hash == second.result.proof.proof_hash
    assert first.result.proof.support_root_id == second.result.proof.support_root_id


@pytest.mark.parametrize(
    ("limits", "limit_name"),
    [
        (ReasoningLimits(max_derived_literals=1), "max_derived_literals"),
        (ReasoningLimits(max_rule_firings=1), "max_rule_firings"),
        (ReasoningLimits(max_rounds=1), "max_rounds"),
        (ReasoningLimits(max_proof_nodes=1), "max_proof_nodes"),
    ],
)
def test_resource_limits_fail_closed(limits: ReasoningLimits, limit_name: str) -> None:
    theory = load_theory("unary-multistep.json")

    with pytest.raises(ResourceLimitError) as error:
        ForwardChainingEngine(limits).reason(theory)

    assert error.value.limit_name == limit_name


def test_optional_timeout_is_a_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([0.0, 2.0])
    monkeypatch.setattr(
        "verilogic_ns_api.reasoning.engine.time.perf_counter", lambda: next(readings)
    )

    with pytest.raises(ResourceLimitError) as error:
        ForwardChainingEngine(ReasoningLimits(timeout_seconds=1.0)).reason(
            load_theory("unary-multistep.json")
        )

    assert error.value.limit_name == "timeout_seconds"


def test_initial_fact_limit_fails_before_returning_a_partial_result() -> None:
    with pytest.raises(ResourceLimitError) as error:
        ForwardChainingEngine(ReasoningLimits(max_derived_literals=1)).reason(
            load_theory("inconsistent.json")
        )

    assert error.value.limit_name == "max_derived_literals"
    assert error.value.observed == 2
