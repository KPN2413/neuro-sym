import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from verilogic_ns_api.reasoning.configuration import ProofVerificationError
from verilogic_ns_api.reasoning.engine import ForwardChainingEngine
from verilogic_ns_api.reasoning.models import (
    DerivedLiteralNode,
    ProofDAG,
    ReasoningStatus,
    RuleApplicationNode,
    Theory,
    canonical_json,
    canonical_literal,
    theory_hash,
)
from verilogic_ns_api.reasoning.proofs import (
    make_source_derivation,
    recompute_node_id,
    recompute_proof_hash,
)
from verilogic_ns_api.reasoning.schema_export import (
    proof_schema,
    reasoning_result_schema,
)
from verilogic_ns_api.reasoning.verifier import ProofVerifier, _assert_acyclic

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
THEORY_FIXTURES = REPOSITORY_ROOT / "examples" / "theories"
SCHEMA_DIRECTORY = REPOSITORY_ROOT / "schemas"


def load_payload(name: str) -> dict[str, object]:
    return json.loads((THEORY_FIXTURES / name).read_text(encoding="utf-8"))


def load_theory(name: str) -> Theory:
    return Theory.model_validate(load_payload(name))


def replace_application(
    proof: ProofDAG,
    old_application: RuleApplicationNode,
    application: RuleApplicationNode,
) -> ProofDAG:
    application = application.model_copy(update={"node_id": recompute_node_id(application)})
    old_derived = next(
        node
        for node in proof.nodes
        if isinstance(node, DerivedLiteralNode)
        and node.rule_application_node_id == old_application.node_id
    )
    derived = old_derived.model_copy(update={"rule_application_node_id": application.node_id})
    derived = derived.model_copy(update={"node_id": recompute_node_id(derived)})
    nodes_by_id = {
        node.node_id: node
        for node in proof.nodes
        if node.node_id not in {old_application.node_id, old_derived.node_id}
    }
    nodes_by_id[application.node_id] = application
    nodes_by_id[derived.node_id] = derived
    updated = proof.model_copy(
        update={
            "support_root_id": (
                derived.node_id
                if proof.support_root_id == old_derived.node_id
                else proof.support_root_id
            ),
            "opposition_root_id": (
                derived.node_id
                if proof.opposition_root_id == old_derived.node_id
                else proof.opposition_root_id
            ),
            "nodes": tuple(nodes_by_id[key] for key in sorted(nodes_by_id)),
        }
    )
    return updated.model_copy(update={"proof_hash": recompute_proof_hash(updated)})


@pytest.mark.parametrize(
    "fixture_name",
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
    ],
)
def test_every_valid_reasoning_fixture_passes_json_schema_and_typed_validation(
    fixture_name: str,
) -> None:
    schema = json.loads((SCHEMA_DIRECTORY / "theory.v1.schema.json").read_text(encoding="utf-8"))
    payload = load_payload(fixture_name)

    assert list(Draft202012Validator(schema).iter_errors(payload)) == []
    Theory.model_validate(payload)


@pytest.mark.parametrize(
    ("fixture_name", "message"),
    [
        ("unsafe-head-variable.json", "unbound head variables"),
        ("predicate-arity-conflict.json", "expects arity"),
        ("malformed-source-reference.json", "unknown source"),
    ],
)
def test_semantically_invalid_fixtures_fail_typed_validation(
    fixture_name: str, message: str
) -> None:
    payload = json.loads((THEORY_FIXTURES / "invalid" / fixture_name).read_text(encoding="utf-8"))

    with pytest.raises(ValidationError, match=message):
        Theory.model_validate(payload)


@pytest.mark.parametrize(
    "fixture_name",
    ["non-ground-fact.json", "unsupported-arity.json"],
)
def test_new_structurally_invalid_fixtures_fail_json_schema(fixture_name: str) -> None:
    schema = json.loads((SCHEMA_DIRECTORY / "theory.v1.schema.json").read_text(encoding="utf-8"))
    payload = json.loads((THEORY_FIXTURES / "invalid" / fixture_name).read_text(encoding="utf-8"))

    assert list(Draft202012Validator(schema).iter_errors(payload))


def test_semantic_validator_rejects_duplicate_ids_undeclared_entities_and_types() -> None:
    payload = load_payload("unary-multistep.json")
    payload["source_statements"].append(copy.deepcopy(payload["source_statements"][0]))
    with pytest.raises(ValidationError, match="duplicate source statement"):
        Theory.model_validate(payload)

    payload = load_payload("unary-multistep.json")
    payload["facts"][0]["arguments"][0]["id"] = "missing"
    with pytest.raises(ValidationError, match="undeclared entity"):
        Theory.model_validate(payload)

    payload = load_payload("unary-multistep.json")
    payload["entities"][0]["type"] = "Person"
    payload["predicates"][0]["argument_types"] = ["Animal"]
    with pytest.raises(ValidationError, match="type mismatch"):
        Theory.model_validate(payload)


def test_generated_proof_and_result_schemas_match_typed_models() -> None:
    saved_proof = json.loads(
        (SCHEMA_DIRECTORY / "proof.v1.schema.json").read_text(encoding="utf-8")
    )
    saved_result = json.loads(
        (SCHEMA_DIRECTORY / "reasoning-result.v1.schema.json").read_text(encoding="utf-8")
    )

    Draft202012Validator.check_schema(saved_proof)
    Draft202012Validator.check_schema(saved_result)
    assert saved_proof == proof_schema()
    assert saved_result == reasoning_result_schema()


def test_proof_contains_exact_sources_rule_and_sorted_substitution() -> None:
    theory = load_theory("binary-join.json")

    outcome = ForwardChainingEngine().reason(theory)
    applications = [
        node for node in outcome.result.proof.nodes if isinstance(node, RuleApplicationNode)
    ]

    assert len(applications) == 1
    application = applications[0]
    assert application.rule_id == "r1"
    assert application.source_id == "s3"
    assert application.source_text == "Likes and trusts compose into connected."
    assert [(item.variable, item.entity) for item in application.substitution] == [
        ("X", "alice"),
        ("Y", "bob"),
        ("Z", "carol"),
    ]
    assert ProofVerifier().verify_result(theory, outcome.result).valid is True


def test_canonical_derivation_selects_the_lowest_rule_id() -> None:
    theory = load_theory("multiple-derivations.json")

    proof = ForwardChainingEngine().reason(theory).result.proof
    applications = [node for node in proof.nodes if isinstance(node, RuleApplicationNode)]

    assert [node.rule_id for node in applications] == ["r1"]


def test_direct_fact_is_preferred_over_a_longer_derivation() -> None:
    payload = load_payload("unary-multistep.json")
    payload["source_statements"].append({"id": "s9", "text": "Alice is trusted directly."})
    payload["facts"].append(
        {
            "predicate": "trusted",
            "arguments": [{"kind": "entity", "id": "alice"}],
            "negated": False,
            "source_id": "s9",
        }
    )
    theory = Theory.model_validate(payload)

    proof = ForwardChainingEngine().reason(theory).result.proof

    assert len(proof.nodes) == 1
    assert proof.nodes[0].node_type == "source_fact"
    assert proof.nodes[0].source_id == "s9"


def test_duplicate_source_facts_choose_the_lexicographically_first_source() -> None:
    payload = load_payload("entailed.json")
    duplicate = copy.deepcopy(payload["facts"][0])
    duplicate["source_id"] = "z9"
    payload["source_statements"].append({"id": "z9", "text": "Duplicate fact source."})
    payload["facts"].insert(0, duplicate)
    theory = Theory.model_validate(payload)

    proof = ForwardChainingEngine().reason(theory).result.proof

    source_nodes = [node for node in proof.nodes if node.node_type == "source_fact"]
    assert source_nodes[0].source_id == "s1"


def test_canonical_proof_serialization_is_byte_stable() -> None:
    theory = load_theory("multiple-derivations.json")

    first = ForwardChainingEngine().reason(theory).result.proof
    second = ForwardChainingEngine().reason(theory).result.proof

    assert canonical_json(first) == canonical_json(second)
    assert first.proof_hash == second.proof_hash


def test_tampered_proof_hash_is_rejected() -> None:
    theory = load_theory("unary-multistep.json")
    proof = ForwardChainingEngine().reason(theory).result.proof
    tampered = proof.model_copy(update={"proof_hash": "0" * 64})

    with pytest.raises(ProofVerificationError, match="proof_hash"):
        ProofVerifier().verify_proof(theory, tampered)


def test_tampered_node_is_rejected_even_when_outer_proof_hash_is_recomputed() -> None:
    theory = load_theory("unary-multistep.json")
    proof = ForwardChainingEngine().reason(theory).result.proof
    nodes = list(proof.nodes)
    source_index = next(
        index for index, node in enumerate(nodes) if node.node_type == "source_fact"
    )
    nodes[source_index] = nodes[source_index].model_copy(update={"source_text": "Tampered"})
    tampered = proof.model_copy(update={"nodes": tuple(nodes)})
    tampered = tampered.model_copy(update={"proof_hash": recompute_proof_hash(tampered)})

    with pytest.raises(ProofVerificationError, match="node_hash"):
        ProofVerifier().verify_proof(theory, tampered)


def test_semantically_tampered_rule_source_is_rejected_after_rehashing_graph() -> None:
    theory = load_theory("unary-multistep.json")
    proof = ForwardChainingEngine().reason(theory).result.proof
    old_application = next(
        node
        for node in proof.nodes
        if isinstance(node, RuleApplicationNode) and node.rule_id == "r2"
    )
    application = old_application.model_copy(update={"source_text": "Incorrect source"})
    tampered = replace_application(proof, old_application, application)

    with pytest.raises(ProofVerificationError, match="source_text"):
        ProofVerifier().verify_proof(theory, tampered)


def test_changed_substitution_is_rejected_after_rehashing_graph() -> None:
    theory = load_theory("binary-join.json")
    proof = ForwardChainingEngine().reason(theory).result.proof
    old_application = next(node for node in proof.nodes if isinstance(node, RuleApplicationNode))
    bindings = list(old_application.substitution)
    bindings[-1] = bindings[-1].model_copy(update={"entity": "bob"})
    application = old_application.model_copy(update={"substitution": tuple(bindings)})
    tampered = replace_application(proof, old_application, application)

    with pytest.raises(ProofVerificationError, match=r"premise|conclusion"):
        ProofVerifier().verify_proof(theory, tampered)


def test_wrong_rule_and_missing_premises_are_rejected_after_rehashing() -> None:
    theory = load_theory("unary-multistep.json")
    proof = ForwardChainingEngine().reason(theory).result.proof
    old_application = next(
        node
        for node in proof.nodes
        if isinstance(node, RuleApplicationNode) and node.rule_id == "r2"
    )
    wrong_rule = replace_application(
        proof,
        old_application,
        old_application.model_copy(
            update={"rule_id": "r1", "source_id": "s2", "source_text": "Every kind entity is calm."}
        ),
    )
    with pytest.raises(ProofVerificationError, match=r"premise|conclusion"):
        ProofVerifier().verify_proof(theory, wrong_rule)

    missing = replace_application(
        proof,
        old_application,
        old_application.model_copy(update={"premise_node_ids": ()}),
    )
    with pytest.raises(ProofVerificationError, match="premise_count"):
        ProofVerifier().verify_proof(theory, missing)


def test_unreachable_and_duplicate_nodes_are_rejected() -> None:
    theory = load_theory("multiple-derivations.json")
    proof = ForwardChainingEngine().reason(theory).result.proof
    unused_fact = theory.facts[1]
    unused = make_source_derivation(
        canonical_literal(unused_fact),
        source_id=unused_fact.source_id,
        source_text=theory.source_statements[1].text,
    )
    extra_nodes = tuple(
        sorted((*proof.nodes, *unused.nodes.values()), key=lambda node: node.node_id)
    )
    with_extra = proof.model_copy(update={"nodes": extra_nodes})
    with_extra = with_extra.model_copy(update={"proof_hash": recompute_proof_hash(with_extra)})
    with pytest.raises(ProofVerificationError, match="unreachable_node"):
        ProofVerifier().verify_proof(theory, with_extra)

    duplicated = proof.model_copy(update={"nodes": (*proof.nodes, proof.nodes[-1])})
    duplicated = duplicated.model_copy(update={"proof_hash": recompute_proof_hash(duplicated)})
    with pytest.raises(ProofVerificationError, match="duplicate_node"):
        ProofVerifier().verify_proof(theory, duplicated)


def test_cyclic_reference_graph_is_rejected() -> None:
    with pytest.raises(ProofVerificationError, match="cycle"):
        _assert_acyclic({"a": ("b",), "b": ("a",)}, None)


def test_wrong_root_and_incomplete_inconsistent_proof_are_rejected() -> None:
    theory = load_theory("entailed.json")
    proof = ForwardChainingEngine().reason(theory).result.proof
    wrong_root = proof.model_copy(update={"support_root_id": None})
    wrong_root = wrong_root.model_copy(update={"proof_hash": recompute_proof_hash(wrong_root)})
    with pytest.raises(ProofVerificationError, match="support_root"):
        ProofVerifier().verify_proof(theory, wrong_root)

    inconsistent_theory = load_theory("inconsistent.json")
    inconsistent = ForwardChainingEngine().reason(inconsistent_theory).result.proof
    incomplete = inconsistent.model_copy(update={"opposition_root_id": None})
    incomplete = incomplete.model_copy(update={"proof_hash": recompute_proof_hash(incomplete)})
    with pytest.raises(ProofVerificationError, match="opposition_root"):
        ProofVerifier().verify_proof(inconsistent_theory, incomplete)


def test_wrong_depth_dangling_and_extra_premises_are_rejected() -> None:
    theory = load_theory("unary-multistep.json")
    proof = ForwardChainingEngine().reason(theory).result.proof
    old_application = next(
        node
        for node in proof.nodes
        if isinstance(node, RuleApplicationNode) and node.rule_id == "r2"
    )
    wrong_depth = replace_application(
        proof,
        old_application,
        old_application.model_copy(update={"depth": old_application.depth + 1}),
    )
    with pytest.raises(ProofVerificationError, match="depth"):
        ProofVerifier().verify_proof(theory, wrong_depth)

    dangling = replace_application(
        proof,
        old_application,
        old_application.model_copy(update={"premise_node_ids": ("0" * 64,)}),
    )
    with pytest.raises(ProofVerificationError, match="premise"):
        ProofVerifier().verify_proof(theory, dangling)

    extra = replace_application(
        proof,
        old_application,
        old_application.model_copy(
            update={"premise_node_ids": (*old_application.premise_node_ids, proof.nodes[0].node_id)}
        ),
    )
    with pytest.raises(ProofVerificationError, match="premise_count"):
        ProofVerifier().verify_proof(theory, extra)


def test_independent_verifier_rejects_a_fabricated_unknown_result() -> None:
    theory = load_theory("entailed.json")
    query = ForwardChainingEngine().reason(theory).result.query
    payload = {
        "schema_version": "1.0",
        "theory_hash": theory_hash(theory),
        "query": query.model_dump(mode="json"),
        "status": ReasoningStatus.UNKNOWN.value,
        "support_root_id": None,
        "opposition_root_id": None,
        "nodes": [],
    }
    from verilogic_ns_api.reasoning.models import sha256_payload

    fabricated = ProofDAG(**payload, proof_hash=sha256_payload(payload))

    with pytest.raises(ProofVerificationError, match="incorrect_status"):
        ProofVerifier().verify_proof(theory, fabricated)


def test_proof_is_bound_to_the_exact_theory() -> None:
    theory = load_theory("entailed.json")
    proof = ForwardChainingEngine().reason(theory).result.proof
    payload = copy.deepcopy(theory.model_dump(mode="json"))
    payload["source_statements"][0]["text"] += " changed"
    modified = Theory.model_validate(payload)

    with pytest.raises(ProofVerificationError, match="theory_hash"):
        ProofVerifier().verify_proof(modified, proof)
