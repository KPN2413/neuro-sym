from __future__ import annotations

from dataclasses import dataclass

from verilogic_ns_api.reasoning.configuration import ReasoningLimits, ResourceLimitError
from verilogic_ns_api.reasoning.models import (
    CanonicalLiteral,
    DerivedLiteralNode,
    ProofDAG,
    ProofNode,
    ReasoningStatus,
    Rule,
    RuleApplicationNode,
    SourceFactNode,
    SubstitutionBinding,
    Theory,
    sha256_payload,
    theory_hash,
)


@dataclass(frozen=True)
class Derivation:
    literal: CanonicalLiteral
    depth: int
    root_id: str
    nodes: dict[str, ProofNode]
    choice_key: tuple[object, ...]


def make_source_derivation(
    literal: CanonicalLiteral,
    *,
    source_id: str,
    source_text: str,
) -> Derivation:
    payload = {
        "node_type": "source_fact",
        "literal": literal.model_dump(mode="json"),
        "source_id": source_id,
        "source_text": source_text,
        "depth": 0,
    }
    node_id = sha256_payload(payload)
    node = SourceFactNode(node_id=node_id, **payload)
    return Derivation(
        literal=literal,
        depth=0,
        root_id=node_id,
        nodes={node_id: node},
        choice_key=(0, 1, source_id, node_id),
    )


def make_rule_derivation(
    literal: CanonicalLiteral,
    *,
    rule: Rule,
    substitution: dict[str, str],
    premises: tuple[Derivation, ...],
    source_text: str,
) -> Derivation:
    depth = 1 + max(premise.depth for premise in premises)
    bindings = tuple(
        SubstitutionBinding(variable=variable, entity=entity)
        for variable, entity in sorted(substitution.items())
    )
    premise_ids = tuple(premise.root_id for premise in premises)
    application_payload = {
        "node_type": "rule_application",
        "rule_id": rule.id,
        "source_id": rule.source_id,
        "source_text": source_text,
        "substitution": [binding.model_dump(mode="json") for binding in bindings],
        "premise_node_ids": list(premise_ids),
        "conclusion": literal.model_dump(mode="json"),
        "depth": depth,
    }
    application_id = sha256_payload(application_payload)
    application = RuleApplicationNode(node_id=application_id, **application_payload)
    derived_payload = {
        "node_type": "derived_literal",
        "literal": literal.model_dump(mode="json"),
        "rule_application_node_id": application_id,
        "depth": depth,
    }
    root_id = sha256_payload(derived_payload)
    derived = DerivedLiteralNode(node_id=root_id, **derived_payload)
    nodes: dict[str, ProofNode] = {}
    for premise in premises:
        nodes.update(premise.nodes)
    nodes[application_id] = application
    nodes[root_id] = derived
    stable_tie = sha256_payload(
        {
            "rule_id": rule.id,
            "substitution": list(sorted(substitution.items())),
            "premises": list(premise_ids),
            "literal": literal.model_dump(mode="json"),
        }
    )
    return Derivation(
        literal=literal,
        depth=depth,
        root_id=root_id,
        nodes=nodes,
        choice_key=(
            depth,
            len(nodes),
            rule.id,
            tuple(sorted(substitution.items())),
            premise_ids,
            stable_tie,
        ),
    )


def build_proof(
    theory: Theory,
    *,
    query: CanonicalLiteral,
    status: ReasoningStatus,
    support: Derivation | None,
    opposition: Derivation | None,
    limits: ReasoningLimits,
) -> ProofDAG:
    nodes: dict[str, ProofNode] = {}
    if support is not None:
        nodes.update(support.nodes)
    if opposition is not None:
        nodes.update(opposition.nodes)
    if len(nodes) > limits.max_proof_nodes:
        raise ResourceLimitError("max_proof_nodes", limits.max_proof_nodes, len(nodes))
    ordered_nodes = tuple(nodes[node_id] for node_id in sorted(nodes))
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "theory_hash": theory_hash(theory),
        "query": query.model_dump(mode="json"),
        "status": status.value,
        "support_root_id": support.root_id if support is not None else None,
        "opposition_root_id": opposition.root_id if opposition is not None else None,
        "nodes": [node.model_dump(mode="json") for node in ordered_nodes],
    }
    return ProofDAG(**payload, proof_hash=sha256_payload(payload))


def recompute_node_id(node: ProofNode) -> str:
    payload = node.model_dump(mode="json", exclude={"node_id"})
    return sha256_payload(payload)


def recompute_proof_hash(proof: ProofDAG) -> str:
    payload = proof.model_dump(mode="json", exclude={"proof_hash"})
    return sha256_payload(payload)
