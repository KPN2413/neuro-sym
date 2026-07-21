from __future__ import annotations

from collections.abc import Mapping

from verilogic_ns_api.reasoning.configuration import (
    ProofVerificationError,
    ReasoningLimits,
    ResourceLimitError,
)
from verilogic_ns_api.reasoning.models import (
    CanonicalLiteral,
    DerivedLiteralNode,
    EntityTerm,
    ProofDAG,
    ProofNode,
    ProofVerificationResult,
    ReasoningResult,
    ReasoningStatus,
    Rule,
    RuleApplicationNode,
    RuleLiteral,
    SourceFactNode,
    Theory,
    VariableTerm,
    canonical_literal,
    theory_hash,
)
from verilogic_ns_api.reasoning.proofs import recompute_node_id, recompute_proof_hash


class ProofVerifier:
    def __init__(self, limits: ReasoningLimits | None = None) -> None:
        self.limits = limits or ReasoningLimits()

    def verify_result(self, theory: Theory, result: ReasoningResult) -> ProofVerificationResult:
        if result.theory_id != theory.theory_id:
            self._fail("theory_id", "result theory ID does not match the supplied theory")
        if result.query != canonical_literal(theory.query):
            self._fail("query", "result query does not match the supplied theory")
        closure = self._naive_closure(theory)
        conflicts = _naive_conflicts(closure)
        if result.closure_contains_conflicts != bool(conflicts):
            self._fail("conflict_flag", "closure conflict flag is incorrect")
        if result.conflict_count != len(conflicts):
            self._fail("conflict_count", "closure conflict count is incorrect")
        return self.verify_proof(
            theory, result.proof, expected_status=result.status, closure=closure
        )

    def verify_proof(
        self,
        theory: Theory,
        proof: ProofDAG,
        *,
        expected_status: ReasoningStatus | None = None,
        closure: set[CanonicalLiteral] | None = None,
    ) -> ProofVerificationResult:
        if proof.schema_version != "1.0":
            self._fail("schema_version", "unsupported proof schema version")
        expected_theory_hash = theory_hash(theory)
        if proof.theory_hash != expected_theory_hash:
            self._fail("theory_hash", "proof belongs to a different or modified theory")
        query = canonical_literal(theory.query)
        if proof.query != query:
            self._fail("query", "proof query does not match the theory query")
        if expected_status is not None and proof.status is not expected_status:
            self._fail("status", "proof status does not match the reasoning result")
        if recompute_proof_hash(proof) != proof.proof_hash:
            self._fail("proof_hash", "proof hash is incorrect")
        if len(proof.nodes) > self.limits.max_proof_nodes:
            raise ResourceLimitError(
                "max_proof_nodes", self.limits.max_proof_nodes, len(proof.nodes)
            )

        nodes: dict[str, ProofNode] = {}
        for node in proof.nodes:
            if node.node_id in nodes:
                self._fail("duplicate_node", f"duplicate proof node ID {node.node_id}")
            if recompute_node_id(node) != node.node_id:
                self._fail("node_hash", f"proof node hash is incorrect for {node.node_id}")
            nodes[node.node_id] = node
        if tuple(nodes) != tuple(sorted(nodes)):
            self._fail("node_order", "proof nodes are not in canonical order")

        roots = self._validate_roots(proof, nodes, query)
        graph = self._validate_nodes(theory, nodes)
        reachable = _reachable_nodes(graph, roots)
        if reachable != set(nodes):
            extras = sorted(set(nodes) - reachable)
            self._fail("unreachable_node", f"proof contains unreachable nodes: {extras[:3]}")

        independently_derived = closure if closure is not None else self._naive_closure(theory)
        actual_status = _classify(query, independently_derived)
        if proof.status is not actual_status:
            self._fail(
                "incorrect_status",
                f"proof claims {proof.status.value}, independent closure gives {actual_status.value}",
            )
        return ProofVerificationResult(
            status=proof.status,
            proof_hash=proof.proof_hash,
            node_count=len(nodes),
        )

    def _validate_roots(
        self,
        proof: ProofDAG,
        nodes: Mapping[str, ProofNode],
        query: CanonicalLiteral,
    ) -> tuple[str, ...]:
        support_required = proof.status in {ReasoningStatus.ENTAILED, ReasoningStatus.INCONSISTENT}
        opposition_required = proof.status in {
            ReasoningStatus.CONTRADICTED,
            ReasoningStatus.INCONSISTENT,
        }
        if support_required != (proof.support_root_id is not None):
            self._fail("support_root", "support-root presence does not match status")
        if opposition_required != (proof.opposition_root_id is not None):
            self._fail("opposition_root", "opposition-root presence does not match status")
        if proof.status is ReasoningStatus.UNKNOWN and proof.nodes:
            self._fail("unknown_proof", "UNKNOWN must not contain a fabricated proof")

        roots: list[str] = []
        if proof.support_root_id is not None:
            node = nodes.get(proof.support_root_id)
            if not _is_literal_node(node) or node.literal != query:
                self._fail("support_root", "support root does not prove the query")
            roots.append(proof.support_root_id)
        if proof.opposition_root_id is not None:
            node = nodes.get(proof.opposition_root_id)
            if not _is_literal_node(node) or node.literal != query.opposite():
                self._fail("opposition_root", "opposition root does not prove the query complement")
            roots.append(proof.opposition_root_id)
        return tuple(roots)

    def _validate_nodes(
        self, theory: Theory, nodes: Mapping[str, ProofNode]
    ) -> dict[str, tuple[str, ...]]:
        sources = {source.id: source.text for source in theory.source_statements}
        rules = {rule.id: rule for rule in theory.rules}
        entity_ids = {entity.id for entity in theory.entities}
        graph: dict[str, tuple[str, ...]] = {}
        for node_id, node in nodes.items():
            if isinstance(node, SourceFactNode):
                matches = [
                    fact
                    for fact in theory.facts
                    if fact.source_id == node.source_id and canonical_literal(fact) == node.literal
                ]
                if not matches:
                    self._fail("source_fact", f"node {node_id} is not an exact source fact")
                if sources.get(node.source_id) != node.source_text:
                    self._fail("source_text", f"node {node_id} has incorrect source text")
                if node.depth != 0:
                    self._fail("depth", f"source fact {node_id} must have depth zero")
                graph[node_id] = ()
                continue

            if isinstance(node, RuleApplicationNode):
                rule = rules.get(node.rule_id)
                if rule is None or rule.source_id != node.source_id:
                    self._fail("rule", f"node {node_id} references the wrong rule")
                if sources.get(node.source_id) != node.source_text:
                    self._fail("source_text", f"node {node_id} has incorrect rule source text")
                substitution_items = tuple(
                    (binding.variable, binding.entity) for binding in node.substitution
                )
                if substitution_items != tuple(sorted(substitution_items)):
                    self._fail(
                        "substitution_order", f"node {node_id} substitution is not canonical"
                    )
                substitution = dict(substitution_items)
                if len(substitution) != len(substitution_items):
                    self._fail("substitution", f"node {node_id} has duplicate variable bindings")
                if not set(substitution.values()).issubset(entity_ids):
                    self._fail("substitution", f"node {node_id} binds an undeclared entity")
                required = _used_variables(rule)
                if set(substitution) != required:
                    self._fail(
                        "substitution", f"node {node_id} does not ground every used variable"
                    )
                ordered_body = tuple(sorted(rule.body, key=_rule_literal_key))
                if len(node.premise_node_ids) != len(ordered_body):
                    self._fail("premise_count", f"node {node_id} has the wrong premise count")
                premise_depths: list[int] = []
                for pattern, premise_id in zip(ordered_body, node.premise_node_ids, strict=True):
                    premise = nodes.get(premise_id)
                    if not _is_literal_node(premise):
                        self._fail("premise", f"node {node_id} has a missing or invalid premise")
                    if premise.literal != _ground_independently(pattern, substitution):
                        self._fail(
                            "premise", f"node {node_id} premise does not match its rule body"
                        )
                    premise_depths.append(premise.depth)
                expected_conclusion = _ground_independently(rule.head, substitution)
                if node.conclusion != expected_conclusion:
                    self._fail("conclusion", f"node {node_id} has the wrong grounded conclusion")
                expected_depth = 1 + max(premise_depths)
                if node.depth != expected_depth:
                    self._fail("depth", f"node {node_id} has incorrect application depth")
                graph[node_id] = node.premise_node_ids
                continue

            if isinstance(node, DerivedLiteralNode):
                application = nodes.get(node.rule_application_node_id)
                if not isinstance(application, RuleApplicationNode):
                    self._fail("application", f"node {node_id} has a dangling rule application")
                if node.literal != application.conclusion:
                    self._fail("conclusion", f"node {node_id} differs from its rule conclusion")
                if node.depth != application.depth:
                    self._fail("depth", f"node {node_id} has incorrect derived depth")
                graph[node_id] = (node.rule_application_node_id,)
                continue
            self._fail("node_type", f"node {node_id} has an unsupported type")
        _assert_acyclic(graph, self._fail)
        return graph

    def _naive_closure(self, theory: Theory) -> set[CanonicalLiteral]:
        closure = {canonical_literal(fact) for fact in theory.facts}
        if len(closure) > self.limits.max_derived_literals:
            raise ResourceLimitError(
                "max_derived_literals", self.limits.max_derived_literals, len(closure)
            )
        firings = 0
        rounds = 0
        while True:
            new_literals: set[CanonicalLiteral] = set()
            for rule in theory.rules:
                for substitution in _naive_matches(rule.body, closure):
                    firings += 1
                    if firings > self.limits.max_rule_firings:
                        raise ResourceLimitError(
                            "max_rule_firings", self.limits.max_rule_firings, firings
                        )
                    conclusion = _ground_independently(rule.head, substitution)
                    if conclusion not in closure:
                        new_literals.add(conclusion)
            if not new_literals:
                return closure
            rounds += 1
            if rounds > self.limits.max_rounds:
                raise ResourceLimitError("max_rounds", self.limits.max_rounds, rounds)
            closure.update(new_literals)
            if len(closure) > self.limits.max_derived_literals:
                raise ResourceLimitError(
                    "max_derived_literals", self.limits.max_derived_literals, len(closure)
                )

    @staticmethod
    def _fail(code: str, message: str) -> None:
        raise ProofVerificationError(code, message)


def _rule_literal_key(literal: RuleLiteral) -> tuple[object, ...]:
    arguments = tuple(
        ("entity", term.id) if isinstance(term, EntityTerm) else ("variable", term.name)
        for term in literal.arguments
    )
    return (literal.predicate, literal.negated, arguments, literal.source_id)


def _ground_independently(
    literal: RuleLiteral, substitution: Mapping[str, str]
) -> CanonicalLiteral:
    arguments: list[str] = []
    for term in literal.arguments:
        if isinstance(term, EntityTerm):
            arguments.append(term.id)
        elif isinstance(term, VariableTerm):
            if term.name not in substitution:
                raise ProofVerificationError(
                    "substitution", f"missing variable binding {term.name!r}"
                )
            arguments.append(substitution[term.name])
    return CanonicalLiteral(
        predicate=literal.predicate,
        arguments=tuple(arguments),
        negated=literal.negated,
    )


def _naive_matches(
    body: tuple[RuleLiteral, ...], closure: set[CanonicalLiteral]
) -> list[dict[str, str]]:
    states: list[dict[str, str]] = [{}]
    for pattern in sorted(body, key=_rule_literal_key):
        candidates = sorted(
            (
                literal
                for literal in closure
                if literal.predicate == pattern.predicate
                and literal.negated == pattern.negated
                and len(literal.arguments) == len(pattern.arguments)
            ),
            key=CanonicalLiteral.sort_key,
        )
        next_states: list[dict[str, str]] = []
        for state in states:
            for candidate in candidates:
                matched = dict(state)
                valid = True
                for term, entity in zip(pattern.arguments, candidate.arguments, strict=True):
                    if isinstance(term, EntityTerm):
                        if term.id != entity:
                            valid = False
                            break
                    else:
                        previous = matched.get(term.name)
                        if previous is not None and previous != entity:
                            valid = False
                            break
                        matched[term.name] = entity
                if valid:
                    next_states.append(matched)
        states = _deduplicate_substitutions(next_states)
        if not states:
            break
    return states


def _deduplicate_substitutions(items: list[dict[str, str]]) -> list[dict[str, str]]:
    unique = {tuple(sorted(item.items())): item for item in items}
    return [unique[key] for key in sorted(unique)]


def _used_variables(rule: Rule) -> set[str]:
    result: set[str] = set()
    for literal in (*rule.body, rule.head):
        result.update(term.name for term in literal.arguments if isinstance(term, VariableTerm))
    return result


def _is_literal_node(node: ProofNode | None) -> bool:
    return isinstance(node, SourceFactNode | DerivedLiteralNode)


def _reachable_nodes(graph: Mapping[str, tuple[str, ...]], roots: tuple[str, ...]) -> set[str]:
    reachable: set[str] = set()
    stack = list(roots)
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        if node_id not in graph:
            raise ProofVerificationError("dangling_reference", f"unknown node {node_id}")
        reachable.add(node_id)
        stack.extend(graph[node_id])
    return reachable


def _assert_acyclic(graph: Mapping[str, tuple[str, ...]], fail: object) -> None:
    del fail
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ProofVerificationError("cycle", "proof graph contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for child in graph.get(node_id, ()):
            if child not in graph:
                raise ProofVerificationError(
                    "dangling_reference", f"proof references unknown node {child}"
                )
            visit(child)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in graph:
        visit(node_id)


def _classify(query: CanonicalLiteral, closure: set[CanonicalLiteral]) -> ReasoningStatus:
    support = query in closure
    opposition = query.opposite() in closure
    if support and opposition:
        return ReasoningStatus.INCONSISTENT
    if support:
        return ReasoningStatus.ENTAILED
    if opposition:
        return ReasoningStatus.CONTRADICTED
    return ReasoningStatus.UNKNOWN


def _naive_conflicts(closure: set[CanonicalLiteral]) -> set[CanonicalLiteral]:
    return {
        literal.model_copy(update={"negated": False})
        for literal in closure
        if literal.negated and literal.opposite() in closure
    }
