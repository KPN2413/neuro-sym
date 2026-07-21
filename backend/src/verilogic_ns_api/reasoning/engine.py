from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from verilogic_ns_api.reasoning.configuration import ReasoningLimits, ResourceLimitError
from verilogic_ns_api.reasoning.models import (
    CanonicalLiteral,
    ClosureEntry,
    ReasoningOutput,
    ReasoningResult,
    ReasoningStatus,
    ReasoningTelemetry,
    SaturationOutput,
    Theory,
    canonical_literal,
    theory_hash,
)
from verilogic_ns_api.reasoning.proofs import (
    Derivation,
    build_proof,
    make_rule_derivation,
    make_source_derivation,
)
from verilogic_ns_api.reasoning.unification import (
    IndexKey,
    ground_literal,
    iter_body_matches,
    literal_index_key,
    rule_literal_sort_key,
)


@dataclass(frozen=True)
class _SaturationState:
    closure: dict[CanonicalLiteral, Derivation]
    conflicts: tuple[CanonicalLiteral, ...]
    telemetry: ReasoningTelemetry


class ForwardChainingEngine:
    def __init__(self, limits: ReasoningLimits | None = None) -> None:
        self.limits = limits or ReasoningLimits()

    def reason(self, theory: Theory) -> ReasoningOutput:
        state = self._saturate(theory)
        query = canonical_literal(theory.query)
        opposite = query.opposite()
        support = state.closure.get(query)
        opposition = state.closure.get(opposite)
        if support is not None and opposition is not None:
            status = ReasoningStatus.INCONSISTENT
        elif support is not None:
            status = ReasoningStatus.ENTAILED
        elif opposition is not None:
            status = ReasoningStatus.CONTRADICTED
        else:
            status = ReasoningStatus.UNKNOWN
        proof = build_proof(
            theory,
            query=query,
            status=status,
            support=support,
            opposition=opposition,
            limits=self.limits,
        )
        result = ReasoningResult(
            theory_id=theory.theory_id,
            status=status,
            query=query,
            closure_contains_conflicts=bool(state.conflicts),
            conflict_count=len(state.conflicts),
            proof=proof,
        )
        return ReasoningOutput(result=result, telemetry=state.telemetry)

    def saturate(self, theory: Theory) -> SaturationOutput:
        state = self._saturate(theory)
        entries = tuple(
            ClosureEntry(literal=literal, depth=derivation.depth)
            for literal, derivation in sorted(
                state.closure.items(), key=lambda item: item[0].sort_key()
            )
        )
        return SaturationOutput(
            theory_id=theory.theory_id,
            theory_hash=theory_hash(theory),
            closure=entries,
            conflicts=state.conflicts,
            telemetry=state.telemetry,
        )

    def _saturate(self, theory: Theory) -> _SaturationState:
        started = time.perf_counter()
        sources = {source.id: source.text for source in theory.source_statements}
        closure: dict[CanonicalLiteral, Derivation] = {}
        duplicate_conclusions = 0
        for fact in sorted(
            theory.facts,
            key=lambda item: (
                item.predicate,
                item.negated,
                tuple(term.id for term in item.arguments),
                item.source_id,
            ),
        ):
            literal = canonical_literal(fact)
            candidate = make_source_derivation(
                literal,
                source_id=fact.source_id,
                source_text=sources[fact.source_id],
            )
            current = closure.get(literal)
            if current is None or candidate.choice_key < current.choice_key:
                closure[literal] = candidate
            else:
                duplicate_conclusions += 1
        self._check_limit("max_derived_literals", len(closure))

        initial_unique_count = len(closure)
        delta = set(closure)
        rounds = 0
        rule_instances = 0
        successful_firings = 0
        ordered_rules = tuple(
            sorted(
                theory.rules,
                key=lambda rule: (
                    rule.id,
                    tuple(
                        rule_literal_sort_key(literal)
                        for literal in sorted(rule.body, key=rule_literal_sort_key)
                    ),
                ),
            )
        )
        while delta:
            self._check_timeout(started)
            index = _build_index(closure)
            candidates: dict[CanonicalLiteral, Derivation] = {}
            for rule in ordered_rules:
                for substitution, premise_literals in iter_body_matches(rule.body, index, delta):
                    rule_instances += 1
                    self._check_limit("max_rule_firings", rule_instances)
                    conclusion = ground_literal(rule.head, substitution)
                    premise_derivations = tuple(closure[premise] for premise in premise_literals)
                    candidate = make_rule_derivation(
                        conclusion,
                        rule=rule,
                        substitution=substitution,
                        premises=premise_derivations,
                        source_text=sources[rule.source_id],
                    )
                    self._check_limit("max_proof_nodes", len(candidate.nodes))
                    if conclusion in closure:
                        duplicate_conclusions += 1
                        continue
                    previous = candidates.get(conclusion)
                    if previous is None:
                        candidates[conclusion] = candidate
                        successful_firings += 1
                    elif candidate.choice_key < previous.choice_key:
                        candidates[conclusion] = candidate
                        duplicate_conclusions += 1
                    else:
                        duplicate_conclusions += 1
                self._check_timeout(started)

            if not candidates:
                break
            if rounds + 1 > self.limits.max_rounds:
                raise ResourceLimitError("max_rounds", self.limits.max_rounds, rounds + 1)
            if len(closure) + len(candidates) > self.limits.max_derived_literals:
                raise ResourceLimitError(
                    "max_derived_literals",
                    self.limits.max_derived_literals,
                    len(closure) + len(candidates),
                )
            delta = set(candidates)
            closure.update(candidates)
            rounds += 1

        conflicts = _find_conflicts(closure)
        elapsed_ms = (time.perf_counter() - started) * 1000
        telemetry = ReasoningTelemetry(
            initial_fact_count=len(theory.facts),
            derived_fact_count=len(closure) - initial_unique_count,
            total_closure_size=len(closure),
            conflict_count=len(conflicts),
            rounds=rounds,
            rule_instances_considered=rule_instances,
            successful_rule_firings=successful_firings,
            duplicate_conclusions=duplicate_conclusions,
            maximum_proof_depth=max((item.depth for item in closure.values()), default=0),
            execution_duration_ms=elapsed_ms,
            resource_limits=self.limits.as_dict(),
        )
        return _SaturationState(closure=closure, conflicts=conflicts, telemetry=telemetry)

    def _check_limit(self, name: str, observed: int) -> None:
        limit = int(getattr(self.limits, name))
        if observed > limit:
            raise ResourceLimitError(name, limit, observed)

    def _check_timeout(self, started: float) -> None:
        if self.limits.timeout_seconds is None:
            return
        elapsed = time.perf_counter() - started
        if elapsed > self.limits.timeout_seconds:
            raise ResourceLimitError("timeout_seconds", self.limits.timeout_seconds, elapsed)


def _build_index(
    closure: dict[CanonicalLiteral, Derivation],
) -> dict[IndexKey, tuple[CanonicalLiteral, ...]]:
    grouped: defaultdict[IndexKey, list[CanonicalLiteral]] = defaultdict(list)
    for literal in closure:
        grouped[literal_index_key(literal)].append(literal)
    return {
        key: tuple(sorted(literals, key=CanonicalLiteral.sort_key))
        for key, literals in grouped.items()
    }


def _find_conflicts(
    closure: dict[CanonicalLiteral, Derivation],
) -> tuple[CanonicalLiteral, ...]:
    conflicts = {
        literal.model_copy(update={"negated": False})
        for literal in closure
        if literal.negated and literal.opposite() in closure
    }
    return tuple(sorted(conflicts, key=CanonicalLiteral.sort_key))
