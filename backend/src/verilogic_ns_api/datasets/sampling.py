from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from pydantic import Field

from verilogic_ns_api.datasets.errors import SamplingError
from verilogic_ns_api.research.models import (
    BenchmarkExample,
    GoldLabel,
    SamplingConfig,
    SamplingStrategy,
    Split,
    StrictModel,
)


def _order_key(example: BenchmarkExample, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{example.example_id}".encode()).hexdigest()
    return digest, example.example_id


def _take_deterministic(
    examples: list[BenchmarkExample], count: int, seed: int
) -> list[BenchmarkExample]:
    return sorted(examples, key=lambda example: _order_key(example, seed))[:count]


def _balanced_sample(
    examples: list[BenchmarkExample], config: SamplingConfig
) -> list[BenchmarkExample]:
    groups: dict[GoldLabel, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        groups[example.gold_label].append(example)
    requested_labels = config.labels or list(GoldLabel)
    missing = [label.value for label in requested_labels if not groups[label]]
    if missing:
        raise SamplingError(f"Balanced sampling has no examples for labels: {', '.join(missing)}")

    label_count = len(requested_labels)
    if config.max_examples is None:
        quota = min(len(groups[label]) for label in requested_labels)
        quotas = {label: quota for label in requested_labels}
    else:
        if config.max_examples < label_count:
            raise SamplingError(
                "Balanced sample size must be at least the number of requested labels"
            )
        base, remainder = divmod(config.max_examples, label_count)
        quotas = {label: base + (index < remainder) for index, label in enumerate(requested_labels)}
    impossible = {
        label.value: quotas[label] - len(groups[label])
        for label in requested_labels
        if quotas[label] > len(groups[label])
    }
    if impossible:
        raise SamplingError(f"Balanced sample request exceeds available groups: {impossible}")

    selected: list[BenchmarkExample] = []
    for label in requested_labels:
        selected.extend(_take_deterministic(groups[label], quotas[label], config.seed))
    return sorted(selected, key=lambda example: _order_key(example, config.seed))


def _stratified_sample(
    examples: list[BenchmarkExample], config: SamplingConfig
) -> list[BenchmarkExample]:
    if config.max_examples is None or config.max_examples >= len(examples):
        return sorted(examples, key=lambda example: _order_key(example, config.seed))
    groups: dict[GoldLabel, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        groups[example.gold_label].append(example)

    target = config.max_examples
    exact = {label: len(group) * target / len(examples) for label, group in groups.items()}
    quotas = {label: math.floor(value) for label, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(groups, key=lambda label: (-(exact[label] - quotas[label]), label.value))
    for label in order:
        if remaining == 0:
            break
        if quotas[label] < len(groups[label]):
            quotas[label] += 1
            remaining -= 1
    if remaining:
        raise SamplingError("Unable to allocate the requested stratified sample")

    selected: list[BenchmarkExample] = []
    for label, quota in quotas.items():
        selected.extend(_take_deterministic(groups[label], quota, config.seed))
    return sorted(selected, key=lambda example: _order_key(example, config.seed))


def sample_examples(
    examples: Iterable[BenchmarkExample], config: SamplingConfig
) -> list[BenchmarkExample]:
    if Split.TEST in config.allowed_splits and not config.allow_test:
        raise SamplingError("Selecting the test split requires allow_test=true")

    allowed_splits = set(config.allowed_splits)
    allowed_labels = set(config.labels) if config.labels else None
    allowed_depths = set(config.reasoning_depths) if config.reasoning_depths else None

    def is_eligible(example: BenchmarkExample) -> bool:
        return (
            example.split in allowed_splits
            and (allowed_labels is None or example.gold_label in allowed_labels)
            and (allowed_depths is None or example.reasoning_depth in allowed_depths)
        )

    if config.max_examples is not None:
        return _bounded_sample((example for example in examples if is_eligible(example)), config)

    eligible = [example for example in examples if is_eligible(example)]
    if not eligible:
        raise SamplingError("No examples match the sampling configuration")

    if config.strategy is SamplingStrategy.BALANCED:
        return _balanced_sample(eligible, config)
    if config.strategy is SamplingStrategy.STRATIFIED:
        return _stratified_sample(eligible, config)
    count = min(config.max_examples or len(eligible), len(eligible))
    return _take_deterministic(eligible, count, config.seed)


def _retain_best(
    candidates: list[BenchmarkExample],
    example: BenchmarkExample,
    *,
    capacity: int,
    seed: int,
) -> None:
    candidates.append(example)
    candidates.sort(key=lambda item: _order_key(item, seed))
    if len(candidates) > capacity:
        candidates.pop()


def _bounded_sample(
    examples: Iterable[BenchmarkExample], config: SamplingConfig
) -> list[BenchmarkExample]:
    """Select deterministically while retaining only O(max_examples) records per stratum."""
    target = config.max_examples
    if target is None:
        raise AssertionError("bounded sampling requires max_examples")

    if config.strategy is SamplingStrategy.RANDOM:
        selected: list[BenchmarkExample] = []
        eligible_count = 0
        for example in examples:
            eligible_count += 1
            _retain_best(selected, example, capacity=target, seed=config.seed)
        if not eligible_count:
            raise SamplingError("No examples match the sampling configuration")
        return selected

    groups: dict[GoldLabel, list[BenchmarkExample]] = defaultdict(list)
    available: Counter[GoldLabel] = Counter()

    if config.strategy is SamplingStrategy.BALANCED:
        requested_labels = config.labels or list(GoldLabel)
        if target < len(requested_labels):
            raise SamplingError(
                "Balanced sample size must be at least the number of requested labels"
            )
        base, remainder = divmod(target, len(requested_labels))
        quotas = {label: base + (index < remainder) for index, label in enumerate(requested_labels)}
        for example in examples:
            available[example.gold_label] += 1
            if example.gold_label in quotas:
                _retain_best(
                    groups[example.gold_label],
                    example,
                    capacity=quotas[example.gold_label],
                    seed=config.seed,
                )
        missing = [label.value for label in requested_labels if not available[label]]
        if missing:
            raise SamplingError(
                f"Balanced sampling has no examples for labels: {', '.join(missing)}"
            )
        impossible = {
            label.value: quotas[label] - available[label]
            for label in requested_labels
            if quotas[label] > available[label]
        }
        if impossible:
            raise SamplingError(f"Balanced sample request exceeds available groups: {impossible}")
        selected = [example for label in requested_labels for example in groups[label]]
        return sorted(selected, key=lambda example: _order_key(example, config.seed))

    for example in examples:
        available[example.gold_label] += 1
        _retain_best(
            groups[example.gold_label],
            example,
            capacity=target,
            seed=config.seed,
        )
    eligible_count = sum(available.values())
    if not eligible_count:
        raise SamplingError("No examples match the sampling configuration")
    if target >= eligible_count:
        selected = [example for group in groups.values() for example in group]
        return sorted(selected, key=lambda example: _order_key(example, config.seed))

    exact = {label: count * target / eligible_count for label, count in available.items()}
    quotas = {label: math.floor(value) for label, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(available, key=lambda label: (-(exact[label] - quotas[label]), label.value))
    for label in order:
        if not remaining:
            break
        if quotas[label] < available[label]:
            quotas[label] += 1
            remaining -= 1
    if remaining:
        raise SamplingError("Unable to allocate the requested stratified sample")
    selected = [example for label, quota in quotas.items() for example in groups[label][:quota]]
    return sorted(selected, key=lambda example: _order_key(example, config.seed))


class OverlapReport(StrictModel):
    duplicate_example_ids_within_split: dict[str, int] = Field(default_factory=dict)
    duplicate_example_ids_across_splits: dict[str, list[str]] = Field(default_factory=dict)
    duplicate_question_ids_across_splits: dict[str, list[str]] = Field(default_factory=dict)
    duplicate_context_query_pairs_across_splits: dict[str, list[str]] = Field(default_factory=dict)
    theory_overlaps_across_splits: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def duplicate_count(self) -> int:
        return sum(
            len(mapping)
            for mapping in (
                self.duplicate_example_ids_within_split,
                self.duplicate_example_ids_across_splits,
                self.duplicate_question_ids_across_splits,
                self.duplicate_context_query_pairs_across_splits,
                self.theory_overlaps_across_splits,
            )
        )


class LeakageAccumulator:
    """Track hashes and identifiers without retaining full examples."""

    def __init__(self) -> None:
        self._example_occurrences: Counter[tuple[str, Split]] = Counter()
        self._example_splits: dict[str, set[Split]] = defaultdict(set)
        self._question_splits: dict[str, set[Split]] = defaultdict(set)
        self._content_splits: dict[str, set[Split]] = defaultdict(set)
        self._theory_splits: dict[str, set[Split]] = defaultdict(set)

    def add(self, example: BenchmarkExample) -> None:
        self._example_occurrences[(example.example_id, example.split)] += 1
        self._example_splits[example.example_id].add(example.split)
        if example.theory_id and example.question_id:
            qualified_question_id = f"{example.variant}/{example.theory_id}/{example.question_id}"
            self._question_splits[qualified_question_id].add(example.split)
        self._content_splits[example.provenance.content_sha256].add(example.split)
        if example.theory_id:
            self._theory_splits[f"{example.variant}/{example.theory_id}"].add(example.split)

    @staticmethod
    def _cross_split(mapping: dict[str, set[Split]]) -> dict[str, list[str]]:
        return {
            key: sorted(split.value for split in splits)
            for key, splits in sorted(mapping.items())
            if len(splits) > 1
        }

    def report(self) -> OverlapReport:
        within = {
            f"{split.value}:{example_id}": count
            for (example_id, split), count in sorted(
                self._example_occurrences.items(), key=lambda item: (item[0][1].value, item[0][0])
            )
            if count > 1
        }
        return OverlapReport(
            duplicate_example_ids_within_split=within,
            duplicate_example_ids_across_splits=self._cross_split(self._example_splits),
            duplicate_question_ids_across_splits=self._cross_split(self._question_splits),
            duplicate_context_query_pairs_across_splits=self._cross_split(self._content_splits),
            theory_overlaps_across_splits=self._cross_split(self._theory_splits),
        )


def detect_overlaps(examples: Iterable[BenchmarkExample]) -> OverlapReport:
    accumulator = LeakageAccumulator()
    for example in examples:
        accumulator.add(example)
    return accumulator.report()
