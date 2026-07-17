from pathlib import Path

import pytest

from verilogic_ns_api.datasets.errors import SamplingError
from verilogic_ns_api.datasets.proofwriter import ProofWriterLoader, stable_content_hash
from verilogic_ns_api.datasets.sampling import detect_overlaps, sample_examples
from verilogic_ns_api.research.models import (
    GoldLabel,
    SamplingConfig,
    SamplingStrategy,
    Split,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "proofwriter" / "proofwriter-dataset-V2020.12.3"


def development_examples():
    loader = ProofWriterLoader(FIXTURE_ROOT)
    return [
        *loader.iter_examples(variant="depth-1", split=Split.TRAIN),
        *loader.iter_examples(variant="depth-1", split=Split.DEVELOPMENT),
    ]


def test_seed_reproducibility() -> None:
    examples = development_examples()
    config = SamplingConfig(seed=81, max_examples=4)

    first = sample_examples(examples, config)
    second = sample_examples(reversed(examples), config)

    assert [item.example_id for item in first] == [item.example_id for item in second]


def test_bounded_balanced_sampling_accepts_a_one_shot_stream() -> None:
    source = (example for example in development_examples())

    selected = sample_examples(
        source,
        SamplingConfig(max_examples=3, strategy=SamplingStrategy.BALANCED),
    )

    assert len(selected) == 3
    assert {example.gold_label for example in selected} == set(GoldLabel)


def test_label_filtering() -> None:
    selected = sample_examples(
        development_examples(),
        SamplingConfig(labels=[GoldLabel.UNKNOWN]),
    )

    assert len(selected) == 2
    assert {item.gold_label for item in selected} == {GoldLabel.UNKNOWN}


def test_reasoning_depth_filtering() -> None:
    selected = sample_examples(
        development_examples(),
        SamplingConfig(reasoning_depths=[1]),
    )

    assert len(selected) == 2
    assert all(item.reasoning_depth == 1 for item in selected)


def test_balanced_sampling() -> None:
    selected = sample_examples(
        development_examples(),
        SamplingConfig(
            seed=3,
            max_examples=6,
            strategy=SamplingStrategy.BALANCED,
        ),
    )

    assert len(selected) == 6
    assert {label: sum(item.gold_label is label for item in selected) for label in GoldLabel} == {
        GoldLabel.ENTAILED: 2,
        GoldLabel.CONTRADICTED: 2,
        GoldLabel.UNKNOWN: 2,
    }


def test_stratified_sampling_preserves_equal_fixture_distribution() -> None:
    selected = sample_examples(
        development_examples(),
        SamplingConfig(
            seed=3,
            max_examples=3,
            strategy=SamplingStrategy.STRATIFIED,
        ),
    )

    assert {item.gold_label for item in selected} == set(GoldLabel)


def test_impossible_balanced_sample_is_rejected() -> None:
    with pytest.raises(SamplingError, match="exceeds available"):
        sample_examples(
            development_examples(),
            SamplingConfig(
                labels=[GoldLabel.ENTAILED],
                max_examples=5,
                strategy=SamplingStrategy.BALANCED,
            ),
        )


def test_duplicate_sampling_filters_are_rejected() -> None:
    with pytest.raises(ValueError, match="labels must not contain duplicates"):
        SamplingConfig(labels=[GoldLabel.UNKNOWN, GoldLabel.UNKNOWN])


def test_test_split_requires_explicit_flag() -> None:
    loader = ProofWriterLoader(FIXTURE_ROOT)
    test_examples = list(loader.iter_examples(variant="depth-1", split=Split.TEST))

    with pytest.raises(SamplingError, match="allow_test"):
        sample_examples(
            test_examples,
            SamplingConfig(allowed_splits=[Split.TEST]),
        )


def test_default_sampler_never_selects_test() -> None:
    loader = ProofWriterLoader(FIXTURE_ROOT)
    mixed = [
        *development_examples(),
        *loader.iter_examples(variant="depth-1", split=Split.TEST),
    ]

    selected = sample_examples(mixed, SamplingConfig())

    assert all(item.split in {Split.TRAIN, Split.DEVELOPMENT} for item in selected)


def test_overlap_report_preserves_and_reports_cross_split_overlap() -> None:
    original = development_examples()[0]
    overlapping = original.model_copy(update={"split": Split.DEVELOPMENT})

    report = detect_overlaps([original, overlapping])

    assert original.example_id in report.duplicate_example_ids_across_splits
    assert report.duplicate_question_ids_across_splits
    assert original.provenance.content_sha256 in report.duplicate_context_query_pairs_across_splits
    assert report.theory_overlaps_across_splits


def test_duplicate_within_split_is_reported() -> None:
    example = development_examples()[0]

    report = detect_overlaps([example, example])

    assert report.duplicate_example_ids_within_split[f"train:{example.example_id}"] == 2


def test_content_hash_is_stable_and_order_sensitive() -> None:
    first = stable_content_hash("A. B.", "C?")
    second = stable_content_hash("A. B.", "C?")
    changed = stable_content_hash("B. A.", "C?")

    assert first == second
    assert first != changed
