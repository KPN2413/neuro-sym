from __future__ import annotations

import json
from pathlib import Path

import pytest

from verilogic_ns_api.datasets.errors import (
    AmbiguousWorldAssumptionError,
    DatasetRecordError,
    DuplicateExampleError,
)
from verilogic_ns_api.datasets.proofwriter import (
    LoaderStats,
    ProofWriterLoader,
    map_gold_label,
)
from verilogic_ns_api.research.models import GoldLabel, Split, WorldAssumption

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "proofwriter" / "proofwriter-dataset-V2020.12.3"


def fixture_record(split: str = "train") -> dict:
    path = FIXTURE_ROOT / "OWA" / "depth-1" / f"meta-{split}.jsonl"
    return json.loads(path.read_text(encoding="utf-8").strip())


def write_source(
    root: Path,
    records: list[dict],
    *,
    world: str = "OWA",
    split: str = "train",
) -> Path:
    dataset_root = root / "proofwriter-dataset-test"
    directory = dataset_root / world / "depth-1"
    directory.mkdir(parents=True)
    path = directory / f"meta-{split}.jsonl"
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    return dataset_root


def test_valid_owa_record_streams_one_example_per_question() -> None:
    loader = ProofWriterLoader(FIXTURE_ROOT)

    examples = list(loader.iter_examples(variant="depth-1", split=Split.TRAIN))

    assert len(examples) == 3
    assert all(example.world_assumption is WorldAssumption.OPEN for example in examples)
    assert all(example.source_relative_path.endswith("meta-train.jsonl") for example in examples)


@pytest.mark.parametrize(
    ("raw_label", "question", "expected"),
    [
        (True, {"strategy": "proof"}, GoldLabel.ENTAILED),
        (
            False,
            {"strategy": "inv-proof", "proofsWithIntermediates": [{"proof": "x"}]},
            GoldLabel.CONTRADICTED,
        ),
        ("Unknown", {"strategy": "random"}, GoldLabel.UNKNOWN),
    ],
)
def test_owa_label_mapping_is_explicit(raw_label, question, expected) -> None:
    assert (
        map_gold_label(raw_label, world_assumption=WorldAssumption.OPEN, question=question)
        is expected
    )


def test_cwa_false_is_never_silently_mapped_to_contradiction() -> None:
    with pytest.raises(AmbiguousWorldAssumptionError, match="CWA"):
        map_gold_label(
            False,
            world_assumption=WorldAssumption.CLOSED,
            question={"strategy": "random"},
        )


def test_owa_false_without_opposite_proof_metadata_is_ambiguous() -> None:
    with pytest.raises(AmbiguousWorldAssumptionError, match="proof metadata"):
        map_gold_label(
            False,
            world_assumption=WorldAssumption.OPEN,
            question={"strategy": "inv-proof"},
        )


def test_invalid_label_is_rejected() -> None:
    with pytest.raises(DatasetRecordError, match="Unsupported"):
        map_gold_label(
            "maybe",
            world_assumption=WorldAssumption.OPEN,
            question={},
        )


def test_loader_refuses_ambiguous_cwa_record() -> None:
    loader = ProofWriterLoader(FIXTURE_ROOT)

    with pytest.raises(DatasetRecordError, match="CWA"):
        list(
            loader.iter_examples(
                variant="depth-1",
                split=Split.TRAIN,
                world_assumption=WorldAssumption.CLOSED,
            )
        )


def test_missing_required_record_field_has_file_and_line_context(tmp_path: Path) -> None:
    record = fixture_record()
    del record["theory"]
    source = write_source(tmp_path, [record])

    with pytest.raises(DatasetRecordError, match=r"meta-train\.jsonl:1"):
        list(ProofWriterLoader(source).iter_examples(variant="depth-1", split=Split.TRAIN))


def test_missing_question_field_is_counted_without_silent_success(tmp_path: Path) -> None:
    record = fixture_record()
    del record["questions"]["Q1"]["answer"]
    source = write_source(tmp_path, [record])
    stats = LoaderStats()

    examples = list(
        ProofWriterLoader(source).iter_examples(
            variant="depth-1", split=Split.TRAIN, strict=False, stats=stats
        )
    )

    assert len(examples) == 2
    assert stats.invalid_questions == 1
    assert stats.missing_fields["questions.answer"] == 1


def test_invalid_natlang_sentence_is_rejected_instead_of_discarded(
    tmp_path: Path,
) -> None:
    record = fixture_record()
    record["sentences"] = {"sentence1": ""}
    source = write_source(tmp_path, [record])

    with pytest.raises(DatasetRecordError, match=r"meta-train\.jsonl:1"):
        list(ProofWriterLoader(source).iter_examples(variant="depth-1", split=Split.TRAIN))


def test_duplicate_deterministic_id_is_rejected(tmp_path: Path) -> None:
    record = fixture_record()
    source = write_source(tmp_path, [record, record])

    with pytest.raises(DatasetRecordError) as error:
        list(ProofWriterLoader(source).iter_examples(variant="depth-1", split=Split.TRAIN))

    assert isinstance(error.value.__cause__, DuplicateExampleError)


def test_deterministic_id_is_stable_across_loader_instances() -> None:
    first = next(
        ProofWriterLoader(FIXTURE_ROOT).iter_examples(variant="depth-1", split=Split.TRAIN)
    )
    second = next(
        ProofWriterLoader(FIXTURE_ROOT).iter_examples(variant="depth-1", split=Split.TRAIN)
    )

    assert first.example_id == second.example_id
    assert first.example_id.endswith("Synthetic-OWA-Train-1/Q1")


def test_invalid_reasoning_depth_has_record_context(tmp_path: Path) -> None:
    record = fixture_record()
    record["questions"]["Q1"]["QDep"] = []
    source = write_source(tmp_path, [record])

    with pytest.raises(DatasetRecordError, match=r"meta-train\.jsonl:1:Q1"):
        list(ProofWriterLoader(source).iter_examples(variant="depth-1", split=Split.TRAIN))


def test_proof_and_structured_metadata_are_preserved() -> None:
    example = next(
        ProofWriterLoader(FIXTURE_ROOT).iter_examples(variant="depth-1", split=Split.TRAIN)
    )

    assert example.gold_proofs is not None
    assert example.gold_proofs["proofs"] == "[(triple1)]"
    assert example.structured_facts["triple1"].text == "Mira is calm."
    assert example.structured_rules["rule1"].representation is not None
    assert example.original_raw_label is True


def test_streaming_does_not_read_the_next_record_before_needed(tmp_path: Path) -> None:
    first = fixture_record()
    second = fixture_record()
    second["id"] = "Synthetic-OWA-Train-2"
    source = write_source(tmp_path, [first, second])
    stats = LoaderStats()
    iterator = ProofWriterLoader(source).iter_examples(
        variant="depth-1", split=Split.TRAIN, stats=stats
    )

    next(iterator)

    assert stats.records_seen == 1
    assert stats.examples_yielded == 1


def test_official_split_is_preserved() -> None:
    example = next(
        ProofWriterLoader(FIXTURE_ROOT).iter_examples(variant="depth-1", split=Split.DEVELOPMENT)
    )

    assert example.split is Split.DEVELOPMENT
    assert "/dev/" not in example.example_id
