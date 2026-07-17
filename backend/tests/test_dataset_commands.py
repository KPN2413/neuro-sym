import json
from pathlib import Path

import pytest

from verilogic_ns_api.datasets.errors import ExistingDataError, SamplingError
from verilogic_ns_api.datasets.inspection import inspect_proofwriter
from verilogic_ns_api.datasets.preparation import prepare_proofwriter
from verilogic_ns_api.evaluation.cli import run_evaluation
from verilogic_ns_api.research.models import EvaluationConfig, Split

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "proofwriter" / "proofwriter-dataset-V2020.12.3"


def test_synthetic_inspection_reports_aggregates_without_examples() -> None:
    report = inspect_proofwriter(FIXTURE_ROOT, variants=["depth-1"])

    assert report["example_count"] == 9
    assert report["label_distribution"] == {
        "CONTRADICTED": 3,
        "ENTAILED": 3,
        "UNKNOWN": 3,
    }
    assert not report["copyrighted_examples_included"]
    assert "examples" not in report
    assert report["source_provenance"] == {
        "source_type": "directory",
        "sha256": None,
        "checksum_status": "not-applicable",
        "publisher_verified_checksum": False,
    }


def test_preparation_writes_validated_jsonl_and_manifest(tmp_path: Path) -> None:
    output = prepare_proofwriter(
        data_source=FIXTURE_ROOT,
        output_root=tmp_path / "processed",
        variant="depth-1",
        splits=[Split.TRAIN, Split.DEVELOPMENT],
        max_examples_per_split=2,
        dataset_manifest_reference="manifest.json",
    )

    assert len((output / "normalized-train.jsonl").read_text().splitlines()) == 2
    assert len((output / "normalized-dev.jsonl").read_text().splitlines()) == 2
    manifest = json.loads((output / "preparation-manifest.json").read_text())
    assert manifest["partial"] is True
    assert manifest["example_counts"] == {"dev": 2, "train": 2}


def test_preparation_test_split_guard(tmp_path: Path) -> None:
    with pytest.raises(SamplingError, match="allow_test"):
        prepare_proofwriter(
            data_source=FIXTURE_ROOT,
            output_root=tmp_path,
            variant="depth-1",
            splits=[Split.TEST],
        )


def test_preparation_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    arguments = {
        "data_source": FIXTURE_ROOT,
        "output_root": tmp_path,
        "variant": "depth-1",
        "splits": [Split.TRAIN],
    }
    prepare_proofwriter(**arguments)

    with pytest.raises(ExistingDataError, match="already exists"):
        prepare_proofwriter(**arguments)


def test_synthetic_smoke_evaluation_configuration_runs(tmp_path: Path) -> None:
    config = EvaluationConfig.model_validate(
        {
            "schema_version": "1.0",
            "dataset": {
                "data_source": str(FIXTURE_ROOT),
                "variant": "depth-1",
                "splits": ["train", "dev"],
                "manifest_reference": "datasets/proofwriter/manifest.example.json",
            },
            "sampling": {
                "seed": 17,
                "max_examples": 6,
                "strategy": "balanced",
            },
            "predictor": {"kind": "constant_unknown", "version": "1.0"},
            "run": {
                "output_directory": str(tmp_path / "runs"),
                "run_id": "synthetic-smoke",
            },
        }
    )

    output = run_evaluation(config)

    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["total_examples"] == 6
    assert metrics["answered_examples"] == 6
    assert metrics["coverage"] == 1


def test_evaluation_configuration_refuses_test_split_before_loading(tmp_path: Path) -> None:
    config = EvaluationConfig.model_validate(
        {
            "dataset": {
                "data_source": str(FIXTURE_ROOT),
                "variant": "depth-1",
                "splits": ["test"],
                "manifest_reference": "manifest.json",
            },
            "sampling": {"allowed_splits": ["test"], "allow_test": False},
            "run": {"output_directory": str(tmp_path)},
        }
    )

    with pytest.raises(SamplingError, match="allow_test"):
        run_evaluation(config)
