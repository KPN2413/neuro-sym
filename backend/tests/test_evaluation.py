from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from verilogic_ns_api.datasets.proofwriter import ProofWriterLoader
from verilogic_ns_api.evaluation.metrics import compute_metrics
from verilogic_ns_api.evaluation.predictors import (
    ConstantUnknownPredictor,
    MappingPredictor,
)
from verilogic_ns_api.evaluation.runner import EvaluationRunner, ExistingRunError
from verilogic_ns_api.research.models import (
    GoldLabel,
    PredictionInput,
    PredictionLabel,
    PredictionRecord,
    RunStatus,
    Split,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "proofwriter" / "proofwriter-dataset-V2020.12.3"


def examples():
    loader = ProofWriterLoader(FIXTURE_ROOT)
    return [
        *loader.iter_examples(variant="depth-1", split=Split.TRAIN),
        *loader.iter_examples(variant="depth-1", split=Split.DEVELOPMENT),
    ]


def prediction(
    example_id: str,
    label: PredictionLabel,
    *,
    error_type: str | None = None,
    abstention_reason: str | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        run_id="test-run",
        example_id=example_id,
        predicted_label=label,
        error_type=error_type,
        abstention_reason=abstention_reason,
        latency_ms=1,
        predictor_name="test",
        predictor_version="1",
        timestamp=datetime.now(UTC),
    )


def runner(tmp_path: Path) -> EvaluationRunner:
    return EvaluationRunner(
        output_root=tmp_path / "runs",
        dataset_manifest_reference="datasets/proofwriter/manifest.example.json",
        configuration={"sampling": {"seed": 7}},
        seed=7,
        selected_splits=[Split.TRAIN, Split.DEVELOPMENT],
    )


def test_perfect_predictions() -> None:
    items = examples()
    predictions = [
        prediction(item.example_id, PredictionLabel(item.gold_label.value)) for item in items
    ]

    report = compute_metrics(items, predictions)

    assert report.accuracy == 1
    assert report.answered_only_accuracy == 1
    assert report.coverage == 1
    assert report.selective_risk == 0
    assert report.macro_precision == 1
    assert report.macro_recall == 1
    assert report.macro_f1 == 1


def test_completely_incorrect_predictions() -> None:
    items = examples()
    wrong = {
        GoldLabel.ENTAILED: PredictionLabel.CONTRADICTED,
        GoldLabel.CONTRADICTED: PredictionLabel.UNKNOWN,
        GoldLabel.UNKNOWN: PredictionLabel.ENTAILED,
    }

    report = compute_metrics(
        items,
        [prediction(item.example_id, wrong[item.gold_label]) for item in items],
    )

    assert report.accuracy == 0
    assert report.answered_only_accuracy == 0
    assert report.selective_risk == 1


def test_mixed_three_label_metrics_match_manual_values() -> None:
    items = examples()[:3]
    predictions = [
        prediction(items[0].example_id, PredictionLabel.ENTAILED),
        prediction(items[1].example_id, PredictionLabel.ENTAILED),
        prediction(items[2].example_id, PredictionLabel.CONTRADICTED),
    ]

    report = compute_metrics(items, predictions)

    assert report.accuracy == pytest.approx(1 / 3)
    assert report.macro_precision == pytest.approx(1 / 6)
    assert report.macro_recall == pytest.approx(1 / 3)
    assert report.macro_f1 == pytest.approx(2 / 9)
    assert report.confusion_matrix["CONTRADICTED"]["ENTAILED"] == 1


def test_abstention_reduces_coverage_but_not_answered_only_accuracy() -> None:
    items = examples()[:3]
    predictions = [
        prediction(items[0].example_id, PredictionLabel.ABSTAIN, abstention_reason="low"),
        prediction(items[1].example_id, PredictionLabel.CONTRADICTED),
        prediction(items[2].example_id, PredictionLabel.UNKNOWN),
    ]

    report = compute_metrics(items, predictions)

    assert report.abstained_examples == 1
    assert report.coverage == pytest.approx(2 / 3)
    assert report.accuracy == pytest.approx(2 / 3)
    assert report.answered_only_accuracy == 1
    assert report.selective_risk == 0
    assert report.confusion_matrix["ENTAILED"]["ABSTAIN"] == 1


def test_error_is_counted_separately() -> None:
    items = examples()[:1]
    report = compute_metrics(
        items,
        [prediction(items[0].example_id, PredictionLabel.ERROR, error_type="Failure")],
    )

    assert report.errored_examples == 1
    assert report.answered_examples == 0
    assert report.coverage == 0
    assert report.answered_only_accuracy is None
    assert report.selective_risk is None


def test_per_depth_metrics_are_reported() -> None:
    items = examples()
    predictions = [
        prediction(item.example_id, PredictionLabel(item.gold_label.value)) for item in items
    ]

    report = compute_metrics(items, predictions)

    assert report.per_depth_metrics["0"].total == 4
    assert report.per_depth_metrics["1"].total == 2
    assert report.per_depth_metrics["1"].accuracy == 1


def test_constant_unknown_predictor_is_a_valid_trivial_baseline() -> None:
    item = examples()[0].for_prediction()

    record = ConstantUnknownPredictor().predict(item, run_id="test-run")

    assert record.predicted_label is PredictionLabel.UNKNOWN
    assert record.example_id == item.example_id


def test_mapping_predictor_uses_only_explicit_test_mapping() -> None:
    item = examples()[0].for_prediction()
    predictor = MappingPredictor({item.example_id: PredictionLabel.ENTAILED})

    assert predictor.predict(item, run_id="test-run").predicted_label is PredictionLabel.ENTAILED


def test_runner_writes_complete_atomic_outputs(tmp_path: Path) -> None:
    result = runner(tmp_path).run(ConstantUnknownPredictor(), examples(), run_id="atomic-run")

    assert result.manifest.status is RunStatus.COMPLETE
    assert result.run_directory.name == "atomic-run"
    assert (result.run_directory / "predictions.jsonl").is_file()
    assert (result.run_directory / "metrics.json").is_file()
    assert (result.run_directory / "run-manifest.json").is_file()
    assert result.manifest.git_dirty is not None
    assert not list(result.run_directory.glob("*.tmp"))
    lines = (result.run_directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert "gold_label" not in json.loads(lines[0])


def test_runner_protects_existing_complete_run(tmp_path: Path) -> None:
    evaluator = runner(tmp_path)
    evaluator.run(ConstantUnknownPredictor(), examples(), run_id="same-run")

    with pytest.raises(ExistingRunError, match="already exists"):
        evaluator.run(ConstantUnknownPredictor(), examples(), run_id="same-run")


def test_runner_protects_existing_incomplete_run(tmp_path: Path) -> None:
    evaluator = runner(tmp_path)
    incomplete = tmp_path / "runs" / "pending.incomplete"
    incomplete.mkdir(parents=True)

    with pytest.raises(ExistingRunError, match="already exists"):
        evaluator.run(ConstantUnknownPredictor(), examples(), run_id="pending")


class FailingPredictor:
    name = "failing"
    version = "1"

    def predict(self, example: PredictionInput, *, run_id: str) -> PredictionRecord:
        raise RuntimeError("synthetic predictor failure")


def test_predictor_errors_are_recorded_and_run_continues(tmp_path: Path) -> None:
    result = runner(tmp_path).run(FailingPredictor(), examples(), run_id="error-run")

    assert result.manifest.status is RunStatus.COMPLETE
    assert result.manifest.error_count == 6
    assert result.metrics.errored_examples == 6
    assert all(record.predicted_label is PredictionLabel.ERROR for record in result.predictions)


class InvalidLabelPredictor:
    name = "invalid"
    version = "1"

    def predict(self, example: PredictionInput, *, run_id: str):
        return {
            "run_id": run_id,
            "example_id": example.example_id,
            "predicted_label": "NOT_A_LABEL",
            "latency_ms": 0,
            "predictor_name": self.name,
            "predictor_version": self.version,
            "timestamp": datetime.now(UTC),
        }


def test_invalid_prediction_label_becomes_typed_error(tmp_path: Path) -> None:
    result = runner(tmp_path).run(InvalidLabelPredictor(), examples(), run_id="invalid-run")

    assert result.metrics.invalid_prediction_count == 6
    assert {record.error_type for record in result.predictions} == {"InvalidPredictionError"}


class MismatchedVersionPredictor(ConstantUnknownPredictor):
    def predict(self, example: PredictionInput, *, run_id: str) -> PredictionRecord:
        return (
            super()
            .predict(example, run_id=run_id)
            .model_copy(update={"predictor_version": "wrong-version"})
        )


def test_mismatched_predictor_version_becomes_typed_error(tmp_path: Path) -> None:
    result = runner(tmp_path).run(
        MismatchedVersionPredictor(), examples(), run_id="version-mismatch"
    )

    assert result.metrics.invalid_prediction_count == 6


class GoldLeakageProbe:
    name = "gold-probe"
    version = "1"

    def predict(self, example: PredictionInput, *, run_id: str) -> PredictionRecord:
        assert "gold_label" not in type(example).model_fields
        assert not hasattr(example, "original_raw_label")
        return (
            ConstantUnknownPredictor()
            .predict(example, run_id=run_id)
            .model_copy(update={"predictor_name": self.name, "predictor_version": self.version})
        )


def test_real_predictor_interface_has_no_gold_label_access(tmp_path: Path) -> None:
    result = runner(tmp_path).run(GoldLeakageProbe(), examples(), run_id="gold-safe")

    assert result.manifest.error_count == 0


def test_internal_metric_failure_leaves_failed_incomplete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_metrics(*args, **kwargs):
        raise RuntimeError("synthetic metric failure")

    monkeypatch.setattr("verilogic_ns_api.evaluation.runner.compute_metrics", fail_metrics)

    with pytest.raises(RuntimeError, match="metric failure"):
        runner(tmp_path).run(ConstantUnknownPredictor(), examples(), run_id="failed-run")

    incomplete = tmp_path / "runs" / "failed-run.incomplete"
    manifest = json.loads((incomplete / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not (tmp_path / "runs" / "failed-run").exists()
