from __future__ import annotations

from datetime import UTC, datetime

from verilogic_ns_api.research.models import (
    PredictionInput,
    PredictionLabel,
    PredictionRecord,
)


class ConstantUnknownPredictor:
    name = "constant-unknown"

    def __init__(self, version: str = "1.0") -> None:
        self.version = version

    def predict(self, example: PredictionInput, *, run_id: str) -> PredictionRecord:
        return PredictionRecord(
            run_id=run_id,
            example_id=example.example_id,
            predicted_label=PredictionLabel.UNKNOWN,
            latency_ms=0,
            predictor_name=self.name,
            predictor_version=self.version,
            timestamp=datetime.now(UTC),
        )


class MappingPredictor:
    """Explicit mapping predictor for automated tests only; never use for research runs."""

    name = "mapping-fixture-only"
    version = "test-only"

    def __init__(self, predictions: dict[str, PredictionLabel]) -> None:
        self._predictions = predictions.copy()

    def predict(self, example: PredictionInput, *, run_id: str) -> PredictionRecord:
        if example.example_id not in self._predictions:
            raise KeyError(f"No fixture prediction for {example.example_id}")
        return PredictionRecord(
            run_id=run_id,
            example_id=example.example_id,
            predicted_label=self._predictions[example.example_id],
            latency_ms=0,
            predictor_name=self.name,
            predictor_version=self.version,
            timestamp=datetime.now(UTC),
        )
