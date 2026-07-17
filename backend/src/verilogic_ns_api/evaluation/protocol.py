from __future__ import annotations

from typing import Protocol, runtime_checkable

from verilogic_ns_api.research.models import PredictionInput, PredictionRecord


class FatalPredictorError(RuntimeError):
    """Abort a run immediately for authentication or systemic configuration failures."""


@runtime_checkable
class Predictor(Protocol):
    """Model-independent predictor contract with no access to gold labels."""

    name: str
    version: str

    def predict(self, example: PredictionInput, *, run_id: str) -> PredictionRecord:
        """Return one typed prediction for a gold-redacted benchmark input."""
        ...
