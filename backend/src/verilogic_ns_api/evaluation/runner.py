from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from verilogic_ns_api import __version__
from verilogic_ns_api.evaluation.metrics import compute_metrics
from verilogic_ns_api.evaluation.protocol import FatalPredictorError, Predictor
from verilogic_ns_api.research.models import (
    BenchmarkExample,
    MetricReport,
    PredictionLabel,
    PredictionRecord,
    RunManifest,
    RunStatus,
    Split,
)

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


class ExistingRunError(RuntimeError):
    pass


class InvalidPredictionError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunResult:
    run_directory: Path
    manifest: RunManifest
    metrics: MetricReport
    predictions: list[PredictionRecord]


def generated_run_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def _git_state() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    commit = revision.stdout.strip().lower()
    validated_commit = commit if re.fullmatch(r"[a-f0-9]{40}", commit) else None
    return validated_commit, bool(status.stdout.strip())


def _write_json_atomic(path: Path, model: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    payload = model.model_dump(mode="json") if hasattr(model, "model_dump") else model
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _write_jsonl_atomic(path: Path, predictions: list[PredictionRecord]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for prediction in predictions:
                stream.write(prediction.model_dump_json())
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


class EvaluationRunner:
    def __init__(
        self,
        *,
        output_root: Path,
        dataset_manifest_reference: str,
        configuration: dict[str, Any],
        seed: int,
        selected_splits: list[Split],
    ) -> None:
        self.output_root = output_root.resolve()
        self.dataset_manifest_reference = dataset_manifest_reference
        self.configuration = configuration
        self.seed = seed
        self.selected_splits = selected_splits

    def run(
        self,
        predictor: Predictor,
        examples: Iterable[BenchmarkExample],
        *,
        run_id: str | None = None,
        run_id_prefix: str = "verilogic-run",
        max_workers: int = 1,
    ) -> RunResult:
        resolved_run_id = run_id or generated_run_id(run_id_prefix)
        if not SAFE_RUN_ID.fullmatch(resolved_run_id):
            raise ValueError("run_id contains unsafe path characters")
        if not 1 <= max_workers <= 16:
            raise ValueError("max_workers must be between 1 and 16")
        items = list(examples)
        example_ids = [example.example_id for example in items]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("Evaluation input contains duplicate example IDs")

        self.output_root.mkdir(parents=True, exist_ok=True)
        final_directory = (self.output_root / resolved_run_id).resolve()
        incomplete_directory = (self.output_root / f"{resolved_run_id}.incomplete").resolve()
        for path in (final_directory, incomplete_directory):
            if not path.is_relative_to(self.output_root):
                raise ValueError("Run path escapes output root")
            if path.exists():
                raise ExistingRunError(f"Run output already exists: {path}")
        incomplete_directory.mkdir()

        started_at = datetime.now(UTC)
        git_commit, git_dirty = _git_state()
        manifest = RunManifest(
            run_id=resolved_run_id,
            status=RunStatus.INCOMPLETE,
            dataset_manifest_reference=self.dataset_manifest_reference,
            selected_splits=self.selected_splits,
            configuration=self.configuration,
            seed=self.seed,
            predictor_name=predictor.name,
            predictor_version=predictor.version,
            git_commit=git_commit,
            git_dirty=git_dirty,
            started_at=started_at,
            environment={
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "verilogic_package_version": __version__,
            },
            example_count=len(items),
            success_count=0,
            abstention_count=0,
            error_count=0,
        )
        _write_json_atomic(incomplete_directory / "run-manifest.json", manifest)

        predictions: list[PredictionRecord] = []
        try:

            def predict_one(example: BenchmarkExample) -> PredictionRecord:
                start = perf_counter()
                try:
                    candidate = predictor.predict(example.for_prediction(), run_id=resolved_run_id)
                    payload = (
                        candidate.model_dump(mode="python")
                        if hasattr(candidate, "model_dump")
                        else candidate
                    )
                    prediction = PredictionRecord.model_validate(payload)
                    if prediction.run_id != resolved_run_id:
                        raise InvalidPredictionError("Predictor returned a mismatched run_id")
                    if prediction.example_id != example.example_id:
                        raise InvalidPredictionError("Predictor returned a mismatched example_id")
                    if prediction.predictor_name != predictor.name:
                        raise InvalidPredictionError("Predictor returned a mismatched name")
                    if prediction.predictor_version != predictor.version:
                        raise InvalidPredictionError("Predictor returned a mismatched version")
                    prediction = prediction.model_copy(
                        update={"latency_ms": (perf_counter() - start) * 1000}
                    )
                except FatalPredictorError:
                    raise
                except Exception as error:
                    invalid = isinstance(error, (ValidationError, InvalidPredictionError))
                    prediction = PredictionRecord(
                        run_id=resolved_run_id,
                        example_id=example.example_id,
                        predicted_label=PredictionLabel.ERROR,
                        error_type=("InvalidPredictionError" if invalid else type(error).__name__),
                        latency_ms=(perf_counter() - start) * 1000,
                        predictor_name=predictor.name,
                        predictor_version=predictor.version,
                        timestamp=datetime.now(UTC),
                    )
                return prediction

            if max_workers == 1:
                predictions = [predict_one(example) for example in items]
            else:
                with ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="verilogic-predict"
                ) as executor:
                    predictions = list(executor.map(predict_one, items))

            metrics = compute_metrics(items, predictions)
            _write_jsonl_atomic(incomplete_directory / "predictions.jsonl", predictions)
            _write_json_atomic(incomplete_directory / "metrics.json", metrics)
            completed_manifest = manifest.model_copy(
                update={
                    "status": RunStatus.COMPLETE,
                    "completed_at": datetime.now(UTC),
                    "success_count": metrics.answered_examples,
                    "abstention_count": metrics.abstained_examples,
                    "error_count": metrics.errored_examples,
                }
            )
            _write_json_atomic(incomplete_directory / "run-manifest.json", completed_manifest)
            os.replace(incomplete_directory, final_directory)
        except Exception:
            failed_manifest = manifest.model_copy(
                update={"status": RunStatus.FAILED, "completed_at": datetime.now(UTC)}
            )
            _write_json_atomic(incomplete_directory / "run-manifest.json", failed_manifest)
            raise

        return RunResult(
            run_directory=final_directory,
            manifest=completed_manifest,
            metrics=metrics,
            predictions=predictions,
        )
