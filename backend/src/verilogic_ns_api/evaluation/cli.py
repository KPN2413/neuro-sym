from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path

import yaml
from pydantic import ValidationError

from verilogic_ns_api.datasets.errors import DatasetError, SamplingError
from verilogic_ns_api.datasets.proofwriter import ProofWriterLoader
from verilogic_ns_api.datasets.sampling import sample_examples
from verilogic_ns_api.evaluation.predictors import ConstantUnknownPredictor
from verilogic_ns_api.evaluation.runner import EvaluationRunner, ExistingRunError
from verilogic_ns_api.research.models import EvaluationConfig, Split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m verilogic_ns_api.evaluation",
        description="Run model-independent VeriLogic-NS evaluations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run an evaluation from a YAML configuration")
    run.add_argument("--config", type=Path, required=True)
    return parser


def _load_config(path: Path) -> EvaluationConfig:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Evaluation configuration must be a YAML mapping")
    return EvaluationConfig.model_validate(payload)


def run_evaluation(config: EvaluationConfig) -> Path:
    if Split.TEST in config.dataset.splits and not config.sampling.allow_test:
        raise SamplingError("Loading test data requires sampling.allow_test=true")
    loader = ProofWriterLoader(
        Path(config.dataset.data_source),
        dataset_version=config.dataset.version,
        dataset_manifest_reference=config.dataset.manifest_reference,
    )
    examples = chain.from_iterable(
        loader.iter_examples(variant=config.dataset.variant, split=split, strict=True)
        for split in config.dataset.splits
    )
    selected = sample_examples(examples, config.sampling)
    predictor = ConstantUnknownPredictor(version=config.predictor.version)
    runner = EvaluationRunner(
        output_root=Path(config.run.output_directory),
        dataset_manifest_reference=config.dataset.manifest_reference,
        configuration=config.model_dump(mode="json"),
        seed=config.sampling.seed,
        selected_splits=sorted(
            {example.split for example in selected}, key=lambda item: item.value
        ),
    )
    result = runner.run(
        predictor,
        selected,
        run_id=config.run.run_id,
        run_id_prefix=config.run.run_id_prefix,
    )
    return result.run_directory


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args.config)
        output = run_evaluation(config)
    except (OSError, ValueError, ValidationError, DatasetError, ExistingRunError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"run_directory": str(output)}, indent=2))
    return 0
