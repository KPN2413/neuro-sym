from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from verilogic_ns_api.baselines.configuration import resolve_repository_path
from verilogic_ns_api.research.models import Split
from verilogic_ns_api.semantic_parsing.cache import ParserResponseCache
from verilogic_ns_api.semantic_parsing.configuration import prepare_parser_experiment
from verilogic_ns_api.semantic_parsing.evaluation import run_parser_evaluation
from verilogic_ns_api.semantic_parsing.models import (
    CandidateQueryOutput,
    CandidateTheoryOutput,
    QueryParseInput,
    TheoryParseInput,
)
from verilogic_ns_api.semantic_parsing.prompts import PromptRegistry
from verilogic_ns_api.semantic_parsing.provider import OllamaStructuredProvider
from verilogic_ns_api.semantic_parsing.service import SemanticParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local gold-isolated neural semantic parser")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "replay"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name != "plan":
            command.add_argument("--dataset", choices=("calibration", "pilot"), default="pilot")
            command.add_argument("--run-id")
    for name in ("parse-theory", "parse-query"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--cache-only", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--kind", choices=("theory", "query"), required=True)
    validate.add_argument("--input", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            model = CandidateTheoryOutput if args.kind == "theory" else CandidateQueryOutput
            candidate = model.model_validate_json(args.input.read_text(encoding="utf-8"))
            print(json.dumps(candidate.model_dump(mode="json"), indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate":
            report = args.run / "metrics.json"
            if not report.is_file():
                raise ValueError("completed metrics.json is unavailable")
            print(report.read_text(encoding="utf-8"))
            return 0
        prepared = prepare_parser_experiment(args.config)
        if args.command == "plan":
            theories = {
                (
                    item.theory_id or item.example_id,
                    tuple(source.text for source in item.source_statements),
                )
                for item in prepared.pilot_examples
            }
            report = {
                "provider": "ollama-local-only",
                "model": prepared.config.runtime.model,
                "model_digest": prepared.config.runtime.model_digest,
                "pilot_examples": len(prepared.pilot_examples),
                "pilot_unique_theories": len(theories),
                "planned_pilot_requests": len(theories) + len(prepared.pilot_examples),
                "calibration_examples": len(prepared.calibration_examples),
                "test_split": False,
                "gold_fields_rendered": False,
                "hosted_calls": 0,
                "api_cost_usd": 0,
                "theory_prompt_hash": prepared.config.theory_prompt_sha256,
                "query_prompt_hash": prepared.config.query_prompt_sha256,
                "pilot_manifest_hash": prepared.config.pilot_manifest_sha256,
                "calibration_manifest_hash": prepared.config.calibration_manifest_sha256,
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        replay = args.command == "replay" or getattr(args, "cache_only", False)
        provider = None if replay else OllamaStructuredProvider(prepared.config.runtime)
        service = _service(prepared, provider=provider, replay_only=replay)
        try:
            if args.command in {"parse-theory", "parse-query"}:
                input_model = (
                    TheoryParseInput if args.command == "parse-theory" else QueryParseInput
                )
                value = input_model.model_validate_json(args.input.read_text(encoding="utf-8"))
                result = (
                    service.parse_theory(value)
                    if args.command == "parse-theory"
                    else service.parse_query(value)
                )
                print(json.dumps(result.outcome.model_dump(mode="json"), indent=2, sort_keys=True))
                return 0 if result.candidate is not None else 2
            examples = (
                prepared.calibration_examples
                if args.dataset == "calibration"
                else prepared.pilot_examples
            )
            split = Split.TRAIN if args.dataset == "calibration" else Split.DEVELOPMENT
            run_id = args.run_id or (
                f"semantic-parser-{args.dataset}-{'replay' if replay else 'live'}-"
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            )
            output_root = resolve_repository_path(prepared.root, prepared.config.output_directory)
            report = run_parser_evaluation(
                examples=examples,
                data_source=resolve_repository_path(prepared.root, prepared.config.data_source),
                variant=prepared.config.variant,
                split=split,
                parser=service,
                output_directory=output_root / run_id,
                run_id=run_id,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        finally:
            if provider is not None:
                provider.close()
    except (OSError, ValueError, ValidationError) as error:
        print(f"semantic-parser error: {error}", file=sys.stderr)
        return 2


def _service(prepared, *, provider, replay_only: bool) -> SemanticParser:
    registry = PromptRegistry(prepared.root)
    theory_prompt, theory_hash = registry.load(prepared.config.theory_prompt)
    query_prompt, query_hash = registry.load(prepared.config.query_prompt)
    cache_path = resolve_repository_path(prepared.root, prepared.config.cache_directory)
    return SemanticParser(
        config=prepared.config.runtime,
        theory_prompt=theory_prompt,
        theory_prompt_hash=theory_hash,
        query_prompt=query_prompt,
        query_prompt_hash=query_hash,
        cache=ParserResponseCache(cache_path),
        provider=provider,
        replay_only=replay_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
