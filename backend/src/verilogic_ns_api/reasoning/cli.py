from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from verilogic_ns_api.reasoning.configuration import (
    FormalRepresentationError,
    ProofVerificationError,
    ReasoningError,
    ReasoningLimits,
    ResourceLimitError,
)
from verilogic_ns_api.reasoning.engine import ForwardChainingEngine
from verilogic_ns_api.reasoning.models import (
    ProofDAG,
    ReasoningOutput,
    ReasoningResult,
    Theory,
)
from verilogic_ns_api.reasoning.proofwriter import (
    run_conformance,
    select_conformance_examples,
)
from verilogic_ns_api.reasoning.verifier import ProofVerifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m verilogic_ns_api.reasoning",
        description="Validate, saturate, reason over, and verify typed VeriLogic-NS theories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reason = subparsers.add_parser("reason", help="Classify a theory query and emit its proof")
    _add_theory_input(reason)
    _add_output(reason)
    _add_limits(reason)
    reason.add_argument("--human", action="store_true", help="Print a concise human summary")

    saturate = subparsers.add_parser("saturate", help="Compute the complete signed closure")
    _add_theory_input(saturate)
    _add_output(saturate)
    _add_limits(saturate)

    inspect = subparsers.add_parser(
        "inspect-closure", help="Summarize closure size, depth, and conflicts"
    )
    _add_theory_input(inspect)
    _add_output(inspect)
    _add_limits(inspect)
    inspect.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    verify = subparsers.add_parser(
        "verify-proof", help="Independently replay and verify a reasoning proof"
    )
    verify.add_argument("--theory", required=True, type=Path)
    verify.add_argument("--proof", required=True, type=Path)
    _add_output(verify)
    _add_limits(verify)

    conformance = subparsers.add_parser(
        "conformance-run",
        help="Run the formal ProofWriter oracle-structure conformance evaluation",
    )
    conformance.add_argument("--data-source", required=True, type=Path)
    conformance.add_argument("--variant", default="depth-5")
    conformance.add_argument("--per-cell", type=int, default=20)
    conformance.add_argument("--seed", type=int, default=20260713)
    conformance.add_argument(
        "--selection-manifest",
        type=Path,
        help="Use the exact example IDs in a Phase 3 selection manifest",
    )
    _add_output(conformance)
    _add_limits(conformance)
    return parser


def _add_theory_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path, help="Typed theory.v1 JSON file")


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, help="Output path beneath the current directory")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")


def _add_limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-derived-literals", type=int, default=100_000)
    parser.add_argument("--max-rule-firings", type=int, default=1_000_000)
    parser.add_argument("--max-rounds", type=int, default=1_000)
    parser.add_argument("--max-proof-nodes", type=int, default=100_000)
    parser.add_argument("--timeout-seconds", type=float)


def _limits(args: argparse.Namespace) -> ReasoningLimits:
    return ReasoningLimits(
        max_derived_literals=args.max_derived_literals,
        max_rule_firings=args.max_rule_firings,
        max_rounds=args.max_rounds,
        max_proof_nodes=args.max_proof_nodes,
        timeout_seconds=args.timeout_seconds,
    )


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError("Input JSON file does not exist")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Input JSON exceeds the 50 MiB safety limit")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_theory(path: Path) -> Theory:
    return Theory.model_validate(_load_json(path))


def _safe_output_path(path: Path) -> Path:
    root = Path.cwd().resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Output path must remain beneath the current working directory")
    return resolved


def _write_output(payload: Any, output: Path | None, *, force: bool) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    target = _safe_output_path(output)
    if target.exists() and not force:
        raise ValueError("Output already exists; pass --force to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _execute(args: argparse.Namespace) -> None:
    limits = _limits(args)
    if args.command == "reason":
        theory = _load_theory(args.input)
        outcome = ForwardChainingEngine(limits).reason(theory)
        if args.human and args.output is None:
            result = outcome.result
            print(f"Status: {result.status.value}")
            print(f"Conflicts in closure: {result.conflict_count}")
            print(f"Proof nodes: {len(result.proof.nodes)}")
            print(f"Proof hash: {result.proof.proof_hash}")
            return
        _write_output(outcome.model_dump(mode="json"), args.output, force=args.force)
        return

    if args.command == "saturate":
        theory = _load_theory(args.input)
        output = ForwardChainingEngine(limits).saturate(theory)
        _write_output(output.model_dump(mode="json"), args.output, force=args.force)
        return

    if args.command == "inspect-closure":
        theory = _load_theory(args.input)
        output = ForwardChainingEngine(limits).saturate(theory)
        summary = {
            "theory_id": output.theory_id,
            "closure_size": len(output.closure),
            "conflict_count": len(output.conflicts),
            "maximum_depth": output.telemetry.maximum_proof_depth,
            "rounds": output.telemetry.rounds,
        }
        if not args.json and args.output is None:
            for key, value in summary.items():
                print(f"{key}: {value}")
            return
        _write_output(summary, args.output, force=args.force)
        return

    if args.command == "verify-proof":
        theory = _load_theory(args.theory)
        payload = _load_json(args.proof)
        verifier = ProofVerifier(limits)
        if isinstance(payload, dict) and "result" in payload:
            reasoning_output = ReasoningOutput.model_validate(payload)
            verified = verifier.verify_result(theory, reasoning_output.result)
        elif isinstance(payload, dict) and "proof" in payload:
            result = ReasoningResult.model_validate(payload)
            verified = verifier.verify_result(theory, result)
        else:
            proof = ProofDAG.model_validate(payload)
            verified = verifier.verify_proof(theory, proof)
        _write_output(verified.model_dump(mode="json"), args.output, force=args.force)
        return

    if args.command == "conformance-run":
        ids: set[str] | None = None
        if args.selection_manifest is not None:
            manifest = _load_json(args.selection_manifest)
            if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
                raise ValueError("Selection manifest must contain an entries array")
            ids = {
                item["example_id"]
                for item in manifest["entries"]
                if isinstance(item, dict) and isinstance(item.get("example_id"), str)
            }
            if len(ids) != len(manifest["entries"]):
                raise ValueError("Selection manifest contains invalid or duplicate example IDs")
        examples = select_conformance_examples(
            args.data_source,
            variant=args.variant,
            per_cell=args.per_cell,
            seed=args.seed,
            example_ids=ids,
        )
        report = run_conformance(
            examples,
            engine=ForwardChainingEngine(limits),
            verifier=ProofVerifier(limits),
        )
        _write_output(report, args.output, force=args.force)
        if report["mismatch_count"]:
            raise FormalRepresentationError(
                f"Conformance produced {report['mismatch_count']} classification mismatches"
            )
        return
    raise ValueError(f"Unsupported command {args.command!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _execute(args)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 2
    except ResourceLimitError as error:
        print(f"RESOURCE_LIMIT: {error}", file=sys.stderr)
        return 3
    except ProofVerificationError as error:
        print(f"PROOF_INVALID: {error}", file=sys.stderr)
        return 4
    except (FormalRepresentationError, ReasoningError) as error:
        print(f"REASONING_ERROR: {error}", file=sys.stderr)
        return 5
    return 0
