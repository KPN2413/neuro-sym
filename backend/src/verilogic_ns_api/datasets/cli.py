from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verilogic_ns_api.datasets.acquisition import (
    DEFAULT_MAX_DOWNLOAD_BYTES,
    DEFAULT_MAX_EXTRACTED_BYTES,
    PROOFWRITER_URL,
    PROOFWRITER_VERSION,
    acquire_proofwriter,
)
from verilogic_ns_api.datasets.errors import DatasetError
from verilogic_ns_api.datasets.inspection import inspect_proofwriter
from verilogic_ns_api.datasets.preparation import prepare_proofwriter
from verilogic_ns_api.research.models import Split


def _default_archive() -> Path:
    return (
        Path("datasets")
        / "proofwriter"
        / "raw"
        / "archives"
        / f"proofwriter-dataset-{PROOFWRITER_VERSION}.zip"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m verilogic_ns_api.datasets",
        description="Safe ProofWriter acquisition, inspection, and preparation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download a supported dataset safely")
    download.add_argument("dataset", choices=["proofwriter"])
    download.add_argument("--dataset-root", type=Path, default=Path("datasets/proofwriter"))
    download.add_argument("--url", default=PROOFWRITER_URL)
    download.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_DOWNLOAD_BYTES)
    download.add_argument("--expected-sha256")
    download.add_argument("--force", action="store_true")
    download.add_argument("--extract", action="store_true")
    download.add_argument("--max-extracted-bytes", type=int, default=DEFAULT_MAX_EXTRACTED_BYTES)

    inspect = subparsers.add_parser("inspect", help="Inspect dataset layout and aggregate metadata")
    inspect.add_argument("dataset", choices=["proofwriter"])
    inspect.add_argument("--data-source", type=Path, default=_default_archive())
    inspect.add_argument("--variant", action="append", dest="variants")
    inspect.add_argument("--split", action="append", choices=[split.value for split in Split])
    inspect.add_argument("--max-examples-per-split", type=int)

    prepare = subparsers.add_parser("prepare", help="Validate and normalize ProofWriter OWA data")
    prepare.add_argument("dataset", choices=["proofwriter"])
    prepare.add_argument("--data-source", type=Path, default=_default_archive())
    prepare.add_argument("--output-root", type=Path, default=Path("datasets/proofwriter/processed"))
    prepare.add_argument("--variant", required=True)
    prepare.add_argument(
        "--split",
        action="append",
        choices=[split.value for split in Split],
        default=None,
    )
    prepare.add_argument("--include-test", action="store_true")
    prepare.add_argument("--max-examples-per-split", type=int)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--manifest-reference")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            result = acquire_proofwriter(
                args.dataset_root,
                url=args.url,
                max_bytes=args.max_bytes,
                expected_sha256=args.expected_sha256,
                force=args.force,
                extract=args.extract,
                max_extracted_bytes=args.max_extracted_bytes,
            )
            payload = {
                "archive": str(result.download.archive_path),
                "size_bytes": result.download.size_bytes,
                "sha256": result.download.sha256,
                "checksum_status": (
                    "expected-and-matched" if args.expected_sha256 else "observed-only"
                ),
                "download_skipped": result.download.skipped,
                "manifest": str(result.manifest_path),
                "extraction": str(result.extraction_path) if result.extraction_path else None,
            }
        elif args.command == "inspect":
            splits = [Split(split) for split in args.split] if args.split else None
            payload = inspect_proofwriter(
                args.data_source,
                variants=args.variants,
                splits=splits,
                max_examples_per_split=args.max_examples_per_split,
            )
        else:
            splits = [Split(split) for split in (args.split or ["train", "dev"])]
            output = prepare_proofwriter(
                data_source=args.data_source,
                output_root=args.output_root,
                variant=args.variant,
                splits=splits,
                allow_test=args.include_test,
                max_examples_per_split=args.max_examples_per_split,
                force=args.force,
                dataset_manifest_reference=args.manifest_reference,
            )
            payload = {"prepared_directory": str(output), "splits": [s.value for s in splits]}
    except (DatasetError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
