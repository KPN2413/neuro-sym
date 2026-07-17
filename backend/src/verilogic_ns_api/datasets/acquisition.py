from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import httpx

from verilogic_ns_api.datasets.errors import (
    ChecksumMismatchError,
    DatasetAcquisitionError,
    DownloadSizeError,
    DownloadTimeoutError,
    ExistingDataError,
    InvalidArchiveError,
)

PROOFWRITER_VERSION = "V2020.12.3"
PROOFWRITER_URL = (
    "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
)
DEFAULT_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_ENTRIES = 10_000


@dataclass(frozen=True)
class DownloadResult:
    archive_path: Path
    size_bytes: int
    sha256: str
    observed_url: str
    observed_at: datetime
    response_headers: dict[str, str]
    skipped: bool


@dataclass(frozen=True)
class AcquisitionResult:
    download: DownloadResult
    manifest_path: Path
    extraction_path: Path | None


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip_archive(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            bad_entry = archive.testzip()
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise InvalidArchiveError(f"Invalid ZIP archive {path.name}: {error}") from error
    if bad_entry is not None:
        raise InvalidArchiveError(f"ZIP integrity check failed at entry: {bad_entry}")


def _validate_expected_checksum(observed: str, expected: str | None) -> None:
    if expected is None:
        return
    normalized = expected.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ChecksumMismatchError(
            "Expected SHA-256 must contain exactly 64 hexadecimal characters"
        )
    if observed != normalized:
        raise ChecksumMismatchError(f"SHA-256 mismatch: expected {normalized}, observed {observed}")


def download_archive(
    *,
    url: str,
    archive_path: Path,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    expected_sha256: str | None = None,
    force: bool = False,
    connect_timeout_seconds: float = 15.0,
    read_timeout_seconds: float = 60.0,
    chunk_size: int = 1024 * 1024,
    transport: httpx.BaseTransport | None = None,
) -> DownloadResult:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_suffix(f"{archive_path.suffix}.part")

    if archive_path.exists() and not force:
        try:
            validate_zip_archive(archive_path)
            observed_sha256 = sha256_file(archive_path)
            _validate_expected_checksum(observed_sha256, expected_sha256)
        except (InvalidArchiveError, ChecksumMismatchError) as error:
            raise ExistingDataError(
                f"Existing archive is not valid and will not be replaced without --force: {error}"
            ) from error
        size = archive_path.stat().st_size
        if size > max_bytes:
            raise DownloadSizeError(
                f"Existing archive is {size} bytes, above the configured limit of {max_bytes}"
            )
        return DownloadResult(
            archive_path=archive_path,
            size_bytes=size,
            sha256=observed_sha256,
            observed_url=url,
            observed_at=datetime.now(UTC),
            response_headers={},
            skipped=True,
        )

    with suppress(FileNotFoundError):
        partial_path.unlink()

    digest = hashlib.sha256()
    bytes_written = 0
    response_headers: dict[str, str] = {}
    observed_url = url
    timeout = httpx.Timeout(
        connect=connect_timeout_seconds,
        read=read_timeout_seconds,
        write=read_timeout_seconds,
        pool=connect_timeout_seconds,
    )

    try:
        with (
            httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                transport=transport,
            ) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            observed_url = str(response.url)
            response_headers = {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower() in {"content-length", "content-type", "etag", "last-modified"}
            }
            declared_length = response.headers.get("content-length")
            if declared_length is not None:
                try:
                    declared_size = int(declared_length)
                except ValueError as error:
                    raise DatasetAcquisitionError(
                        f"Invalid Content-Length header: {declared_length!r}"
                    ) from error
                if declared_size > max_bytes:
                    raise DownloadSizeError(
                        f"Server declared {declared_size} bytes, above limit {max_bytes}"
                    )
            with partial_path.open("wb") as destination:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise DownloadSizeError(
                            f"Download exceeded configured limit of {max_bytes} bytes"
                        )
                    digest.update(chunk)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
    except httpx.TimeoutException as error:
        with suppress(FileNotFoundError):
            partial_path.unlink()
        raise DownloadTimeoutError(f"Timed out while downloading {url}") from error
    except httpx.HTTPStatusError as error:
        with suppress(FileNotFoundError):
            partial_path.unlink()
        raise DatasetAcquisitionError(
            f"Dataset server returned HTTP {error.response.status_code} for {url}"
        ) from error
    except httpx.HTTPError as error:
        with suppress(FileNotFoundError):
            partial_path.unlink()
        raise DatasetAcquisitionError(
            f"Network failure while downloading {url}: {error}"
        ) from error
    except (OSError, DatasetAcquisitionError):
        with suppress(FileNotFoundError):
            partial_path.unlink()
        raise

    observed_sha256 = digest.hexdigest()
    try:
        _validate_expected_checksum(observed_sha256, expected_sha256)
        validate_zip_archive(partial_path)
        os.replace(partial_path, archive_path)
    except Exception:
        with suppress(FileNotFoundError):
            partial_path.unlink()
        raise

    return DownloadResult(
        archive_path=archive_path,
        size_bytes=bytes_written,
        sha256=observed_sha256,
        observed_url=observed_url,
        observed_at=datetime.now(UTC),
        response_headers=response_headers,
        skipped=False,
    )


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    if not name or "\x00" in name or "\\" in name:
        raise InvalidArchiveError(f"Unsafe ZIP entry name: {name!r}")
    member = PurePosixPath(name)
    if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
        raise InvalidArchiveError(f"ZIP path traversal rejected: {name!r}")
    if any(":" in part for part in member.parts):
        raise InvalidArchiveError(f"Drive-like ZIP path rejected: {name!r}")
    file_type = (info.external_attr >> 16) & 0o170000
    if file_type == stat.S_IFLNK:
        raise InvalidArchiveError(f"Symbolic-link ZIP entry rejected: {name!r}")
    return member


def safe_extract_zip(
    archive_path: Path,
    extraction_parent: Path,
    *,
    archive_sha256: str | None = None,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
    max_entries: int = DEFAULT_MAX_ARCHIVE_ENTRIES,
) -> Path:
    archive_sha256 = archive_sha256 or sha256_file(archive_path)
    extraction_parent.mkdir(parents=True, exist_ok=True)
    final_directory = extraction_parent / archive_sha256
    if final_directory.exists():
        return final_directory

    temporary_directory = extraction_parent / f".{archive_sha256}.extracting-{uuid4().hex}"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > max_entries:
                raise InvalidArchiveError(
                    f"Archive has {len(infos)} entries, above limit {max_entries}"
                )
            total_size = sum(info.file_size for info in infos)
            if total_size > max_extracted_bytes:
                raise InvalidArchiveError(
                    f"Archive expands to {total_size} bytes, above limit {max_extracted_bytes}"
                )
            members = [(info, _safe_member_path(info)) for info in infos]

            temporary_directory.mkdir(parents=False, exist_ok=False)
            resolved_root = temporary_directory.resolve()
            for info, member in members:
                destination = temporary_directory.joinpath(*member.parts)
                resolved_destination = destination.resolve()
                if not resolved_destination.is_relative_to(resolved_root):
                    raise InvalidArchiveError(
                        f"ZIP entry escapes extraction root: {info.filename!r}"
                    )
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        os.replace(temporary_directory, final_directory)
    except (OSError, zipfile.BadZipFile, InvalidArchiveError) as error:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        if isinstance(error, InvalidArchiveError):
            raise
        raise InvalidArchiveError(f"Failed to extract {archive_path.name}: {error}") from error
    return final_directory


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        with suppress(FileNotFoundError):
            temporary_path.unlink()
        raise


def acquire_proofwriter(
    dataset_root: Path,
    *,
    url: str = PROOFWRITER_URL,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    expected_sha256: str | None = None,
    force: bool = False,
    extract: bool = False,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
    transport: httpx.BaseTransport | None = None,
) -> AcquisitionResult:
    archive_path = (
        dataset_root / "raw" / "archives" / f"proofwriter-dataset-{PROOFWRITER_VERSION}.zip"
    )
    download = download_archive(
        url=url,
        archive_path=archive_path,
        max_bytes=max_bytes,
        expected_sha256=expected_sha256,
        force=force,
        transport=transport,
    )
    extraction_path = None
    if extract:
        extraction_path = safe_extract_zip(
            archive_path,
            dataset_root / "raw" / "extracted",
            archive_sha256=download.sha256,
            max_extracted_bytes=max_extracted_bytes,
        )

    checksum_status = "expected-and-matched" if expected_sha256 else "observed-only"
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "ProofWriter",
        "dataset_version": PROOFWRITER_VERSION,
        "source_url": url,
        "observed_url": download.observed_url,
        "observed_at": download.observed_at.isoformat(),
        "archive": {
            "relative_path": archive_path.relative_to(dataset_root).as_posix(),
            "size_bytes": download.size_bytes,
            "sha256": download.sha256,
            "checksum_status": checksum_status,
            "publisher_verified_checksum": False,
        },
        "http_metadata": download.response_headers,
        "download_skipped": download.skipped,
        "extraction_relative_path": (
            extraction_path.relative_to(dataset_root).as_posix() if extraction_path else None
        ),
        "license_status": "not-stated-in-archive",
    }
    manifest_path = dataset_root / "raw" / "provenance.json"
    _write_json_atomic(manifest_path, manifest)
    return AcquisitionResult(
        download=download,
        manifest_path=manifest_path,
        extraction_path=extraction_path,
    )
