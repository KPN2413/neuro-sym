from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import Path

import httpx
import pytest

from verilogic_ns_api.datasets.acquisition import (
    download_archive,
    safe_extract_zip,
)
from verilogic_ns_api.datasets.errors import (
    ChecksumMismatchError,
    DatasetAcquisitionError,
    DownloadSizeError,
    DownloadTimeoutError,
    ExistingDataError,
    InvalidArchiveError,
)


def zip_bytes(content: bytes = b"synthetic") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("proofwriter-dataset-test/OWA/depth-1/meta-train.jsonl", content)
    return buffer.getvalue()


def transport_for(content: bytes, *, status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=content,
            headers={"content-length": str(len(content)), "content-type": "application/zip"},
            request=request,
        )

    return httpx.MockTransport(handler)


def test_successful_mocked_download_is_atomic_and_hashed(tmp_path: Path) -> None:
    content = zip_bytes()
    destination = tmp_path / "archive.zip"

    result = download_archive(
        url="https://example.test/archive.zip",
        archive_path=destination,
        transport=transport_for(content),
    )

    assert destination.read_bytes() == content
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert result.size_bytes == len(content)
    assert not result.skipped
    assert not destination.with_suffix(".zip.part").exists()


class InterruptingStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b"partial"
        raise httpx.ReadError("interrupted")


class BytesStream(httpx.SyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __iter__(self):
        yield self.content


def test_interrupted_download_removes_partial_file(tmp_path: Path) -> None:
    destination = tmp_path / "archive.zip"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=InterruptingStream(), request=request)

    with pytest.raises(DatasetAcquisitionError, match="Network failure"):
        download_archive(
            url="https://example.test/archive.zip",
            archive_path=destination,
            transport=httpx.MockTransport(handler),
        )

    assert not destination.exists()
    assert not destination.with_suffix(".zip.part").exists()


def test_timeout_is_reported_and_partial_is_cleaned(tmp_path: Path) -> None:
    destination = tmp_path / "archive.zip"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(DownloadTimeoutError, match="Timed out"):
        download_archive(
            url="https://example.test/archive.zip",
            archive_path=destination,
            transport=httpx.MockTransport(handler),
        )

    assert not destination.with_suffix(".zip.part").exists()


def test_non_success_http_response_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetAcquisitionError, match="HTTP 503"):
        download_archive(
            url="https://example.test/archive.zip",
            archive_path=tmp_path / "archive.zip",
            transport=transport_for(b"unavailable", status_code=503),
        )


def test_declared_size_limit_is_rejected(tmp_path: Path) -> None:
    content = zip_bytes()
    with pytest.raises(DownloadSizeError, match="above limit"):
        download_archive(
            url="https://example.test/archive.zip",
            archive_path=tmp_path / "archive.zip",
            max_bytes=len(content) - 1,
            transport=transport_for(content),
        )


def test_streamed_size_limit_is_rejected_without_content_length(tmp_path: Path) -> None:
    content = zip_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BytesStream(content), request=request)

    destination = tmp_path / "archive.zip"
    with pytest.raises(DownloadSizeError, match="exceeded"):
        download_archive(
            url="https://example.test/archive.zip",
            archive_path=destination,
            max_bytes=len(content) - 1,
            chunk_size=10,
            transport=httpx.MockTransport(handler),
        )
    assert not destination.with_suffix(".zip.part").exists()


def test_invalid_zip_is_rejected_and_cleaned(tmp_path: Path) -> None:
    destination = tmp_path / "archive.zip"
    with pytest.raises(InvalidArchiveError):
        download_archive(
            url="https://example.test/archive.zip",
            archive_path=destination,
            transport=transport_for(b"not a zip"),
        )
    assert not destination.exists()
    assert not destination.with_suffix(".zip.part").exists()


def test_explicit_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    content = zip_bytes()
    with pytest.raises(ChecksumMismatchError, match="mismatch"):
        download_archive(
            url="https://example.test/archive.zip",
            archive_path=tmp_path / "archive.zip",
            expected_sha256="0" * 64,
            transport=transport_for(content),
        )


def test_valid_existing_archive_is_reused_without_network(tmp_path: Path) -> None:
    destination = tmp_path / "archive.zip"
    content = zip_bytes()
    destination.write_bytes(content)

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected network request: {request.url}")

    result = download_archive(
        url="https://example.test/archive.zip",
        archive_path=destination,
        transport=httpx.MockTransport(unexpected_request),
    )

    assert result.skipped
    assert result.sha256 == hashlib.sha256(content).hexdigest()


def test_force_replaces_existing_archive_only_after_validation(tmp_path: Path) -> None:
    destination = tmp_path / "archive.zip"
    original = zip_bytes(b"old")
    replacement = zip_bytes(b"new")
    destination.write_bytes(original)

    result = download_archive(
        url="https://example.test/archive.zip",
        archive_path=destination,
        force=True,
        transport=transport_for(replacement),
    )

    assert not result.skipped
    assert destination.read_bytes() == replacement


def test_invalid_existing_archive_requires_force(tmp_path: Path) -> None:
    destination = tmp_path / "archive.zip"
    destination.write_bytes(b"broken")

    with pytest.raises(ExistingDataError, match="--force"):
        download_archive(
            url="https://example.test/archive.zip",
            archive_path=destination,
            transport=transport_for(zip_bytes()),
        )


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")

    with pytest.raises(InvalidArchiveError, match="traversal"):
        safe_extract_zip(archive_path, tmp_path / "extract")

    assert not (tmp_path / "escape.txt").exists()
    assert not list((tmp_path / "extract").glob(".*.extracting-*"))


def test_zip_symbolic_link_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("proofwriter/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "target")

    with pytest.raises(InvalidArchiveError, match="Symbolic-link"):
        safe_extract_zip(archive_path, tmp_path / "extract")


def test_safe_extraction_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    archive_path = tmp_path / "safe.zip"
    content = zip_bytes(b"safe")
    archive_path.write_bytes(content)

    first = safe_extract_zip(archive_path, tmp_path / "extract")
    second = safe_extract_zip(archive_path, tmp_path / "extract")

    assert first == second
    assert first.name == hashlib.sha256(content).hexdigest()
    assert (first / "proofwriter-dataset-test/OWA/depth-1/meta-train.jsonl").read_bytes() == b"safe"
