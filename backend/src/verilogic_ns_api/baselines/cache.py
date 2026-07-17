from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from verilogic_ns_api.baselines.models import CacheEntry, LLMRequest, LLMResponse


class CacheError(RuntimeError):
    pass


class CacheMetadataMismatch(CacheError):
    pass


class ReplayCacheMiss(CacheError):
    pass


class ResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path_for(self, request: LLMRequest) -> Path:
        key = request.cache_key
        return self.root / key[:2] / f"{key}.json"

    def relative_reference(self, request: LLMRequest) -> str:
        return self.path_for(request).relative_to(self.root).as_posix()

    def read(self, request: LLMRequest) -> LLMResponse | None:
        path = self.path_for(request)
        if not path.exists():
            return None
        try:
            entry = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError, ValueError, json.JSONDecodeError):
            corrupt = path.with_suffix(f".corrupt-{uuid4().hex}.json")
            with suppress(OSError):
                os.replace(path, corrupt)
            return None
        expected_identity = request.cache_identity()
        if entry.cache_key != request.cache_key or entry.request_identity != expected_identity:
            raise CacheMetadataMismatch(f"Cache metadata mismatch for request {request.cache_key}")
        if entry.response.request_hash != request.cache_key:
            raise CacheMetadataMismatch(
                f"Cached response hash mismatch for request {request.cache_key}"
            )
        return entry.response.model_copy(
            update={"cache_hit": True, "latency_ms": 0.0, "estimated_cost_usd": 0.0}
        )

    def write(self, request: LLMRequest, response: LLMResponse) -> Path:
        if response.request_hash != request.cache_key:
            raise CacheMetadataMismatch("Response request hash does not match cache key")
        path = self.path_for(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".tmp-{uuid4().hex}.json")
        entry = CacheEntry(
            cache_key=request.cache_key,
            request_identity=request.cache_identity(),
            response=response.model_copy(update={"cache_hit": False}),
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(entry.model_dump(mode="json"), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            with suppress(FileNotFoundError):
                temporary.unlink()
            raise
        return path

    @contextmanager
    def lock(
        self,
        request: LLMRequest,
        *,
        timeout_seconds: float = 90.0,
        poll_seconds: float = 0.05,
        stale_after_seconds: float = 300.0,
    ):
        path = self.path_for(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        deadline = time.monotonic() + timeout_seconds
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                with suppress(OSError):
                    age = time.time() - lock_path.stat().st_mtime
                    if age >= stale_after_seconds:
                        lock_path.unlink()
                        continue
                if time.monotonic() >= deadline:
                    raise CacheError(f"Timed out waiting for cache lock {lock_path.name}") from None
                time.sleep(poll_seconds)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            descriptor = None
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                lock_path.unlink()
