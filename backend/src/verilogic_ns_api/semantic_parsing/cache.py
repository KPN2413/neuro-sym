from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from verilogic_ns_api.semantic_parsing.models import ParserResponse
from verilogic_ns_api.semantic_parsing.provider import StructuredRequest


class ParserCacheError(RuntimeError):
    pass


class ParserResponseCache:
    """A parser-only, content-addressed cache with atomic replacement."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, request: StructuredRequest) -> Path:
        return self.root / request.request_hash[:2] / f"{request.request_hash}.json"

    def load(self, request: StructuredRequest) -> ParserResponse | None:
        path = self.path_for(request)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("request_identity") != request.identity():
                raise ParserCacheError("parser cache metadata mismatch")
            response = ParserResponse.model_validate(envelope.get("response"))
        except (OSError, ValueError, ValidationError, AttributeError) as error:
            raise ParserCacheError("parser cache entry is corrupt") from error
        if response.request_hash != request.request_hash:
            raise ParserCacheError("parser cache request hash mismatch")
        return response

    def store(self, request: StructuredRequest, response: ParserResponse) -> Path:
        if response.request_hash != request.request_hash:
            raise ParserCacheError("cannot cache a response for another request")
        path = self.path_for(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": "1.0",
            "request_identity": request.identity(),
            "response": response.model_dump(mode="json"),
        }
        handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    envelope, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path
