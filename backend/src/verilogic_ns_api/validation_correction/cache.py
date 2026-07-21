from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from verilogic_ns_api.semantic_parsing.models import ParserResponse
from verilogic_ns_api.validation_correction.provider import CorrectionTaskRequest


class CorrectionCacheError(RuntimeError):
    pass


class CorrectionResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, request: CorrectionTaskRequest) -> Path:
        namespace = request.namespace.replace(".", "-")
        return self.root / namespace / request.request_hash[:2] / f"{request.request_hash}.json"

    def load(self, request: CorrectionTaskRequest) -> ParserResponse | None:
        path = self.path_for(request)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("request_identity") != request.identity():
                raise CorrectionCacheError("correction cache metadata mismatch")
            response = ParserResponse.model_validate(envelope.get("response"))
        except (OSError, ValueError, ValidationError, AttributeError) as error:
            raise CorrectionCacheError("correction cache entry is corrupt") from error
        if response.request_hash != request.request_hash:
            raise CorrectionCacheError("correction cache request hash mismatch")
        return response

    def store(self, request: CorrectionTaskRequest, response: ParserResponse) -> Path:
        if response.request_hash != request.request_hash:
            raise CorrectionCacheError("cannot cache a response for another request")
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
