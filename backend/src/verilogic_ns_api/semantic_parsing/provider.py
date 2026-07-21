from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter

import httpx
from pydantic import ValidationError

from verilogic_ns_api.baselines.ollama_provider import (
    LOOPBACK_ENDPOINTS,
    OllamaChatResponse,
    OllamaTagsResponse,
    OllamaVersionResponse,
)
from verilogic_ns_api.reasoning.models import sha256_payload
from verilogic_ns_api.semantic_parsing.models import (
    ParserResponse,
    ParserRuntimeConfig,
    ParserTiming,
    ParserUsage,
)


class ParserProviderError(RuntimeError):
    pass


class ParserTransientError(ParserProviderError):
    pass


class ParserTimeoutError(ParserTransientError):
    pass


class ParserConfigurationError(ParserProviderError):
    pass


class ParserStructuredOutputError(ParserProviderError):
    pass


class StructuredRequest:
    def __init__(
        self,
        *,
        kind: str,
        instructions: str,
        input_text: str,
        prompt_hash: str,
        input_hash: str,
        output_schema: dict[str, object],
        schema_hash: str,
        config: ParserRuntimeConfig,
    ) -> None:
        self.kind = kind
        self.instructions = instructions
        self.input_text = input_text
        self.prompt_hash = prompt_hash
        self.input_hash = input_hash
        self.output_schema = output_schema
        self.schema_hash = schema_hash
        self.config = config
        self.request_hash = sha256_payload(self.identity())

    def identity(self) -> dict[str, object]:
        return {
            "namespace": "semantic-parser.v1",
            "kind": self.kind,
            "provider": "ollama",
            "endpoint": self.config.endpoint,
            "provider_version": self.config.provider_version,
            "model": self.config.model,
            "model_digest": self.config.model_digest,
            "options": self.options(),
            "prompt_hash": self.prompt_hash,
            "input_hash": self.input_hash,
            "rendered_input_hash": sha256_payload({"text": self.input_text}),
            "schema_hash": self.schema_hash,
        }

    def options(self) -> dict[str, object]:
        return {
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "num_ctx": self.config.num_ctx,
            "num_predict": (
                self.config.theory_num_predict
                if self.kind == "theory"
                else self.config.query_num_predict
            ),
            "think": self.config.think,
            "keep_alive": self.config.keep_alive,
        }


class OllamaStructuredProvider:
    def __init__(
        self,
        config: ParserRuntimeConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if config.endpoint not in LOOPBACK_ENDPOINTS:
            raise ParserConfigurationError("semantic parser endpoint must be loopback")
        self.config = config
        self._client = httpx.Client(
            base_url=config.endpoint,
            timeout=config.timeout_seconds,
            transport=transport,
            trust_env=False,
            headers={"User-Agent": "VeriLogic-NS/0.1 semantic-parser"},
        )
        self._validate_runtime()

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: object) -> object:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as error:
            raise ParserTimeoutError("local Ollama request timed out") from error
        except httpx.NetworkError as error:
            raise ParserTransientError("local Ollama transport failed") from error
        if response.status_code >= 500:
            raise ParserTransientError("local Ollama returned a temporary server error")
        if response.status_code >= 400:
            raise ParserConfigurationError(
                f"local Ollama rejected the request with status {response.status_code}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise ParserStructuredOutputError("local Ollama returned invalid JSON") from error

    def _validate_runtime(self) -> None:
        try:
            version = OllamaVersionResponse.model_validate(self._request("GET", "/api/version"))
            tags = OllamaTagsResponse.model_validate(self._request("GET", "/api/tags"))
        except ValidationError as error:
            raise ParserConfigurationError("invalid local Ollama metadata") from error
        if version.version != self.config.provider_version:
            raise ParserConfigurationError("Ollama version differs from the frozen config")
        matches = [item for item in tags.models if item.name == self.config.model]
        if len(matches) != 1 or matches[0].digest != self.config.model_digest:
            raise ParserConfigurationError("Ollama model tag or digest differs from config")
        if "cloud" in self.config.model.lower():
            raise ParserConfigurationError("cloud models are forbidden")

    def complete(self, request: StructuredRequest) -> ParserResponse:
        if request.config != self.config:
            raise ParserConfigurationError("request runtime differs from provider runtime")
        options = request.options()
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": request.input_text},
            ],
            "stream": False,
            "format": request.output_schema,
            "options": {
                "temperature": options["temperature"],
                "seed": options["seed"],
                "num_ctx": options["num_ctx"],
                "num_predict": options["num_predict"],
            },
            "think": False,
            "keep_alive": options["keep_alive"],
        }
        started_at = datetime.now(UTC)
        started = perf_counter()
        raw = self._request("POST", "/api/chat", json=payload)
        completed_at = datetime.now(UTC)
        try:
            response = OllamaChatResponse.model_validate(raw)
        except ValidationError as error:
            raise ParserStructuredOutputError("Ollama chat response violated its schema") from error
        if not response.done or response.model != self.config.model:
            raise ParserStructuredOutputError(
                "Ollama returned an incomplete or unexpected response"
            )
        if response.message.thinking:
            raise ParserStructuredOutputError("thinking content was emitted despite think=false")
        try:
            content = json.loads(response.message.content)
        except json.JSONDecodeError as error:
            raise ParserStructuredOutputError("model output was not JSON") from error
        if not isinstance(content, dict):
            raise ParserStructuredOutputError("model output must be a JSON object")
        generation_seconds = response.eval_duration / 1_000_000_000
        return ParserResponse(
            request_hash=request.request_hash,
            configured_model=self.config.model,
            returned_model=response.model,
            provider_version=self.config.provider_version,
            model_digest=self.config.model_digest,
            content=content,
            usage=ParserUsage(
                input_tokens=response.prompt_eval_count,
                output_tokens=response.eval_count,
                total_tokens=response.prompt_eval_count + response.eval_count,
            ),
            timing=ParserTiming(
                total_duration_ms=response.total_duration / 1_000_000,
                load_duration_ms=response.load_duration / 1_000_000,
                prompt_eval_duration_ms=response.prompt_eval_duration / 1_000_000,
                generation_duration_ms=response.eval_duration / 1_000_000,
                generation_tokens_per_second=(
                    response.eval_count / generation_seconds if generation_seconds else None
                ),
            ),
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=(perf_counter() - started) * 1000,
        )
