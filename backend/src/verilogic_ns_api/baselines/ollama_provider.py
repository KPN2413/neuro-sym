from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Literal

import httpx
from pydantic import Field, ValidationError

from verilogic_ns_api.baselines.models import (
    BaselineOutput,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderStatus,
    ProviderTimingTelemetry,
    UsageTelemetry,
)
from verilogic_ns_api.baselines.provider import (
    InvalidProviderResponseError,
    ProviderConfigurationError,
    TransientProviderError,
)
from verilogic_ns_api.baselines.schema import baseline_output_schema
from verilogic_ns_api.research.models import StrictModel

LOOPBACK_ENDPOINTS = {"http://127.0.0.1:11434", "http://localhost:11434"}


def validate_ollama_endpoint(endpoint: str) -> str:
    if endpoint not in LOOPBACK_ENDPOINTS:
        raise ProviderConfigurationError("Ollama endpoint must be loopback HTTP on port 11434")
    return endpoint


class OllamaVersionResponse(StrictModel):
    version: str = Field(min_length=1, max_length=64)


class OllamaModelDetails(StrictModel):
    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: list[str] | None = None
    parameter_size: str = ""
    quantization_level: str = ""
    context_length: int | None = Field(default=None, ge=0)
    embedding_length: int | None = Field(default=None, ge=0)


class OllamaModelRecord(StrictModel):
    name: str
    model: str
    modified_at: datetime
    size: int = Field(ge=0)
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    details: OllamaModelDetails
    capabilities: list[str] | None = None


class OllamaTagsResponse(StrictModel):
    models: list[OllamaModelRecord]


class OllamaRunningModel(StrictModel):
    name: str
    model: str
    size: int = Field(ge=0)
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    details: OllamaModelDetails
    expires_at: datetime
    size_vram: int = Field(ge=0)
    context_length: int = Field(ge=0)


class OllamaProcessResponse(StrictModel):
    models: list[OllamaRunningModel]


class OllamaMessage(StrictModel):
    role: Literal["assistant"]
    content: str
    thinking: str | None = None


class OllamaChatResponse(StrictModel):
    model: str
    created_at: datetime
    message: OllamaMessage
    done: bool
    done_reason: str | None = None
    total_duration: int = Field(default=0, ge=0)
    load_duration: int = Field(default=0, ge=0)
    prompt_eval_count: int = Field(default=0, ge=0)
    prompt_eval_duration: int = Field(default=0, ge=0)
    eval_count: int = Field(default=0, ge=0)
    eval_duration: int = Field(default=0, ge=0)


class OllamaChatProvider:
    name = "ollama"

    def __init__(
        self,
        *,
        config: ProviderConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if config.name != "ollama":
            raise ProviderConfigurationError("Ollama provider requires an Ollama configuration")
        if config.endpoint is None:
            raise ProviderConfigurationError("Ollama endpoint is not configured")
        self.config = config
        self.endpoint = validate_ollama_endpoint(config.endpoint)
        self._client = httpx.Client(
            base_url=self.endpoint,
            timeout=config.timeout_seconds,
            transport=transport,
            trust_env=False,
            headers={"User-Agent": "VeriLogic-NS/0.1 local-baseline"},
        )
        self._validate_runtime()

    def close(self) -> None:
        self._client.close()

    def _json_request(self, method: str, path: str, **kwargs) -> object:
        try:
            response = self._client.request(method, path, **kwargs)
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            raise TransientProviderError("Transient local Ollama transport failure") from error
        if response.status_code >= 500:
            raise TransientProviderError("Transient local Ollama server failure")
        if response.status_code >= 400:
            raise ProviderConfigurationError(
                f"Local Ollama rejected the request with status {response.status_code}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise InvalidProviderResponseError("Ollama returned invalid JSON") from error

    def _validate_runtime(self) -> None:
        try:
            version = OllamaVersionResponse.model_validate(
                self._json_request("GET", "/api/version")
            )
            tags = OllamaTagsResponse.model_validate(self._json_request("GET", "/api/tags"))
        except ValidationError as error:
            raise InvalidProviderResponseError(
                "Ollama runtime metadata violated the expected schema"
            ) from error
        if version.version != self.config.provider_version:
            raise ProviderConfigurationError("Installed Ollama version does not match config")
        matches = [item for item in tags.models if item.name == self.config.model]
        if len(matches) != 1:
            raise ProviderConfigurationError("Configured Ollama model tag is not installed exactly")
        if matches[0].digest != self.config.model_digest:
            raise ProviderConfigurationError("Installed Ollama model digest does not match config")
        if len(tags.models) != 1:
            raise ProviderConfigurationError(
                "Local baseline requires exactly one installed Ollama model"
            )

    def _validate_request(self, request: LLMRequest) -> None:
        expected = {
            "provider": self.name,
            "api_family": self.config.api_family,
            "configured_model": self.config.model,
            "endpoint_identity": self.endpoint,
            "provider_version": self.config.provider_version,
            "model_digest": self.config.model_digest,
            "model_options": self.config.request_options(),
        }
        observed = {name: getattr(request, name) for name in expected}
        if observed != expected:
            raise ProviderConfigurationError(
                "Ollama request metadata does not match the frozen provider configuration"
            )

    def _execution_device(self) -> Literal["cpu", "gpu", "hybrid"]:
        try:
            processes = OllamaProcessResponse.model_validate(self._json_request("GET", "/api/ps"))
        except (ValidationError, TransientProviderError) as error:
            raise ProviderConfigurationError(
                "Could not verify the local Ollama execution device"
            ) from error
        matches = [item for item in processes.models if item.name == self.config.model]
        if len(matches) != 1:
            raise ProviderConfigurationError(
                "Loaded Ollama model was not found in process metadata"
            )
        model = matches[0]
        if model.size_vram == 0:
            observed: Literal["cpu", "gpu", "hybrid"] = "cpu"
        elif model.size_vram >= model.size:
            observed = "gpu"
        else:
            observed = "hybrid"
        if observed != self.config.execution_device:
            raise ProviderConfigurationError(
                "Observed Ollama execution device does not match the frozen configuration"
            )
        return observed

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._validate_request(request)
        options = {
            "temperature": request.model_options["temperature"],
            "seed": request.model_options["seed"],
            "num_ctx": request.model_options["num_ctx"],
            "num_predict": request.model_options["num_predict"],
        }
        payload = {
            "model": request.configured_model,
            "messages": [
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": request.input_text},
            ],
            "stream": False,
            "format": baseline_output_schema(),
            "options": options,
            "think": request.model_options["think"],
            "keep_alive": request.model_options["keep_alive"],
        }
        started_at = datetime.now(UTC)
        start = perf_counter()
        raw_response = self._json_request("POST", "/api/chat", json=payload)
        completed_at = datetime.now(UTC)
        latency_ms = (perf_counter() - start) * 1000
        try:
            response = OllamaChatResponse.model_validate(raw_response)
        except ValidationError as error:
            raise InvalidProviderResponseError(
                "Ollama response violated the strict native chat schema"
            ) from error
        if not response.done:
            raise InvalidProviderResponseError("Ollama response was incomplete")
        if response.model != request.configured_model:
            raise InvalidProviderResponseError("Ollama returned an unexpected model tag")
        if response.message.thinking:
            raise InvalidProviderResponseError(
                "Ollama emitted thinking content despite the disabled setting"
            )
        try:
            parsed = BaselineOutput.model_validate_json(response.message.content)
        except (ValidationError, ValueError) as error:
            raise InvalidProviderResponseError(
                "Ollama response violated the strict baseline output schema"
            ) from error

        execution_device = self._execution_device()
        generation_seconds = response.eval_duration / 1_000_000_000
        timings = ProviderTimingTelemetry(
            total_duration_ms=response.total_duration / 1_000_000,
            load_duration_ms=response.load_duration / 1_000_000,
            prompt_eval_duration_ms=response.prompt_eval_duration / 1_000_000,
            generation_duration_ms=response.eval_duration / 1_000_000,
            generation_tokens_per_second=(
                response.eval_count / generation_seconds if generation_seconds else None
            ),
        )
        usage = UsageTelemetry(
            input_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count,
            reasoning_tokens=0,
            cached_input_tokens=0,
            total_tokens=response.prompt_eval_count + response.eval_count,
        )
        sanitized_payload = {
            "model": response.model,
            "created_at": response.created_at.isoformat(),
            "message": {"role": response.message.role, "content": response.message.content},
            "thinking_present": False,
            "done": response.done,
            "done_reason": response.done_reason,
            "total_duration": response.total_duration,
            "load_duration": response.load_duration,
            "prompt_eval_count": response.prompt_eval_count,
            "prompt_eval_duration": response.prompt_eval_duration,
            "eval_count": response.eval_count,
            "eval_duration": response.eval_duration,
        }
        return LLMResponse(
            request_hash=request.cache_key,
            status=ProviderStatus.SUCCESS,
            label=parsed.label,
            configured_model=request.configured_model,
            returned_model=response.model,
            provider_version=self.config.provider_version,
            model_digest=self.config.model_digest,
            execution_device=execution_device,
            usage=usage,
            provider_timing=timings,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            estimated_cost_usd=0,
            raw_provider_payload=sanitized_payload,
        )
