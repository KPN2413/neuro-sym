from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import openai
from openai import OpenAI
from pydantic import ValidationError

from verilogic_ns_api.baselines.models import (
    BaselineOutput,
    LLMRequest,
    LLMResponse,
    ProviderStatus,
    UsageTelemetry,
)
from verilogic_ns_api.baselines.provider import (
    AuthenticationProviderError,
    InvalidProviderResponseError,
    ProviderConfigurationError,
    TransientProviderError,
)


class OpenAIResponsesProvider:
    name = "openai"

    def __init__(self, *, api_key: str, timeout_seconds: float) -> None:
        if not api_key:
            raise AuthenticationProviderError("OPENAI_API_KEY is not configured")
        self._client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _refusal(response: Any) -> str | None:
        for output in getattr(response, "output", []):
            if getattr(output, "type", None) != "message":
                continue
            for content in getattr(output, "content", []):
                if getattr(content, "type", None) == "refusal":
                    return str(getattr(content, "refusal", "provider_refusal"))
        return None

    def complete(self, request: LLMRequest) -> LLMResponse:
        started_at = datetime.now(UTC)
        start = perf_counter()
        try:
            response = self._client.responses.parse(
                model=request.configured_model,
                reasoning={"effort": request.reasoning_effort},
                instructions=request.instructions,
                input=request.input_text,
                max_output_tokens=request.max_output_tokens,
                text_format=BaselineOutput,
                store=False,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            raise AuthenticationProviderError(
                "OpenAI authentication or permission failure"
            ) from error
        except (
            openai.BadRequestError,
            openai.NotFoundError,
            openai.UnprocessableEntityError,
        ) as error:
            raise ProviderConfigurationError(
                "OpenAI rejected the model, request, or structured-output configuration"
            ) from error
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as error:
            raise TransientProviderError("Transient OpenAI transport failure") from error
        except openai.APIStatusError as error:
            if error.status_code >= 500:
                raise TransientProviderError("Transient OpenAI server failure") from error
            raise ProviderConfigurationError("Non-retryable OpenAI API failure") from error
        except (json.JSONDecodeError, ValidationError) as error:
            raise InvalidProviderResponseError(
                "OpenAI response violated the strict baseline output schema"
            ) from error

        completed_at = datetime.now(UTC)
        latency_ms = (perf_counter() - start) * 1000
        usage = getattr(response, "usage", None)
        usage_model = UsageTelemetry()
        if usage is not None:
            input_details = getattr(usage, "input_tokens_details", None)
            output_details = getattr(usage, "output_tokens_details", None)
            usage_model = UsageTelemetry(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
                cached_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )
        raw_payload = response.model_dump(mode="json", exclude={"instructions"})
        refusal = self._refusal(response)
        if refusal is not None:
            return LLMResponse(
                request_hash=request.cache_key,
                status=ProviderStatus.REFUSAL,
                refusal_reason="provider_refusal",
                provider_request_id=response.id,
                configured_model=request.configured_model,
                returned_model=response.model,
                usage=usage_model,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                raw_provider_payload=raw_payload,
            )
        if getattr(response, "status", None) != "completed":
            raise InvalidProviderResponseError("OpenAI response was incomplete")
        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, BaselineOutput):
            raise InvalidProviderResponseError(
                "OpenAI response did not contain the strict baseline output"
            )
        return LLMResponse(
            request_hash=request.cache_key,
            status=ProviderStatus.SUCCESS,
            label=parsed.label,
            provider_request_id=response.id,
            configured_model=request.configured_model,
            returned_model=response.model,
            usage=usage_model,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            raw_provider_payload=raw_payload,
        )
