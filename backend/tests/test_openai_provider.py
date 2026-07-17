from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from verilogic_ns_api.baselines.models import BaselineOutput, GoldLabel, ProviderStatus, sha256_text
from verilogic_ns_api.baselines.openai_provider import OpenAIResponsesProvider
from verilogic_ns_api.baselines.prompts import PromptTemplate, build_request
from verilogic_ns_api.baselines.provider import (
    AuthenticationProviderError,
    InvalidProviderResponseError,
    ProviderConfigurationError,
    TransientProviderError,
)
from verilogic_ns_api.research.models import (
    PredictionInput,
    SourceStatement,
    Split,
    WorldAssumption,
)


class FakeResponse:
    def __init__(self, *, parsed=None, refusal: str | None = None, status="completed"):
        self.id = "resp_test_123"
        self.model = "gpt-test-returned-2026-07-01"
        self.status = status
        self.output_parsed = parsed
        self.output = []
        if refusal is not None:
            self.output = [
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="refusal", refusal=refusal)],
                )
            ]
        self.usage = SimpleNamespace(
            input_tokens=12,
            output_tokens=5,
            total_tokens=17,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )

    def model_dump(self, **kwargs):
        return {"id": self.id, "model": self.model, "status": self.status}


class FakeClient:
    response: FakeResponse
    kwargs: dict
    error: Exception | None = None

    def __init__(self, **kwargs):
        self.responses = self
        self.client_kwargs = kwargs

    def parse(self, **kwargs):
        type(self).kwargs = kwargs
        if type(self).error is not None:
            raise type(self).error
        return type(self).response


def inference_input() -> PredictionInput:
    return PredictionInput(
        example_id="synthetic/Q1",
        dataset_name="proofwriter",
        dataset_version="synthetic",
        variant="synthetic",
        split=Split.DEVELOPMENT,
        theory_id="T",
        question_id="Q1",
        reasoning_depth=1,
        source_statements=[SourceStatement(source_id="s", text="Ari is calm.", kind="fact")],
        context="Ari is calm.",
        query="Ari is calm.",
        world_assumption=WorldAssumption.OPEN,
        structured_facts={},
        structured_rules={},
        source_relative_path="synthetic",
    )


def llm_request():
    text = "Return one label."
    return build_request(
        template=PromptTemplate(None, "v1", text, sha256_text(text)),  # type: ignore[arg-type]
        example=inference_input(),
        configured_model="gpt-test-configured",
        reasoning_effort="low",
        max_output_tokens=64,
        output_schema_hash="a" * 64,
    )


def provider(monkeypatch: pytest.MonkeyPatch, response: FakeResponse):
    FakeClient.response = response
    FakeClient.error = None
    monkeypatch.setattr("verilogic_ns_api.baselines.openai_provider.OpenAI", FakeClient)
    return OpenAIResponsesProvider(api_key="test-only-key", timeout_seconds=5)


def test_success_maps_strict_output_usage_and_returned_model(monkeypatch) -> None:
    result = provider(
        monkeypatch, FakeResponse(parsed=BaselineOutput(label=GoldLabel.ENTAILED))
    ).complete(llm_request())
    assert result.status is ProviderStatus.SUCCESS
    assert result.label is GoldLabel.ENTAILED
    assert result.provider_request_id == "resp_test_123"
    assert result.configured_model == "gpt-test-configured"
    assert result.returned_model == "gpt-test-returned-2026-07-01"
    assert result.usage.reasoning_tokens == 2
    assert result.usage.cached_input_tokens == 3
    assert FakeClient.kwargs["reasoning"] == {"effort": "low"}
    assert FakeClient.kwargs["text_format"] is BaselineOutput
    assert FakeClient.kwargs["store"] is False
    assert "tools" not in FakeClient.kwargs
    assert "temperature" not in FakeClient.kwargs


def test_explicit_provider_refusal_maps_without_retry_prompt(monkeypatch) -> None:
    result = provider(monkeypatch, FakeResponse(refusal="cannot comply")).complete(llm_request())
    assert result.status is ProviderStatus.REFUSAL
    assert result.refusal_reason == "provider_refusal"
    assert result.label is None


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(parsed=None),
        FakeResponse(parsed=BaselineOutput(label=GoldLabel.UNKNOWN), status="incomplete"),
    ],
)
def test_missing_or_incomplete_strict_output_is_typed_error(monkeypatch, response) -> None:
    with pytest.raises(InvalidProviderResponseError):
        provider(monkeypatch, response).complete(llm_request())


def test_missing_api_key_is_rejected_before_client_construction() -> None:
    with pytest.raises(AuthenticationProviderError, match="OPENAI_API_KEY"):
        OpenAIResponsesProvider(api_key="", timeout_seconds=5)


def test_sdk_retries_are_disabled_and_key_is_never_exposed(monkeypatch) -> None:
    instance = provider(monkeypatch, FakeResponse(parsed=BaselineOutput(label=GoldLabel.UNKNOWN)))
    assert instance._client.client_kwargs["max_retries"] == 0
    assert instance._client.client_kwargs["api_key"] == "test-only-key"
    assert "test-only-key" not in repr(instance)


@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")),
        openai.RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com")),
            body=None,
        ),
        openai.InternalServerError(
            "server failure",
            response=httpx.Response(500, request=httpx.Request("POST", "https://api.openai.com")),
            body=None,
        ),
    ],
)
def test_timeout_rate_limit_and_server_failures_map_to_transient(monkeypatch, error) -> None:
    instance = provider(monkeypatch, FakeResponse())
    FakeClient.error = error
    with pytest.raises(TransientProviderError):
        instance.complete(llm_request())


def test_authentication_and_unsupported_model_fail_without_fallback(monkeypatch) -> None:
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com"))
    instance = provider(monkeypatch, FakeResponse())
    FakeClient.error = openai.AuthenticationError("bad key", response=response, body=None)
    with pytest.raises(AuthenticationProviderError):
        instance.complete(llm_request())

    response = httpx.Response(404, request=httpx.Request("POST", "https://api.openai.com"))
    FakeClient.error = openai.NotFoundError("unknown model", response=response, body=None)
    with pytest.raises(ProviderConfigurationError):
        instance.complete(llm_request())
