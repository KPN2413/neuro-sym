from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from verilogic_ns_api.baselines.cache import ResponseCache
from verilogic_ns_api.baselines.configuration import load_config
from verilogic_ns_api.baselines.models import (
    ProviderConfig,
    ProviderStatus,
    RetryConfig,
    sha256_text,
)
from verilogic_ns_api.baselines.ollama_provider import OllamaChatProvider
from verilogic_ns_api.baselines.prompts import PromptTemplate, build_request
from verilogic_ns_api.baselines.provider import (
    CachedProvider,
    InvalidProviderResponseError,
    ProviderConfigurationError,
    RetryingProvider,
    TransientProviderError,
)
from verilogic_ns_api.research.models import (
    GoldLabel,
    PredictionInput,
    SourceStatement,
    Split,
    WorldAssumption,
)

MODEL = "qwen3.5:4b-q4_K_M"
DIGEST = "a" * 64
ENDPOINT = "http://127.0.0.1:11434"
ROOT = Path(__file__).parents[2]


def provider_config(**updates) -> ProviderConfig:
    payload = {
        "name": "ollama",
        "api_family": "native_chat",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "model_digest": DIGEST,
        "provider_version": "0.31.2",
        "reasoning_effort": "none",
        "temperature": 0,
        "sampling_seed": 20260713,
        "context_tokens": 4096,
        "think": False,
        "keep_alive": "30m",
        "execution_device": "cpu",
        "max_output_tokens": 128,
        "timeout_seconds": 30,
        "concurrency": 1,
    }
    payload.update(updates)
    return ProviderConfig.model_validate(payload)


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


def llm_request(config: ProviderConfig | None = None):
    config = config or provider_config()
    text = "Return one label."
    return build_request(
        template=PromptTemplate(None, "v1", text, sha256_text(text)),  # type: ignore[arg-type]
        example=inference_input(),
        configured_model=config.model,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=config.max_output_tokens,
        output_schema_hash="b" * 64,
        selection_manifest_hash="c" * 64,
        provider=config.name,
        api_family=config.api_family,
        endpoint_identity=config.endpoint,
        provider_version=config.provider_version,
        model_digest=config.model_digest,
        model_options=config.request_options(),
    )


def tags_payload(*, digest: str = DIGEST, extra_models: bool = False) -> dict:
    model = {
        "name": MODEL,
        "model": MODEL,
        "modified_at": "2026-07-13T10:00:00Z",
        "size": 3_389_983_735,
        "digest": digest,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "qwen35",
            "families": ["qwen35"],
            "parameter_size": "4.7B",
            "quantization_level": "Q4_K_M",
        },
    }
    models = [model]
    if extra_models:
        models.append({**model, "name": "extra:latest", "model": "extra:latest"})
    return {"models": models}


def chat_payload(*, content: str = '{"label":"ENTAILED"}', thinking: str = "") -> dict:
    return {
        "model": MODEL,
        "created_at": "2026-07-13T10:01:00Z",
        "message": {"role": "assistant", "content": content, "thinking": thinking},
        "done": True,
        "done_reason": "stop",
        "total_duration": 2_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 40,
        "prompt_eval_duration": 600_000_000,
        "eval_count": 5,
        "eval_duration": 500_000_000,
    }


def process_payload(*, size_vram: int = 0) -> dict:
    tag = tags_payload()["models"][0]
    model = {name: tag[name] for name in ("name", "model", "size", "digest", "details")}
    return {
        "models": [
            {
                **model,
                "expires_at": "2026-07-13T10:31:00Z",
                "size_vram": size_vram,
                "context_length": 4096,
            }
        ]
    }


def mock_transport(
    *,
    version: str = "0.31.2",
    tags: dict | None = None,
    chat: dict | None = None,
    paths: list[str] | None = None,
    request_bodies: list[dict] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if paths is not None:
            paths.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": version})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags or tags_payload())
        if request.url.path == "/api/chat":
            if request_bodies is not None:
                request_bodies.append(json.loads(request.content))
            return httpx.Response(200, json=chat or chat_payload())
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=process_payload())
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("label", list(GoldLabel))
def test_success_uses_strict_local_chat_and_maps_telemetry(label: GoldLabel) -> None:
    paths: list[str] = []
    bodies: list[dict] = []
    config = provider_config()
    provider = OllamaChatProvider(
        config=config,
        transport=mock_transport(
            paths=paths,
            request_bodies=bodies,
            chat=chat_payload(content=json.dumps({"label": label.value})),
        ),
    )
    result = provider.complete(llm_request(config))
    assert result.status is ProviderStatus.SUCCESS
    assert result.label is label
    assert result.returned_model == MODEL
    assert result.model_digest == DIGEST
    assert result.provider_version == "0.31.2"
    assert result.execution_device == "cpu"
    assert result.usage.input_tokens == 40
    assert result.usage.output_tokens == 5
    assert result.usage.reasoning_tokens == 0
    assert result.provider_timing.total_duration_ms == 2000
    assert result.provider_timing.generation_tokens_per_second == 10
    assert paths == ["/api/version", "/api/tags", "/api/chat", "/api/ps"]
    payload = bodies[0]
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0.0,
        "seed": 20260713,
        "num_ctx": 4096,
        "num_predict": 128,
    }
    assert payload["format"]["additionalProperties"] is False
    assert "tools" not in payload
    assert result.raw_provider_payload["thinking_present"] is False  # type: ignore[index]


def test_cache_identity_covers_all_local_reproducibility_fields() -> None:
    original = llm_request()
    variants = [
        original.model_copy(update={"provider_version": "0.31.3"}),
        original.model_copy(update={"model_digest": "d" * 64}),
        original.model_copy(update={"endpoint_identity": "http://localhost:11434"}),
        original.model_copy(update={"selection_manifest_hash": "e" * 64}),
        original.model_copy(update={"model_options": {**original.model_options, "seed": 7}}),
    ]
    assert all(item.cache_key != original.cache_key for item in variants)
    openai_identity = original.model_copy(
        update={
            "provider": "openai",
            "api_family": "responses",
            "endpoint_identity": None,
            "provider_version": None,
            "model_digest": None,
            "model_options": {},
        }
    )
    assert openai_identity.cache_key != original.cache_key


@pytest.mark.parametrize(
    "updates",
    [
        {"endpoint": "http://0.0.0.0:11434"},
        {"endpoint": "https://127.0.0.1:11434"},
        {"model": "qwen3.5:cloud"},
        {"think": True},
        {"temperature": 0.1},
        {"concurrency": 2},
    ],
)
def test_unsafe_or_nonfrozen_config_is_rejected(updates: dict) -> None:
    with pytest.raises(ValidationError):
        provider_config(**updates)


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (mock_transport(version="0.31.3"), "version"),
        (mock_transport(tags=tags_payload(digest="f" * 64)), "digest"),
        (mock_transport(tags=tags_payload(extra_models=True)), "exactly one"),
        (mock_transport(tags={"models": []}), "not installed"),
    ],
)
def test_runtime_drift_blocks_before_inference(transport, message: str) -> None:
    with pytest.raises(ProviderConfigurationError, match=message):
        OllamaChatProvider(config=provider_config(), transport=transport)


@pytest.mark.parametrize(
    "content", ["not-json", '{"label":"ABSTAIN"}', '{"label":"UNKNOWN","extra":1}']
)
def test_invalid_structured_output_is_typed_failure(content: str) -> None:
    provider = OllamaChatProvider(
        config=provider_config(),
        transport=mock_transport(chat=chat_payload(content=content)),
    )
    with pytest.raises(InvalidProviderResponseError, match="strict baseline"):
        provider.complete(llm_request())


def test_thinking_content_is_rejected_and_never_cached(tmp_path: Path) -> None:
    secret_reasoning = "private hidden reasoning"
    provider = OllamaChatProvider(
        config=provider_config(),
        transport=mock_transport(chat=chat_payload(thinking=secret_reasoning)),
    )
    cache = ResponseCache(tmp_path)
    with pytest.raises(InvalidProviderResponseError) as captured:
        CachedProvider(provider, cache).complete(llm_request())
    assert secret_reasoning not in str(captured.value)
    assert cache.read(llm_request()) is None
    assert not list(tmp_path.rglob("*.json"))


def test_mocked_local_response_caches_and_replays_without_chat(tmp_path: Path) -> None:
    paths: list[str] = []
    config = provider_config()
    inner = OllamaChatProvider(config=config, transport=mock_transport(paths=paths))
    cache = ResponseCache(tmp_path)
    first = CachedProvider(inner, cache).complete(llm_request(config))
    replay = CachedProvider(None, cache, replay_only=True).complete(llm_request(config))
    assert first.cache_hit is False
    assert replay.cache_hit is True
    assert paths.count("/api/chat") == 1


def test_local_connection_failure_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("local service unavailable", request=request)

    with pytest.raises(TransientProviderError):
        OllamaChatProvider(config=provider_config(), transport=httpx.MockTransport(handler))


def test_chat_timeout_is_transient_and_bounded_retry_reuses_same_request() -> None:
    chat_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_calls
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.31.2"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags_payload())
        if request.url.path == "/api/chat":
            chat_calls += 1
            if chat_calls == 1:
                raise httpx.ReadTimeout("synthetic timeout", request=request)
            return httpx.Response(200, json=chat_payload(content='{"label":"UNKNOWN"}'))
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=process_payload())
        return httpx.Response(404)

    local = OllamaChatProvider(config=provider_config(), transport=httpx.MockTransport(handler))
    retrying = RetryingProvider(
        local,
        RetryConfig(
            max_attempts=2,
            base_delay_seconds=0,
            max_delay_seconds=0,
            jitter_seconds=0,
        ),
        sleep=lambda _: None,
        random_uniform=lambda *_: 0,
    )
    item = llm_request()
    result = retrying.complete(item)
    assert result.label is GoldLabel.UNKNOWN
    assert result.retry_count == 1
    assert result.request_hash == item.cache_key
    assert chat_calls == 2


def test_frozen_local_configs_match_except_for_demonstrations() -> None:
    direct = load_config(ROOT / "experiments/configs/ollama-direct-pilot.yaml")
    few = load_config(ROOT / "experiments/configs/ollama-few-shot-pilot.yaml")
    assert direct.provider == few.provider
    assert direct.dataset == few.dataset
    assert direct.seed == few.seed
    assert direct.predictor_version == few.predictor_version
    assert direct.pricing is None and few.pricing is None
    assert direct.prompt.expected_sha256 == few.prompt.expected_sha256
    assert direct.prompt.output_schema_sha256 == few.prompt.output_schema_sha256
    assert direct.prompt.demonstration_manifest is None
    assert few.prompt.demonstration_manifest is not None
    assert (
        direct.provider.model_digest
        == "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd"
    )
    assert direct.provider.request_options() == few.provider.request_options()
