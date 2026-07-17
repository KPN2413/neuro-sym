from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from pydantic import ValidationError

from verilogic_ns_api.baselines.cache import CacheError, ResponseCache
from verilogic_ns_api.baselines.comparison import compare_runs, write_comparison
from verilogic_ns_api.baselines.configuration import (
    PreparedBaseline,
    prepare_baseline,
    repository_root,
    resolve_repository_path,
)
from verilogic_ns_api.baselines.models import BaselineCondition, ProviderConfig
from verilogic_ns_api.baselines.ollama_provider import OllamaChatProvider
from verilogic_ns_api.baselines.openai_provider import OpenAIResponsesProvider
from verilogic_ns_api.baselines.planning import build_predictor, plan_baseline
from verilogic_ns_api.baselines.predictors import LLMBaselinePredictor
from verilogic_ns_api.baselines.prompts import Demonstration, load_prompt_template
from verilogic_ns_api.baselines.provider import (
    BudgetGuard,
    CachedProvider,
    CircuitBreakingProvider,
    DeterministicFakeProvider,
    ProviderError,
    RetryingProvider,
    estimate_request_input_tokens,
    worst_case_request_cost_usd,
)
from verilogic_ns_api.baselines.schema import load_schema_hash
from verilogic_ns_api.datasets.errors import DatasetError
from verilogic_ns_api.datasets.proofwriter import ProofWriterLoader
from verilogic_ns_api.evaluation.runner import EvaluationRunner, ExistingRunError
from verilogic_ns_api.research.models import (
    GoldLabel,
    PredictionInput,
    SourceStatement,
    Split,
    WorldAssumption,
)


class LiveApprovalError(ValueError):
    pass


def _add_live_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--confirm-external-data-transfer", action="store_true")
    parser.add_argument("--max-cost-usd", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m verilogic_ns_api.baselines",
        description="Plan, run, replay, and compare frozen VeriLogic-NS LLM baselines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Validate and cost a pilot without network calls")
    plan.add_argument("--config", type=Path, required=True)

    run = subparsers.add_parser("run", help="Run a live or cache-only baseline evaluation")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--mode", choices=("live", "replay"), required=True)
    _add_live_flags(run)

    smoke = subparsers.add_parser(
        "smoke", help="Exercise prediction, caching, evaluation, and replay with a fake provider"
    )
    smoke.add_argument(
        "--condition", choices=[item.value for item in BaselineCondition], required=True
    )
    smoke.add_argument("--output-root", type=Path, default=Path("results/smoke"))

    ollama_smoke = subparsers.add_parser(
        "ollama-smoke",
        help="Exercise the native Ollama adapter and replay with mocked local HTTP",
    )
    ollama_smoke.add_argument(
        "--output-root", type=Path, default=Path("results/smoke/ollama-mocked")
    )

    compare = subparsers.add_parser("compare", help="Create a paired direct/few-shot report")
    compare.add_argument("--direct-run", type=Path, required=True)
    compare.add_argument("--few-shot-run", type=Path, required=True)
    compare.add_argument("--selection-manifest", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    canary = subparsers.add_parser(
        "canary", help="Send exactly one synthetic request per frozen condition, then replay"
    )
    canary.add_argument("--direct-config", type=Path, required=True)
    canary.add_argument("--few-shot-config", type=Path, required=True)
    _add_live_flags(canary)

    canary_plan = subparsers.add_parser(
        "canary-plan", help="Plan the two synthetic canaries without network calls"
    )
    canary_plan.add_argument("--direct-config", type=Path, required=True)
    canary_plan.add_argument("--few-shot-config", type=Path, required=True)
    return parser


def _validate_live_gate(args, required_cost_usd: float) -> float:
    if not args.allow_paid_api:
        raise LiveApprovalError("Live calls require --allow-paid-api")
    if not args.confirm_external_data_transfer:
        raise LiveApprovalError("Live calls require --confirm-external-data-transfer")
    if args.max_cost_usd is None or args.max_cost_usd <= 0:
        raise LiveApprovalError("Live calls require a positive --max-cost-usd")
    if args.max_cost_usd + 1e-12 < required_cost_usd:
        raise LiveApprovalError(
            f"Cost cap ${args.max_cost_usd:.6f} is below the preflight worst case "
            f"${required_cost_usd:.6f}"
        )
    return args.max_cost_usd


def _validate_pricing_is_current(prepared: PreparedBaseline) -> None:
    pricing = prepared.config.pricing
    if pricing is None:
        raise LiveApprovalError("Paid provider configuration is missing pricing metadata")
    age_days = (datetime.now(UTC).date() - pricing.as_of).days
    if age_days < 0 or age_days > 7:
        raise LiveApprovalError(
            "Pricing metadata is not current enough for a paid run; verify official pricing "
            "and update the dated configuration"
        )


def _live_cached_provider(
    prepared: PreparedBaseline,
    cache: ResponseCache,
    budget: BudgetGuard | None = None,
) -> CachedProvider:
    if prepared.config.provider.name == "ollama":
        provider = OllamaChatProvider(config=prepared.config.provider)
        retrying = RetryingProvider(provider, prepared.config.provider.retry)
        protected = CircuitBreakingProvider(
            retrying,
            threshold=prepared.config.provider.retry.circuit_breaker_threshold,
        )
        return CachedProvider(protected, cache)

    if budget is None or prepared.config.pricing is None:
        raise LiveApprovalError("Paid provider requires a budget and pricing metadata")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LiveApprovalError(
            "OPENAI_API_KEY is not configured; set it securely in the local environment"
        )
    provider = OpenAIResponsesProvider(
        api_key=api_key,
        timeout_seconds=prepared.config.provider.timeout_seconds,
    )
    retrying = RetryingProvider(provider, prepared.config.provider.retry)
    protected = CircuitBreakingProvider(
        retrying,
        threshold=prepared.config.provider.retry.circuit_breaker_threshold,
    )
    return CachedProvider(
        protected,
        cache,
        budget=budget,
        pricing=prepared.config.pricing,
    )


def _run_baseline(args) -> Path:
    prepared = prepare_baseline(args.config)
    plan, cache = plan_baseline(prepared)
    if args.mode == "replay":
        if plan.new_provider_requests:
            raise ValueError(
                "Replay cache is incomplete: "
                f"{plan.new_provider_requests} of {plan.planned_requests} requests are missing"
            )
        provider = CachedProvider(None, cache, replay_only=True)
    elif prepared.config.provider.name == "ollama":
        provider = _live_cached_provider(prepared, cache)
    else:
        _validate_pricing_is_current(prepared)
        cap = _validate_live_gate(args, plan.estimated_worst_case_usd)
        provider = _live_cached_provider(prepared, cache, BudgetGuard(cap))

    predictor = build_predictor(prepared, provider, cache)
    runner = EvaluationRunner(
        output_root=resolve_repository_path(
            prepared.repository_root, prepared.config.run.output_directory
        ),
        dataset_manifest_reference=prepared.config.dataset.dataset_manifest_reference,
        configuration=prepared.config.model_dump(mode="json"),
        seed=prepared.config.seed,
        selected_splits=[Split.DEVELOPMENT],
    )
    result = runner.run(
        predictor,
        prepared.pilot_examples,
        run_id=prepared.config.run.run_id,
        run_id_prefix=f"{prepared.config.run.run_id_prefix}-{args.mode}",
        max_workers=(prepared.config.provider.concurrency if args.mode == "live" else 1),
    )
    return result.run_directory


def _synthetic_input(identifier: str = "synthetic/canary/1") -> PredictionInput:
    return PredictionInput(
        example_id=identifier,
        dataset_name="proofwriter",
        dataset_version="synthetic",
        variant="synthetic",
        split=Split.DEVELOPMENT,
        theory_id="synthetic-canary",
        question_id="Q1",
        reasoning_depth=1,
        source_statements=[SourceStatement(source_id="fact1", text="Ari is calm.", kind="fact")],
        context="Ari is calm.",
        query="Ari is calm.",
        world_assumption=WorldAssumption.OPEN,
        structured_facts={},
        structured_rules={},
        source_relative_path="synthetic/in-memory",
    )


def _smoke(condition: BaselineCondition, output_root: Path) -> dict:
    root = repository_root(Path.cwd())
    fixture_root = root / "backend/tests/fixtures/proofwriter/proofwriter-dataset-V2020.12.3"
    loader = ProofWriterLoader(fixture_root)
    examples = list(loader.iter_examples(variant="depth-1", split=Split.DEVELOPMENT))
    train = list(loader.iter_examples(variant="depth-1", split=Split.TRAIN))
    prompt_name = "few_shot" if condition is BaselineCondition.FEW_SHOT else "direct"
    template = load_prompt_template(root / f"prompts/baselines/{prompt_name}/v1.md", version="v1")
    schema_hash = load_schema_hash(root / "schemas/llm-baseline-output.v1.schema.json")
    demonstrations: list[Demonstration] = []
    demo_hash = None
    if condition is BaselineCondition.FEW_SHOT:
        labels = [label for label in GoldLabel for _ in range(2)]
        demonstrations = [
            Demonstration(
                example=train[index % len(train)]
                .for_prediction()
                .model_copy(update={"example_id": f"synthetic/demo/{index + 1}"}),
                label=label,
            )
            for index, label in enumerate(labels)
        ]
        demo_hash = "0" * 64

    cache = ResponseCache(output_root / "cache" / condition.value)
    fake = DeterministicFakeProvider()
    live_like = CachedProvider(fake, cache)
    predictor = LLMBaselinePredictor(
        condition=condition,
        provider=live_like,
        cache=cache,
        template=template,
        configured_model="fake-model",
        reasoning_effort="low",
        max_output_tokens=64,
        output_schema_hash=schema_hash,
        demonstrations=demonstrations,
        demonstration_manifest_hash=demo_hash,
    )
    runner = EvaluationRunner(
        output_root=output_root / "runs",
        dataset_manifest_reference="synthetic-fixture-only",
        configuration={"synthetic": True, "condition": condition.value},
        seed=20260713,
        selected_splits=[Split.DEVELOPMENT],
    )
    nonce = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    first = runner.run(predictor, examples, run_id=f"smoke-{condition.value}-{nonce}")

    replay_predictor = LLMBaselinePredictor(
        condition=condition,
        provider=CachedProvider(None, cache, replay_only=True),
        cache=cache,
        template=template,
        configured_model="fake-model",
        reasoning_effort="low",
        max_output_tokens=64,
        output_schema_hash=schema_hash,
        demonstrations=demonstrations,
        demonstration_manifest_hash=demo_hash,
    )
    replay = runner.run(
        replay_predictor,
        examples,
        run_id=f"smoke-{condition.value}-replay-{nonce}",
    )
    if not all(record.cache_hit for record in replay.predictions):
        raise AssertionError("Synthetic replay did not use the cache for every request")
    return {
        "synthetic_only": True,
        "condition": condition.value,
        "provider_calls": fake.call_count,
        "replay_provider_calls": 0,
        "run_directory": str(first.run_directory),
        "replay_directory": str(replay.run_directory),
    }


def _ollama_smoke(output_root: Path) -> dict:
    model = "qwen3.5:4b-q4_K_M"
    digest = "0" * 64
    chat_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_calls
        details = {
            "parent_model": "",
            "format": "gguf",
            "family": "qwen35",
            "families": ["qwen35"],
            "parameter_size": "4.7B",
            "quantization_level": "Q4_K_M",
        }
        model_record = {
            "name": model,
            "model": model,
            "modified_at": "2026-07-13T10:00:00Z",
            "size": 100,
            "digest": digest,
            "details": details,
        }
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "mock-1.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [model_record]})
        if request.url.path == "/api/chat":
            chat_calls += 1
            return httpx.Response(
                200,
                json={
                    "model": model,
                    "created_at": "2026-07-13T10:01:00Z",
                    "message": {
                        "role": "assistant",
                        "content": '{"label":"UNKNOWN"}',
                        "thinking": "",
                    },
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": 1_000_000,
                    "load_duration": 100_000,
                    "prompt_eval_count": 10,
                    "prompt_eval_duration": 400_000,
                    "eval_count": 3,
                    "eval_duration": 500_000,
                },
            )
        if request.url.path == "/api/ps":
            running_model = {
                name: model_record[name] for name in ("name", "model", "size", "digest", "details")
            }
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            **running_model,
                            "expires_at": "2026-07-13T10:31:00Z",
                            "size_vram": 0,
                            "context_length": 4096,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    config = ProviderConfig(
        name="ollama",
        api_family="native_chat",
        endpoint="http://127.0.0.1:11434",
        model=model,
        model_digest=digest,
        provider_version="mock-1.0",
        reasoning_effort="none",
        temperature=0,
        sampling_seed=20260713,
        context_tokens=4096,
        think=False,
        keep_alive="1m",
        execution_device="cpu",
        max_output_tokens=64,
        concurrency=1,
    )
    root = repository_root(Path.cwd())
    cache = ResponseCache(output_root / f"cache-{uuid4().hex}")
    provider = OllamaChatProvider(config=config, transport=httpx.MockTransport(handler))
    template = load_prompt_template(root / "prompts/baselines/direct/v1.md", version="v1")
    schema_hash = load_schema_hash(root / "schemas/llm-baseline-output.v1.schema.json")
    predictor_kwargs = {
        "condition": BaselineCondition.DIRECT,
        "cache": cache,
        "template": template,
        "configured_model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "max_output_tokens": config.max_output_tokens,
        "output_schema_hash": schema_hash,
        "selection_manifest_hash": "1" * 64,
        "provider_name": config.name,
        "api_family": config.api_family,
        "endpoint_identity": config.endpoint,
        "provider_version": config.provider_version,
        "model_digest": config.model_digest,
        "model_options": config.request_options(),
    }
    live = LLMBaselinePredictor(
        provider=CachedProvider(provider, cache),
        **predictor_kwargs,
    ).predict(_synthetic_input("synthetic/ollama-smoke/1"), run_id="ollama-smoke")
    calls_before_replay = chat_calls
    replay = LLMBaselinePredictor(
        provider=CachedProvider(None, cache, replay_only=True),
        **predictor_kwargs,
    ).predict(_synthetic_input("synthetic/ollama-smoke/1"), run_id="ollama-smoke-replay")
    if not replay.cache_hit or chat_calls != calls_before_replay:
        raise AssertionError("Mocked Ollama replay made an inference request")
    return {
        "synthetic_only": True,
        "mocked_local_http": True,
        "structured_label": live.predicted_label.value,
        "inference_http_calls": chat_calls,
        "replay_inference_http_calls": chat_calls - calls_before_replay,
        "cache_hit_on_replay": replay.cache_hit,
        "thinking_persisted": False,
    }


def _validate_pair(direct: PreparedBaseline, few: PreparedBaseline) -> None:
    if direct.config.condition is not BaselineCondition.DIRECT:
        raise ValueError("--direct-config is not a direct baseline")
    if few.config.condition is not BaselineCondition.FEW_SHOT:
        raise ValueError("--few-shot-config is not a few-shot baseline")
    for field in ("provider", "dataset", "pricing", "seed", "predictor_version"):
        if getattr(direct.config, field) != getattr(few.config, field):
            raise ValueError(f"Frozen direct/few-shot {field} settings do not match")
    if direct.template.text != few.template.text:
        raise ValueError("Direct and few-shot task definitions differ")
    if [item.example_id for item in direct.pilot_examples] != [
        item.example_id for item in few.pilot_examples
    ]:
        raise ValueError("Direct and few-shot pilot example IDs differ")


def _canary(args) -> dict:
    direct = prepare_baseline(args.direct_config)
    few = prepare_baseline(args.few_shot_config)
    _validate_pair(direct, few)
    cache_root = resolve_repository_path(direct.repository_root, direct.config.run.cache_directory)
    cache = ResponseCache(cache_root)
    synthetic = _synthetic_input()
    placeholder = DeterministicFakeProvider()
    direct_predictor = build_predictor(direct, placeholder, cache)
    few_predictor = build_predictor(few, placeholder, cache)
    requests = [
        direct_predictor.request_for(synthetic),
        few_predictor.request_for(synthetic),
    ]
    new_requests = [request for request in requests if cache.read(request) is None]
    pricing = direct.config.pricing
    worst_cost = (
        sum(worst_case_request_cost_usd(request, pricing) for request in new_requests)
        if pricing is not None
        else 0.0
    )
    budget = None
    if direct.config.provider.name == "openai":
        _validate_pricing_is_current(direct)
        _validate_pricing_is_current(few)
        cap = _validate_live_gate(args, worst_cost)
        budget = BudgetGuard(cap)
    providers = [_live_cached_provider(prepared, cache, budget) for prepared in (direct, few)]
    predictors = [
        build_predictor(prepared, provider, cache)
        for prepared, provider in zip((direct, few), providers, strict=True)
    ]
    records = [
        predictor.predict(synthetic, run_id=f"canary-{index + 1}")
        for index, predictor in enumerate(predictors)
    ]
    replay_records = [
        build_predictor(prepared, CachedProvider(None, cache, replay_only=True), cache).predict(
            synthetic, run_id=f"canary-replay-{index + 1}"
        )
        for index, prepared in enumerate((direct, few))
    ]
    if not all(item.cache_hit for item in replay_records):
        raise AssertionError("Canary replay did not use the cache")
    rendered_hashes = [request.rendered_request_hash for request in requests]
    if len(set(rendered_hashes)) != 2:
        raise AssertionError("Direct and few-shot canary requests rendered identically")
    demonstration_count = requests[1].instructions.count('<demonstration index="')
    if demonstration_count != 6:
        raise AssertionError("Few-shot canary did not contain exactly six demonstrations")
    if "gold_label" in type(synthetic).model_fields:
        raise AssertionError("Prediction input unexpectedly exposes development gold labels")
    return {
        "synthetic_requests_planned": len(new_requests),
        "synthetic_requests_completed": sum(not item.cache_hit for item in records),
        "replay_network_requests": 0,
        "structured_labels": [item.predicted_label.value for item in records],
        "provider_request_ids_recorded": [bool(item.provider_request_id) for item in records],
        "configured_models": [item.configured_model for item in records],
        "returned_models": [item.returned_model for item in records],
        "token_usage_recorded": [item.total_tokens for item in records],
        "cache_root": direct.config.run.cache_directory,
        "estimated_cost_usd": sum(item.estimated_cost_usd or 0 for item in records),
        "provider_versions": [item.provider_version for item in records],
        "model_digests": [item.model_digest for item in records],
        "execution_devices": [item.execution_device for item in records],
        "rendered_request_hashes": rendered_hashes,
        "rendered_request_hashes_differ": True,
        "few_shot_demonstration_count": demonstration_count,
        "development_gold_fields_present": False,
        "remote_provider_contacted": False,
        "thinking_persisted": False,
        "provider_total_duration_ms": [item.provider_total_duration_ms for item in records],
        "provider_load_duration_ms": [item.provider_load_duration_ms for item in records],
        "provider_prompt_eval_duration_ms": [
            item.provider_prompt_eval_duration_ms for item in records
        ],
        "provider_generation_duration_ms": [
            item.provider_generation_duration_ms for item in records
        ],
        "generation_tokens_per_second": [item.generation_tokens_per_second for item in records],
    }


def _canary_plan(args) -> dict:
    direct = prepare_baseline(args.direct_config)
    few = prepare_baseline(args.few_shot_config)
    _validate_pair(direct, few)
    cache_root = resolve_repository_path(direct.repository_root, direct.config.run.cache_directory)
    cache = ResponseCache(cache_root)
    synthetic = _synthetic_input()
    requests = [
        build_predictor(prepared, DeterministicFakeProvider(), cache).request_for(synthetic)
        for prepared in (direct, few)
    ]
    conditions = []
    for prepared, request in zip((direct, few), requests, strict=True):
        cache_hit = cache.read(request) is not None
        conditions.append(
            {
                "condition": prepared.config.condition.value,
                "cache_hit": cache_hit,
                "estimated_input_tokens": estimate_request_input_tokens(request),
                "maximum_output_tokens": request.max_output_tokens,
                "estimated_worst_case_usd": (
                    worst_case_request_cost_usd(request, prepared.config.pricing)
                    if not cache_hit and prepared.config.pricing is not None
                    else 0.0
                ),
            }
        )
    return {
        "synthetic_canary_requests": len(requests),
        "new_provider_requests": sum(not item["cache_hit"] for item in conditions),
        "new_billable_requests": (
            sum(not item["cache_hit"] for item in conditions)
            if direct.config.pricing is not None
            else 0
        ),
        "estimated_input_tokens": sum(item["estimated_input_tokens"] for item in conditions),
        "estimated_worst_case_output_tokens": sum(
            item["maximum_output_tokens"] for item in conditions if not item["cache_hit"]
        ),
        "estimated_worst_case_usd": sum(item["estimated_worst_case_usd"] for item in conditions),
        "conditions": conditions,
        "rendered_request_hashes": [item.rendered_request_hash for item in requests],
        "rendered_request_hashes_differ": (
            requests[0].rendered_request_hash != requests[1].rendered_request_hash
        ),
        "few_shot_demonstration_count": requests[1].instructions.count('<demonstration index="'),
        "development_gold_fields_present": "gold_label" in type(synthetic).model_fields,
        "pricing_source": (
            direct.config.pricing.source_url if direct.config.pricing is not None else None
        ),
        "pricing_as_of": (
            direct.config.pricing.as_of.isoformat() if direct.config.pricing is not None else None
        ),
        "cache_root": direct.config.run.cache_directory,
        "external_data_description": (
            "One synthetic context/query in both conditions; the few-shot condition also "
            "includes the six frozen ProofWriter training demonstrations"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            prepared = prepare_baseline(args.config)
            report, _ = plan_baseline(prepared)
            output = report.model_dump(mode="json")
        elif args.command == "run":
            output = {"run_directory": str(_run_baseline(args))}
        elif args.command == "smoke":
            output = _smoke(BaselineCondition(args.condition), args.output_root.resolve())
        elif args.command == "ollama-smoke":
            output = _ollama_smoke(args.output_root.resolve())
        elif args.command == "compare":
            comparison = compare_runs(args.direct_run, args.few_shot_run, args.selection_manifest)
            write_comparison(args.output, comparison)
            output = {
                "comparison": comparison.model_dump(mode="json"),
                "output": str(args.output),
            }
        elif args.command == "canary":
            output = _canary(args)
        elif args.command == "canary-plan":
            output = _canary_plan(args)
        else:  # pragma: no cover - argparse requires a known command.
            raise AssertionError(args.command)
    except (
        DatasetError,
        CacheError,
        ExistingRunError,
        LiveApprovalError,
        OSError,
        ValidationError,
        ValueError,
        ProviderError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0
