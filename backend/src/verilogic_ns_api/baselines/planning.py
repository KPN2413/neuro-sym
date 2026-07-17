from __future__ import annotations

from verilogic_ns_api.baselines.cache import ResponseCache
from verilogic_ns_api.baselines.configuration import PreparedBaseline, resolve_repository_path
from verilogic_ns_api.baselines.models import PlanReport
from verilogic_ns_api.baselines.predictors import LLMBaselinePredictor
from verilogic_ns_api.baselines.provider import (
    DeterministicFakeProvider,
    LLMProvider,
    estimate_request_input_tokens,
    worst_case_request_cost_usd,
)


def build_predictor(
    prepared: PreparedBaseline, provider: LLMProvider, cache: ResponseCache
) -> LLMBaselinePredictor:
    return LLMBaselinePredictor(
        condition=prepared.config.condition,
        provider=provider,
        cache=cache,
        template=prepared.template,
        configured_model=prepared.config.provider.model,
        reasoning_effort=prepared.config.provider.reasoning_effort,
        max_output_tokens=prepared.config.provider.max_output_tokens,
        output_schema_hash=prepared.output_schema_hash,
        selection_manifest_hash=prepared.pilot_manifest.manifest_hash,
        provider_name=prepared.config.provider.name,
        api_family=prepared.config.provider.api_family,
        endpoint_identity=prepared.config.provider.endpoint,
        provider_version=prepared.config.provider.provider_version,
        model_digest=prepared.config.provider.model_digest,
        model_options=prepared.config.provider.request_options(),
        demonstrations=prepared.demonstrations,
        demonstration_manifest_hash=(
            prepared.demonstration_manifest.manifest_hash
            if prepared.demonstration_manifest is not None
            else None
        ),
    )


def plan_baseline(prepared: PreparedBaseline) -> tuple[PlanReport, ResponseCache]:
    cache = ResponseCache(
        resolve_repository_path(prepared.repository_root, prepared.config.run.cache_directory)
    )
    predictor = build_predictor(prepared, provider=DeterministicFakeProvider(), cache=cache)
    requests = [predictor.request_for(item.for_prediction()) for item in prepared.pilot_examples]
    uncached = [request for request in requests if cache.read(request) is None]
    estimated_input_tokens = sum(estimate_request_input_tokens(item) for item in uncached)
    worst_output_tokens = len(uncached) * prepared.config.provider.max_output_tokens
    pricing = prepared.config.pricing
    worst_cost = (
        sum(worst_case_request_cost_usd(item, pricing) for item in uncached)
        if pricing is not None
        else 0.0
    )
    report = PlanReport(
        condition=prepared.config.condition,
        provider=prepared.config.provider.name,
        api_family=prepared.config.provider.api_family,
        configured_model=prepared.config.provider.model,
        reasoning_effort=prepared.config.provider.reasoning_effort,
        planned_requests=len(requests),
        cache_hits=len(requests) - len(uncached),
        new_provider_requests=len(uncached),
        new_billable_requests=(len(uncached) if pricing is not None else 0),
        estimated_input_tokens=estimated_input_tokens,
        maximum_output_tokens_per_request=prepared.config.provider.max_output_tokens,
        estimated_worst_case_output_tokens=worst_output_tokens,
        pricing_source=pricing.source_url if pricing is not None else None,
        pricing_as_of=pricing.as_of if pricing is not None else None,
        estimated_worst_case_usd=worst_cost,
        endpoint_identity=prepared.config.provider.endpoint,
        provider_version=prepared.config.provider.provider_version,
        model_digest=prepared.config.provider.model_digest,
        model_options=prepared.config.provider.request_options(),
        dataset_version=prepared.config.dataset.version,
        dataset_variant=f"OWA/{prepared.config.dataset.variant}",
        dataset_splits=[prepared.pilot_manifest.split],
        external_data_description=(
            "Selected ProofWriter context and query text only; no gold label, proof, "
            "local path, test-split record, or credential"
        ),
        prompt_hash=prepared.template.sha256,
        output_schema_hash=prepared.output_schema_hash,
        selection_manifest_hash=prepared.pilot_manifest.manifest_hash,
        demonstration_manifest_hash=(
            prepared.demonstration_manifest.manifest_hash
            if prepared.demonstration_manifest is not None
            else None
        ),
        config_hash=prepared.config_hash,
    )
    return report, cache
