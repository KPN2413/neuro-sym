from __future__ import annotations

from datetime import UTC, datetime

from pydantic import JsonValue

from verilogic_ns_api.baselines.cache import ResponseCache
from verilogic_ns_api.baselines.models import BaselineCondition, ProviderStatus
from verilogic_ns_api.baselines.prompts import (
    Demonstration,
    PromptTemplate,
    build_request,
)
from verilogic_ns_api.baselines.provider import LLMProvider
from verilogic_ns_api.research.models import (
    PredictionInput,
    PredictionLabel,
    PredictionRecord,
)


class LLMBaselinePredictor:
    version = "1.0"

    def __init__(
        self,
        *,
        condition: BaselineCondition,
        provider: LLMProvider,
        cache: ResponseCache,
        template: PromptTemplate,
        configured_model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        output_schema_hash: str,
        demonstrations: list[Demonstration] | None = None,
        demonstration_manifest_hash: str | None = None,
        selection_manifest_hash: str | None = None,
        provider_name: str = "openai",
        api_family: str = "responses",
        endpoint_identity: str | None = None,
        provider_version: str | None = None,
        model_digest: str | None = None,
        model_options: dict[str, JsonValue] | None = None,
    ) -> None:
        self.condition = condition
        self.name = f"{condition.value.replace('_', '-')}-llm"
        self.provider = provider
        self.cache = cache
        self.template = template
        self.configured_model = configured_model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.output_schema_hash = output_schema_hash
        self.demonstrations = demonstrations or []
        self.demonstration_manifest_hash = demonstration_manifest_hash
        self.selection_manifest_hash = selection_manifest_hash
        self.provider_name = provider_name
        self.api_family = api_family
        self.endpoint_identity = endpoint_identity
        self.provider_version = provider_version
        self.model_digest = model_digest
        self.model_options = model_options or {}
        if condition is BaselineCondition.DIRECT and self.demonstrations:
            raise ValueError("Direct predictor cannot contain demonstrations")
        if condition is BaselineCondition.FEW_SHOT and len(self.demonstrations) != 6:
            raise ValueError("Few-shot predictor requires exactly six demonstrations")

    def request_for(self, example: PredictionInput):
        return build_request(
            template=self.template,
            example=example,
            configured_model=self.configured_model,
            reasoning_effort=self.reasoning_effort,
            max_output_tokens=self.max_output_tokens,
            output_schema_hash=self.output_schema_hash,
            demonstration_manifest_hash=self.demonstration_manifest_hash,
            selection_manifest_hash=self.selection_manifest_hash,
            demonstrations=self.demonstrations,
            provider=self.provider_name,
            api_family=self.api_family,
            endpoint_identity=self.endpoint_identity,
            provider_version=self.provider_version,
            model_digest=self.model_digest,
            model_options=self.model_options,
        )

    def predict(self, example: PredictionInput, *, run_id: str) -> PredictionRecord:
        request = self.request_for(example)
        response = self.provider.complete(request)
        if response.status is ProviderStatus.REFUSAL:
            predicted_label = PredictionLabel.ABSTAIN
            abstention_reason = "provider_refusal"
        else:
            if response.label is None:  # Defensive: LLMResponse validation already enforces this.
                raise ValueError("Provider success response did not contain a label")
            predicted_label = PredictionLabel(response.label.value)
            abstention_reason = None
        return PredictionRecord(
            run_id=run_id,
            example_id=example.example_id,
            predicted_label=predicted_label,
            abstention_reason=abstention_reason,
            latency_ms=response.latency_ms,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            cached_input_tokens=response.usage.cached_input_tokens,
            total_tokens=response.usage.total_tokens,
            raw_output_reference=self.cache.relative_reference(request),
            provider_request_id=response.provider_request_id,
            configured_model=response.configured_model,
            returned_model=response.returned_model,
            provider_version=response.provider_version,
            model_digest=response.model_digest,
            execution_device=response.execution_device,
            provider_total_duration_ms=response.provider_timing.total_duration_ms,
            provider_load_duration_ms=response.provider_timing.load_duration_ms,
            provider_prompt_eval_duration_ms=response.provider_timing.prompt_eval_duration_ms,
            provider_generation_duration_ms=response.provider_timing.generation_duration_ms,
            generation_tokens_per_second=(response.provider_timing.generation_tokens_per_second),
            request_hash=request.cache_key,
            prompt_hash=request.prompt_hash,
            retry_count=response.retry_count,
            cache_hit=response.cache_hit,
            provider_status=response.status.value,
            estimated_cost_usd=response.estimated_cost_usd,
            predictor_name=self.name,
            predictor_version=self.version,
            timestamp=response.completed_at.astimezone(UTC)
            if response.completed_at.tzinfo
            else datetime.now(UTC),
        )


class DirectLLMPredictor(LLMBaselinePredictor):
    def __init__(self, **kwargs) -> None:
        super().__init__(condition=BaselineCondition.DIRECT, **kwargs)


class FewShotLLMPredictor(LLMBaselinePredictor):
    def __init__(self, **kwargs) -> None:
        super().__init__(condition=BaselineCondition.FEW_SHOT, **kwargs)
