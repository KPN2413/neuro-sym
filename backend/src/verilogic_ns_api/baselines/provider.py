from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from verilogic_ns_api.baselines.cache import ReplayCacheMiss, ResponseCache
from verilogic_ns_api.baselines.models import (
    LLMRequest,
    LLMResponse,
    PricingConfig,
    ProviderStatus,
    RetryConfig,
    UsageTelemetry,
)
from verilogic_ns_api.evaluation.protocol import FatalPredictorError
from verilogic_ns_api.research.models import GoldLabel


class ProviderError(RuntimeError):
    """Base class for provider failures with no secret-bearing message."""


class TransientProviderError(ProviderError):
    pass


class ProviderExhaustedError(ProviderError):
    pass


class InvalidProviderResponseError(ProviderError):
    pass


class AuthenticationProviderError(FatalPredictorError, ProviderError):
    pass


class ProviderConfigurationError(FatalPredictorError, ProviderError):
    pass


class CircuitBreakerOpen(FatalPredictorError, ProviderError):
    pass


class BudgetExceededError(FatalPredictorError, ProviderError):
    pass


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one strict label/refusal response or raise a typed failure."""
        ...


def estimate_tokens(text: str) -> int:
    """Conservative provider-independent character heuristic for preflight only."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def estimate_request_input_tokens(request: LLMRequest) -> int:
    return estimate_tokens(request.instructions) + estimate_tokens(request.input_text)


def usage_cost_usd(usage: UsageTelemetry, pricing: PricingConfig) -> float:
    uncached_input = max(0, usage.input_tokens - usage.cached_input_tokens)
    long_context = usage.input_tokens > pricing.long_context_threshold_tokens
    input_multiplier = pricing.long_context_input_multiplier if long_context else 1.0
    output_multiplier = pricing.long_context_output_multiplier if long_context else 1.0
    return (
        uncached_input * pricing.input_usd_per_million * input_multiplier
        + usage.cached_input_tokens * pricing.cached_input_usd_per_million * input_multiplier
        + usage.output_tokens * pricing.output_usd_per_million * output_multiplier
    ) / 1_000_000


def worst_case_request_cost_usd(request: LLMRequest, pricing: PricingConfig) -> float:
    estimated_input = estimate_request_input_tokens(request)
    long_context = estimated_input > pricing.long_context_threshold_tokens
    input_multiplier = pricing.long_context_input_multiplier if long_context else 1.0
    output_multiplier = pricing.long_context_output_multiplier if long_context else 1.0
    return (
        estimated_input * pricing.input_usd_per_million * input_multiplier
        + request.max_output_tokens * pricing.output_usd_per_million * output_multiplier
    ) / 1_000_000


class BudgetGuard:
    def __init__(self, max_cost_usd: float) -> None:
        if max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        self.max_cost_usd = max_cost_usd
        self._committed = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()

    @property
    def committed_cost_usd(self) -> float:
        with self._lock:
            return self._committed

    def reserve(self, worst_case_usd: float) -> float:
        with self._lock:
            if self._committed + self._reserved + worst_case_usd > self.max_cost_usd:
                raise BudgetExceededError("Cost cap would be exceeded before dispatch")
            self._reserved += worst_case_usd
        return worst_case_usd

    def settle(self, reservation: float, actual_usd: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation)
            self._committed += actual_usd
            if self._committed > self.max_cost_usd:
                raise BudgetExceededError("Observed provider usage exceeded the cost cap")


class RetryingProvider:
    def __init__(
        self,
        inner: LLMProvider,
        config: RetryConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.inner = inner
        self.config = config
        self.name = inner.name
        self._sleep = sleep
        self._random_uniform = random_uniform

    def complete(self, request: LLMRequest) -> LLMResponse:
        for attempt in range(self.config.max_attempts):
            try:
                response = self.inner.complete(request)
                return response.model_copy(update={"retry_count": attempt})
            except TransientProviderError as error:
                if attempt + 1 >= self.config.max_attempts:
                    raise ProviderExhaustedError(
                        f"Transient provider failure after {self.config.max_attempts} attempts"
                    ) from error
                delay = min(
                    self.config.max_delay_seconds,
                    self.config.base_delay_seconds * (2**attempt),
                )
                delay += self._random_uniform(0, self.config.jitter_seconds)
                self._sleep(delay)
        raise AssertionError("retry loop did not return or raise")


class CircuitBreakingProvider:
    def __init__(self, inner: LLMProvider, *, threshold: int) -> None:
        self.inner = inner
        self.name = inner.name
        self.threshold = threshold
        self._consecutive_failures = 0
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            if self._consecutive_failures >= self.threshold:
                raise CircuitBreakerOpen("Provider circuit breaker is open")
        try:
            response = self.inner.complete(request)
        except (ProviderExhaustedError, InvalidProviderResponseError) as error:
            with self._lock:
                self._consecutive_failures += 1
                should_open = self._consecutive_failures >= self.threshold
            if should_open:
                raise CircuitBreakerOpen(
                    "Repeated systemic provider failures opened the circuit breaker"
                ) from error
            raise
        else:
            with self._lock:
                self._consecutive_failures = 0
            return response


class CachedProvider:
    def __init__(
        self,
        inner: LLMProvider | None,
        cache: ResponseCache,
        *,
        replay_only: bool = False,
        budget: BudgetGuard | None = None,
        pricing: PricingConfig | None = None,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.replay_only = replay_only
        self.budget = budget
        self.pricing = pricing
        self.name = inner.name if inner is not None else "cache-replay"

    def complete(self, request: LLMRequest) -> LLMResponse:
        cached = self.cache.read(request)
        if cached is not None:
            return cached
        if self.replay_only:
            raise ReplayCacheMiss(f"No cached response for {request.example_id}")
        if self.inner is None:
            raise ProviderConfigurationError("No live provider configured for cache miss")

        with self.cache.lock(request):
            cached = self.cache.read(request)
            if cached is not None:
                return cached
            reservation = 0.0
            if self.budget is not None:
                if self.pricing is None:
                    raise ProviderConfigurationError("Budgeting requires pricing metadata")
                reservation = self.budget.reserve(
                    worst_case_request_cost_usd(request, self.pricing)
                )
            try:
                response = self.inner.complete(request)
            except Exception:
                # Ambiguous transport failures may still be billed, so retain the reservation.
                raise
            if self.budget is not None and self.pricing is not None:
                actual_cost = usage_cost_usd(response.usage, self.pricing)
                response = response.model_copy(update={"estimated_cost_usd": actual_cost})
                try:
                    self.budget.settle(reservation, actual_cost)
                except BudgetExceededError:
                    # Keep a paid success even if observed tokenization exceeds
                    # the conservative estimate used before dispatch.
                    self.cache.write(request, response)
                    raise
            self.cache.write(request, response)
            return response


class DeterministicFakeProvider:
    """Network-free plumbing provider; never represents model performance."""

    name = "fake"

    def __init__(
        self,
        *,
        label: GoldLabel = GoldLabel.UNKNOWN,
        refusal: bool = False,
    ) -> None:
        self.label = label
        self.refusal = refusal
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        now = datetime.now(UTC)
        if self.refusal:
            return LLMResponse(
                request_hash=request.cache_key,
                status=ProviderStatus.REFUSAL,
                refusal_reason="synthetic_refusal",
                configured_model=request.configured_model,
                returned_model="fake-model",
                started_at=now,
                completed_at=now,
                latency_ms=0,
            )
        return LLMResponse(
            request_hash=request.cache_key,
            status=ProviderStatus.SUCCESS,
            label=self.label,
            configured_model=request.configured_model,
            returned_model="fake-model",
            usage=UsageTelemetry(input_tokens=1, output_tokens=1, total_tokens=2),
            started_at=now,
            completed_at=now,
            latency_ms=0,
        )
