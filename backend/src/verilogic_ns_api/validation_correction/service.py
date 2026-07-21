from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from verilogic_ns_api.reasoning.models import sha256_payload
from verilogic_ns_api.semantic_parsing.models import CandidateQueryOutput, CandidateTheoryOutput
from verilogic_ns_api.semantic_parsing.provider import (
    ParserConfigurationError,
    ParserProviderError,
    ParserStructuredOutputError,
    ParserTimeoutError,
    ParserTransientError,
)
from verilogic_ns_api.validation_correction.cache import (
    CorrectionCacheError,
    CorrectionResponseCache,
)
from verilogic_ns_api.validation_correction.models import (
    CorrectionExperimentConfig,
    QueryCorrectionInput,
    QueryCriticInput,
    QueryCriticReport,
    TaskKind,
    TaskOutcome,
    TaskStatus,
    TheoryCorrectionInput,
    TheoryCriticInput,
    TheoryCriticReport,
)
from verilogic_ns_api.validation_correction.prompts import (
    render_correction_input,
    render_critic_input,
)
from verilogic_ns_api.validation_correction.provider import (
    CorrectionTaskRequest,
    OllamaCorrectionProvider,
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class TaskExecution(Generic[T]):
    outcome: TaskOutcome
    value: T | None


class CorrectionTaskService:
    def __init__(
        self,
        *,
        config: CorrectionExperimentConfig,
        prompts: dict[TaskKind, tuple[str, str]],
        cache: CorrectionResponseCache,
        provider: OllamaCorrectionProvider | None,
        replay_only: bool = False,
    ) -> None:
        self.config = config
        self.prompts = prompts
        self.cache = cache
        self.provider = provider
        self.replay_only = replay_only
        self.new_call_count = 0

    def critique_theory(self, value: TheoryCriticInput) -> TaskExecution[TheoryCriticReport]:
        return self._execute(
            kind=TaskKind.CRITIC_THEORY,
            value=value,
            rendered=render_critic_input(value),
            output_model=TheoryCriticReport,
        )

    def critique_query(self, value: QueryCriticInput) -> TaskExecution[QueryCriticReport]:
        return self._execute(
            kind=TaskKind.CRITIC_QUERY,
            value=value,
            rendered=render_critic_input(value),
            output_model=QueryCriticReport,
        )

    def correct_theory(self, value: TheoryCorrectionInput) -> TaskExecution[CandidateTheoryOutput]:
        return self._execute(
            kind=TaskKind.CORRECTION_THEORY,
            value=value,
            rendered=render_correction_input(value),
            output_model=CandidateTheoryOutput,
        )

    def correct_query(self, value: QueryCorrectionInput) -> TaskExecution[CandidateQueryOutput]:
        return self._execute(
            kind=TaskKind.CORRECTION_QUERY,
            value=value,
            rendered=render_correction_input(value),
            output_model=CandidateQueryOutput,
        )

    def _execute(
        self,
        *,
        kind: TaskKind,
        value: BaseModel,
        rendered: str,
        output_model: type[T],
    ) -> TaskExecution[T]:
        if len(rendered) > self.config.limits.maximum_request_characters:
            return _failure(
                kind, "0" * 64, TaskStatus.PROVIDER_ERROR, ValueError("request too large")
            )
        prompt, prompt_hash = self.prompts[kind]
        schema = output_model.model_json_schema()
        request = CorrectionTaskRequest(
            task_kind=kind,
            instructions=prompt,
            input_text=rendered,
            prompt_hash=prompt_hash,
            input_hash=sha256_payload(value.model_dump(mode="json")),
            output_schema=schema,
            schema_hash=sha256_payload(schema),
            num_predict=self._num_predict(kind),
            config=self.config.runtime,
        )
        try:
            response = self.cache.load(request)
        except CorrectionCacheError as error:
            return _failure(kind, request.request_hash, TaskStatus.PROVIDER_ERROR, error)
        cache_hit = response is not None
        if response is None:
            if self.replay_only or self.provider is None:
                return _failure(
                    kind,
                    request.request_hash,
                    TaskStatus.PROVIDER_ERROR,
                    RuntimeError("cache miss in correction replay-only mode"),
                )
            if self.new_call_count >= self.config.limits.maximum_new_pilot_calls:
                return _failure(
                    kind,
                    request.request_hash,
                    TaskStatus.RESOURCE_LIMIT,
                    RuntimeError("frozen Phase 6 local-call budget reached"),
                )
            self.new_call_count += 1
            dispatched = self._dispatch(request)
            if isinstance(dispatched, TaskExecution):
                return dispatched
            response = dispatched
            self.cache.store(request, response)
        try:
            parsed = output_model.model_validate(response.content)
        except ValidationError as error:
            return _failure(
                kind,
                request.request_hash,
                TaskStatus.STRUCTURED_OUTPUT_ERROR,
                error,
                cache_hit=cache_hit,
            )
        return TaskExecution(
            outcome=TaskOutcome(
                task_kind=kind,
                request_hash=request.request_hash,
                status=TaskStatus.SUCCESS,
                cache_hit=cache_hit,
                output=parsed.model_dump(mode="json"),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                duration_ms=response.timing.total_duration_ms,
            ),
            value=parsed,
        )

    def _dispatch(self, request: CorrectionTaskRequest) -> object:
        assert self.provider is not None
        for attempt in range(self.config.runtime.max_attempts):
            try:
                return self.provider.complete(request)
            except ParserTimeoutError as error:
                if attempt + 1 == self.config.runtime.max_attempts:
                    return _failure(
                        request.task_kind, request.request_hash, TaskStatus.TIMEOUT, error
                    )
            except ParserTransientError as error:
                if attempt + 1 == self.config.runtime.max_attempts:
                    return _failure(
                        request.task_kind, request.request_hash, TaskStatus.PROVIDER_ERROR, error
                    )
            except ParserStructuredOutputError as error:
                return _failure(
                    request.task_kind,
                    request.request_hash,
                    TaskStatus.STRUCTURED_OUTPUT_ERROR,
                    error,
                )
            except (ParserConfigurationError, ParserProviderError) as error:
                return _failure(
                    request.task_kind, request.request_hash, TaskStatus.PROVIDER_ERROR, error
                )
            time.sleep(0.25 * (2**attempt))
        raise AssertionError("unreachable correction retry state")

    def _num_predict(self, kind: TaskKind) -> int:
        limits = self.config.limits
        return {
            TaskKind.CRITIC_THEORY: limits.critic_theory_num_predict,
            TaskKind.CRITIC_QUERY: limits.critic_query_num_predict,
            TaskKind.CORRECTION_THEORY: limits.correction_theory_num_predict,
            TaskKind.CORRECTION_QUERY: limits.correction_query_num_predict,
        }[kind]


def _failure(
    kind: TaskKind,
    request_hash: str,
    status: TaskStatus,
    error: Exception,
    *,
    cache_hit: bool = False,
) -> TaskExecution:
    return TaskExecution(
        outcome=TaskOutcome(
            task_kind=kind,
            request_hash=request_hash,
            status=status,
            cache_hit=cache_hit,
            error_type=type(error).__name__,
            error_message=str(error)[:500],
        ),
        value=None,
    )
