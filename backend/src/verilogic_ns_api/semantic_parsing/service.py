from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from verilogic_ns_api.reasoning.models import sha256_payload
from verilogic_ns_api.semantic_parsing.cache import ParserCacheError, ParserResponseCache
from verilogic_ns_api.semantic_parsing.models import (
    CandidateQueryOutput,
    CandidateTheoryOutput,
    ParserKind,
    ParserOutcome,
    ParserRuntimeConfig,
    ParserStatus,
    QueryParseInput,
    TheoryParseInput,
)
from verilogic_ns_api.semantic_parsing.prompts import render_query_input, render_theory_input
from verilogic_ns_api.semantic_parsing.provider import (
    OllamaStructuredProvider,
    ParserConfigurationError,
    ParserProviderError,
    ParserStructuredOutputError,
    ParserTimeoutError,
    ParserTransientError,
    StructuredRequest,
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ParseExecution(Generic[T]):
    outcome: ParserOutcome
    candidate: T | None


class SemanticParser:
    def __init__(
        self,
        *,
        config: ParserRuntimeConfig,
        theory_prompt: str,
        theory_prompt_hash: str,
        query_prompt: str,
        query_prompt_hash: str,
        cache: ParserResponseCache,
        provider: OllamaStructuredProvider | None,
        replay_only: bool = False,
    ) -> None:
        self.config = config
        self.theory_prompt = theory_prompt
        self.theory_prompt_hash = theory_prompt_hash
        self.query_prompt = query_prompt
        self.query_prompt_hash = query_prompt_hash
        self.cache = cache
        self.provider = provider
        self.replay_only = replay_only

    def parse_theory(self, value: TheoryParseInput) -> ParseExecution[CandidateTheoryOutput]:
        return self._parse(
            kind=ParserKind.THEORY,
            value=value,
            instructions=self.theory_prompt,
            prompt_hash=self.theory_prompt_hash,
            rendered=render_theory_input(value),
            output_model=CandidateTheoryOutput,
        )

    def parse_query(self, value: QueryParseInput) -> ParseExecution[CandidateQueryOutput]:
        return self._parse(
            kind=ParserKind.QUERY,
            value=value,
            instructions=self.query_prompt,
            prompt_hash=self.query_prompt_hash,
            rendered=render_query_input(value),
            output_model=CandidateQueryOutput,
        )

    def _parse(
        self,
        *,
        kind: ParserKind,
        value: TheoryParseInput | QueryParseInput,
        instructions: str,
        prompt_hash: str,
        rendered: str,
        output_model: type[T],
    ) -> ParseExecution[T]:
        schema = output_model.model_json_schema()
        request = StructuredRequest(
            kind=kind.value,
            instructions=instructions,
            input_text=rendered,
            prompt_hash=prompt_hash,
            input_hash=value.input_hash,
            output_schema=schema,
            schema_hash=sha256_payload(schema),
            config=self.config,
        )
        try:
            response = self.cache.load(request)
        except ParserCacheError as error:
            return _failure(
                kind, value.input_hash, request.request_hash, ParserStatus.STRUCTURAL_INVALID, error
            )
        cache_hit = response is not None
        if response is None:
            if self.replay_only or self.provider is None:
                return _failure(
                    kind,
                    value.input_hash,
                    request.request_hash,
                    ParserStatus.PROVIDER_ERROR,
                    RuntimeError("cache miss in replay-only mode"),
                )
            response = self._dispatch(request, kind, value.input_hash)
            if isinstance(response, ParseExecution):
                return response
            self.cache.store(request, response)
        try:
            candidate = output_model.model_validate(response.content)
        except ValidationError as error:
            return _failure(
                kind,
                value.input_hash,
                request.request_hash,
                ParserStatus.STRUCTURED_OUTPUT_ERROR,
                error,
                cache_hit=cache_hit,
            )
        outcome = ParserOutcome(
            parser_kind=kind,
            input_hash=value.input_hash,
            request_hash=request.request_hash,
            status=ParserStatus.PARSED,
            cache_hit=cache_hit,
            candidate=candidate.model_dump(mode="json"),
            usage=response.usage,
            timing=response.timing,
        )
        return ParseExecution(outcome=outcome, candidate=candidate)

    def _dispatch(self, request: StructuredRequest, kind: ParserKind, input_hash: str) -> object:
        assert self.provider is not None
        for attempt in range(self.config.max_attempts):
            try:
                return self.provider.complete(request)
            except ParserTimeoutError as error:
                if attempt + 1 == self.config.max_attempts:
                    return _failure(
                        kind, input_hash, request.request_hash, ParserStatus.TIMEOUT, error
                    )
            except ParserTransientError as error:
                if attempt + 1 == self.config.max_attempts:
                    return _failure(
                        kind, input_hash, request.request_hash, ParserStatus.PROVIDER_ERROR, error
                    )
            except ParserStructuredOutputError as error:
                return _failure(
                    kind,
                    input_hash,
                    request.request_hash,
                    ParserStatus.STRUCTURED_OUTPUT_ERROR,
                    error,
                )
            except (ParserConfigurationError, ParserProviderError) as error:
                return _failure(
                    kind, input_hash, request.request_hash, ParserStatus.PROVIDER_ERROR, error
                )
            time.sleep(0.25 * (2**attempt))
        raise AssertionError("unreachable retry state")


def _failure(
    kind: ParserKind,
    input_hash: str,
    request_hash: str,
    status: ParserStatus,
    error: Exception,
    *,
    cache_hit: bool = False,
) -> ParseExecution:
    return ParseExecution(
        outcome=ParserOutcome(
            parser_kind=kind,
            input_hash=input_hash,
            request_hash=request_hash,
            status=status,
            cache_hit=cache_hit,
            error_type=type(error).__name__,
            error_message=str(error)[:1000],
        ),
        candidate=None,
    )
