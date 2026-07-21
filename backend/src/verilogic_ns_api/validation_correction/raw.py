from __future__ import annotations

from dataclasses import dataclass

from verilogic_ns_api.baselines.configuration import resolve_repository_path
from verilogic_ns_api.reasoning.models import sha256_payload
from verilogic_ns_api.semantic_parsing.cache import ParserResponseCache
from verilogic_ns_api.semantic_parsing.configuration import PreparedParserExperiment
from verilogic_ns_api.semantic_parsing.models import CandidateQueryOutput, CandidateTheoryOutput
from verilogic_ns_api.semantic_parsing.prompts import render_query_input, render_theory_input
from verilogic_ns_api.semantic_parsing.provider import StructuredRequest
from verilogic_ns_api.semantic_parsing.views import (
    PreparedTheoryView,
    assert_same_theory,
    prepare_query_view,
    prepare_theory_view,
)


class Phase5CacheMissError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawPhase5Candidates:
    theory_views: dict[str, PreparedTheoryView]
    theories: dict[str, dict[str, object]]
    queries: dict[str, dict[str, object]]
    cache_hits: int


def load_raw_phase5_candidates(
    prepared: PreparedParserExperiment,
    *,
    calibration: bool,
) -> RawPhase5Candidates:
    examples = prepared.calibration_examples if calibration else prepared.pilot_examples
    cache = ParserResponseCache(
        resolve_repository_path(prepared.root, prepared.config.cache_directory)
    )
    theory_views: dict[str, PreparedTheoryView] = {}
    for example in examples:
        key = example.theory_id or example.example_id
        view = prepare_theory_view(example)
        if key in theory_views:
            assert_same_theory(theory_views[key], view)
        else:
            theory_views[key] = view

    theories: dict[str, dict[str, object]] = {}
    for key, view in sorted(theory_views.items()):
        request = StructuredRequest(
            kind="theory",
            instructions=prepared.theory_prompt,
            input_text=render_theory_input(view.public),
            prompt_hash=prepared.config.theory_prompt_sha256,
            input_hash=view.public.input_hash,
            output_schema=CandidateTheoryOutput.model_json_schema(),
            schema_hash=sha256_payload(CandidateTheoryOutput.model_json_schema()),
            config=prepared.config.runtime,
        )
        response = cache.load(request)
        if response is None:
            raise Phase5CacheMissError(f"missing frozen Phase 5 theory cache entry for {key}")
        theories[key] = response.content

    queries: dict[str, dict[str, object]] = {}
    for example in examples:
        view = prepare_query_view(example)
        request = StructuredRequest(
            kind="query",
            instructions=prepared.query_prompt,
            input_text=render_query_input(view.public),
            prompt_hash=prepared.config.query_prompt_sha256,
            input_hash=view.public.input_hash,
            output_schema=CandidateQueryOutput.model_json_schema(),
            schema_hash=sha256_payload(CandidateQueryOutput.model_json_schema()),
            config=prepared.config.runtime,
        )
        response = cache.load(request)
        if response is None:
            raise Phase5CacheMissError(
                f"missing frozen Phase 5 query cache entry for {example.example_id}"
            )
        queries[example.example_id] = response.content
    return RawPhase5Candidates(
        theory_views=theory_views,
        theories=theories,
        queries=queries,
        cache_hits=len(theories) + len(queries),
    )
