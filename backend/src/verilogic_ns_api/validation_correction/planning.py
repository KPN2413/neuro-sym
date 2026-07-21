from __future__ import annotations

from verilogic_ns_api.semantic_parsing.views import prepare_query_view
from verilogic_ns_api.validation_correction.configuration import PreparedCorrectionExperiment
from verilogic_ns_api.validation_correction.feedback import (
    validate_query_candidate,
    validate_theory_candidate,
)
from verilogic_ns_api.validation_correction.raw import load_raw_phase5_candidates


def build_correction_plan(
    prepared: PreparedCorrectionExperiment, *, calibration: bool = False
) -> dict[str, object]:
    examples = (
        prepared.phase5.calibration_examples if calibration else prepared.phase5.pilot_examples
    )
    raw = load_raw_phase5_candidates(prepared.phase5, calibration=calibration)
    bodies = {}
    theory_valid = 0
    for key, view in raw.theory_views.items():
        result = validate_theory_candidate(raw.theories[key], view, theory_id=key)
        if result.valid and result.converted is not None:
            theory_valid += 1
            bodies[key] = result.converted
    query_valid = 0
    for example in examples:
        key = example.theory_id or example.example_id
        result = validate_query_candidate(
            raw.queries[example.example_id],
            prepare_query_view(example),
            body=bodies.get(key),
        )
        query_valid += int(result.valid)
    total = len(raw.theories) + len(raw.queries)
    valid = theory_valid + query_valid
    invalid = total - valid
    maximum_critic_calls = 2 * valid + invalid
    maximum_correction_calls = total
    maximum_new_calls = maximum_critic_calls + maximum_correction_calls
    if maximum_new_calls > prepared.config.limits.maximum_new_pilot_calls:
        raise ValueError("Phase 6 worst-case call plan exceeds the frozen local-call budget")
    average_phase5_call_seconds = 10_903_745.7369 / 57 / 1000
    return {
        "dataset": "calibration" if calibration else "pilot",
        "examples": len(examples),
        "raw_phase5_cache_hits": raw.cache_hits,
        "theory_components": len(raw.theories),
        "query_components": len(raw.queries),
        "raw_valid_components": valid,
        "raw_invalid_components": invalid,
        "maximum_critic_calls": maximum_critic_calls,
        "maximum_correction_calls": maximum_correction_calls,
        "absolute_maximum_new_local_calls": maximum_new_calls,
        "frozen_call_budget": prepared.config.limits.maximum_new_pilot_calls,
        "estimated_worst_case_runtime_hours": maximum_new_calls
        * average_phase5_call_seconds
        / 3600,
        "estimated_cache_impact_mib_upper_bound": maximum_new_calls * 2,
        "model": prepared.config.runtime.model,
        "model_digest": prepared.config.runtime.model_digest,
        "endpoint": prepared.config.runtime.endpoint,
        "concurrency": 1,
        "hosted_calls": 0,
        "api_cost_usd": 0,
        "test_split": False,
    }
