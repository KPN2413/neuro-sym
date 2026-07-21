# Phase 5 Pilot Results

## Status and protocol

The frozen local development pilot completed on 30 ProofWriter OWA development examples. It used
28 unique theory parses plus 30 query parses. One query response was an exact gold-free semantic
cache reuse from train calibration; 57 new local inference calls were made. No test data, hosted
provider, API key, external transfer, paid service, correction loop, or Phase 6 feature was used.

Cache-only replay then produced 58/58 cache hits, zero inference calls, zero input/output tokens,
zero provider duration, and identical semantic and end-to-end metrics.

## Main results

| Condition | Correct / total | Overall accuracy | Coverage | Answered-only accuracy |
|---|---:|---:|---:|---:|
| Direct local LLM (Phase 3) | 17 / 30 | 56.67% | 100% | 56.67% |
| Few-shot local LLM (Phase 3) | 15 / 30 | 50.00% | 100% | 50.00% |
| Phase 5 parser + symbolic engine | 3 / 30 | 10.00% | 13.33% | 75.00% |
| Oracle formal AST + symbolic engine (Phase 4) | 30 / 30 | 100.00% | 100% | 100.00% |

These conditions answer different engineering questions. The oracle is a symbolic ceiling using
dataset-provided formal fields; it is not a language parser. The parser pipeline's overall accuracy
includes all fail-closed parser errors. Answered-only accuracy must not be reported without its 4/30
coverage denominator.

## Parser metrics

- Theory structured-output success: 28/28 unique theories.
- Query structured-output success: 29/30 queries.
- Complete structured theory/query pairs: 29/30; source coverage passed for 19/29 (65.52%).
- Semantic validation passed for 4/19 candidates that reached that gate (21.05%).
- Complete ASTs accepted for reasoning: 4/30.
- Independent proof verification: 4/4 accepted reasoning results (100%).
- Whole-theory exact accuracy: 0/28.
- Exact-query accuracy after the complete boundary: 1/30 (3.33%).
- Source-aligned statement accuracy: 31/496 (6.25%).
- Statement semantic precision/recall/F1: 41.89% / 6.25% / 10.88%.
- Closure precision/recall/F1 on accepted complete cases: 27.42% / 23.94% / 25.56%.
- Parser error taxonomy: 15 `SEMANTIC_INVALID`, 10 `SOURCE_COVERAGE_ERROR`, and 1
  `STRUCTURED_OUTPUT_ERROR`.
- No failure was converted to `UNKNOWN`; no abstention was introduced in Phase 5.

Construction-level results are retained in the ignored replay metrics. Negative and relational
constructions were particularly weak: binary-negative facts and negative-premise rules were 0% exact
in this sample. These are descriptive pilot measurements, not significance claims.

## Efficiency and interpretation

The live pilot's 57 new local calls used 38,397 prompt tokens and 28,086 generated tokens across
unique request telemetry. Local provider duration was about 10,903,746 ms (roughly 181.7 minutes).
API cost was USD 0.00 and hosted calls were zero.

The result shows that the deterministic reasoner is reliable when given correct formal input, while
natural-language formalisation by this small local model is the bottleneck. Phase 6 may evaluate
bounded validation, correction, and abstention, but it must preserve this Phase 5 no-correction
baseline and must not tune on the frozen development results.
