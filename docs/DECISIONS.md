# Architecture Decisions

This is a lightweight decision log. Future decisions append new entries; they do not silently rewrite the rationale for an existing contract.

## D-001: Separate semantic parsing from reasoning

**Status:** accepted
**Decision:** An existing LLM may produce only an untrusted typed-AST candidate. A deterministic engine owns entailment.
**Reason:** This makes formal decisions inspectable and prevents free-form model prose from acting as a proof.

## D-002: Use a restricted, versioned JSON AST

**Status:** accepted
**Decision:** `schemas/theory.v1.schema.json` is the initial parser/reasoner boundary; unknown fields and executable structures are rejected.
**Reason:** JSON is provider-independent and auditable, while explicit versioning protects future compatibility.

## D-003: Use open-world semantics with explicit negation

**Status:** accepted
**Decision:** Missing support yields `UNKNOWN`; `CONTRADICTED` requires a derivation of the query's explicit opposite. Both polarities produce internal `INCONSISTENT`.
**Reason:** This matches the approved research problem and avoids treating absence as falsity.

## D-004: Require source-linked provenance

**Status:** accepted
**Decision:** Source IDs are mandatory on facts, rules, rule literals, and queries; proofs replay against these links.
**Reason:** Explainability requires auditable evidence rather than generated explanations.

## D-005: FastAPI and Next.js remain separate services

**Status:** accepted
**Decision:** The Python research/backend service and the pnpm-managed Next.js App Router frontend have separate build/test boundaries and communicate over a configured HTTP API.
**Reason:** The reasoning ecosystem is Python-oriented while the approved research UI stack is TypeScript/Next.js.

## D-006: Defer provider selection

**Status:** accepted
**Decision:** Phase 1 contains no provider dependency, credential, model call, or provider-specific AST.
**Reason:** The contract and deterministic components should not be coupled to a commercial service before baseline implementation is approved.

## D-007: Use JSONL as canonical experiment evidence

**Status:** accepted
**Decision:** Later per-example inputs/outputs will use append-friendly JSONL; optional SQLite will index run metadata only.
**Reason:** JSONL is inspectable, streamable, diffable at the record level, and straightforward to regenerate aggregates from.

## D-008: Prefer fail-closed abstention

**Status:** accepted
**Decision:** Validation, correction, confidence, consistency, or resource uncertainty prevents an accepted answer.
**Reason:** A visible abstention is safer and more scientifically honest than a guessed formalization.

## D-009: Treat the ProofWriter archive as untrusted input

**Status:** accepted
**Decision:** Phase 2 downloads ProofWriter through a bounded streaming client, validates an optional SHA-256 digest and ZIP integrity, and extracts only through traversal- and symlink-resistant code. Raw archives, extracted data, normalized records, and generated run outputs stay ignored by Git.
**Reason:** A public benchmark is still external input. Bounded, fail-closed acquisition protects the workstation and keeps copyrighted or very large data out of repository history.

## D-010: Normalize only labels justified by open-world evidence

**Status:** accepted
**Decision:** In an explicitly identified open-world ProofWriter variant, `true` maps to `ENTAILED`, `Unknown` maps to `UNKNOWN`, and `false` maps to `CONTRADICTED` only when the record carries an explicit negation-proof strategy and non-empty proof metadata. Closed-world false labels and ambiguous world assumptions are rejected rather than reinterpreted.
**Reason:** VeriLogic-NS distinguishes contradiction from absence. A closed-world negative label cannot safely cross that semantic boundary.

## D-011: Keep gold evidence outside the predictor boundary

**Status:** accepted
**Decision:** The evaluation protocol presents predictors with an immutable `PredictionInput` projection of a `BenchmarkExample`, excluding the gold label, raw source label, and gold proof payload. Gold fields remain available only to the evaluator after prediction.
**Reason:** A separate prediction view makes accidental label leakage structurally harder and lets tests prove that predictors cannot inspect the answer key.

## D-012: Preserve official splits and make sampling reproducible

**Status:** accepted
**Decision:** Normalization preserves the archive's train, development, and test split labels. Sampling uses stable content-derived ordering with an explicit seed, supports random, label-balanced, and label/depth-stratified selection, and refuses test use unless the run explicitly opts in. Overlaps are reported, never silently removed.
**Reason:** Scientific comparisons require repeatable subsets, auditable leakage checks, and an intact held-out test boundary.

## D-013: Persist evaluation evidence atomically

**Status:** accepted
**Decision:** Each run writes a manifest, prediction JSONL, and aggregate metrics into a unique incomplete directory before one atomic promotion. Prediction rows omit gold data; failures retain a visibly incomplete manifest and existing run directories are never overwritten.
**Reason:** Crash-safe, immutable run evidence prevents partial results from being mistaken for completed experiments and supports independent metric regeneration.

## D-014: Report abstention separately from answer correctness

**Status:** accepted
**Decision:** Phase 2 reports overall accuracy, answered-only accuracy, coverage, selective risk, three-label macro precision/recall/F1, a confusion matrix with `ABSTAIN` and `ERROR` prediction columns, and per-label/per-depth slices. Abstentions and errors are not counted as correct answers.
**Reason:** A selective system can appear accurate by answering very little. Coverage and selective risk expose that tradeoff while the complete confusion matrix preserves every outcome.

## D-015: Freeze comparable Phase 3 LLM baselines

**Status:** accepted
**Decision:** Direct and few-shot baselines use one provider-neutral request contract, the same OpenAI Responses model/settings, strict three-label schema, task definition, and ordered 30-record OWA development pilot. Few-shot adds only six deterministically selected training demonstrations. Every artifact is hash-checked before use.
**Reason:** Structural equality makes the demonstration intervention auditable and prevents label leakage or accidental prompt drift from invalidating the paired comparison.

## D-016: Make paid execution opt-in and cache-addressed

**Status:** accepted
**Decision:** `OPENAI_API_KEY` presence is insufficient to call a provider. Live execution requires separate paid-use and external-transfer flags plus a positive cap that covers the preflight worst case. Valid responses are atomically cached by all behavior-affecting request fields; replay has no live provider and fails on a miss.
**Reason:** Commercial calls and benchmark transfer are consequential actions. Explicit gates, bounded retries/concurrency, a circuit breaker, and validated replay reduce unintended spend, duplicate work, and irreproducible evidence.

## D-017: Use deterministic signed least-fixpoint reasoning

**Status:** accepted
**Decision:** The Phase 4 engine treats positive and explicit negative literals as separate signed atoms, applies safe conjunctive rules by deterministic forward chaining, and classifies both-polarity support as query-specific `INCONSISTENT`. It performs neither contraposition nor negation as failure.
**Reason:** These semantics match the approved open-world fragment, terminate over a finite domain, and avoid both absence-as-falsity and explosive inference.

## D-018: Make proofs canonical and independently replayable

**Status:** accepted
**Decision:** Conclusions carry a source-linked, SHA-256-addressed proof DAG selected by a documented canonical order. A separate verifier checks nodes/rules/sources/graph structure and independently recomputes the least-fixpoint status with a naive algorithm.
**Reason:** Deterministic producer output alone is insufficient evidence. Independent replay exposes tampering, producer defects, and fabricated unknown results.

## D-019: Limit ProofWriter adaptation to oracle formal fields

**Status:** accepted
**Decision:** Phase 4 may deterministically parse ProofWriter's provided formal S-expressions for OWA development conformance, but it may not infer formal logic from benchmark prose. Raw records and generated conformance results remain ignored.
**Reason:** This measures the symbolic component's ceiling without conflating it with the unimplemented semantic parser or leaking the test split.

## D-020: Record recovered local Git provenance

**Status:** accepted
**Decision:** The supplied source ZIP did not contain `.git`. The recovered workspace therefore has a new local history beginning with source snapshot `9cd76a6`; Phase 3 operational evidence is checkpointed at `7eddcac`. These hashes are local provenance and do not pretend to reconstruct omitted history.
**Reason:** Honest provenance prevents the recovered repository from being confused with the original unseen Git history.

## D-021: Split theory and query semantic parsing

**Status:** accepted
**Decision:** Phase 5 makes one local request per unique natural-language theory and a separate request
per query. Prompts receive only neutral `sentN` source identifiers and text; original source IDs are
restored from an internal non-provider mapping.
**Reason:** Theory reuse reduces inference work, while the dedicated gold-free view makes label,
formal-representation, proof, depth, path, and raw-key leakage structurally difficult.

## D-022: Keep Phase 5 correction-free and fail closed

**Status:** accepted
**Decision:** Phase 5 performs no reflection, repair prompt, voting, confidence gate, or solver
feedback. A parser/provider/schema/source/semantic failure becomes evaluation `ERROR`, never the valid
open-world answer `UNKNOWN`. Only an identical transient transport retry is allowed.
**Reason:** A correction-free baseline isolates raw semantic-parser quality and prevents hidden model
reasoning or post-result adaptation from inflating performance.

## D-023: Freeze and retain the negative parser result

**Status:** accepted
**Decision:** The final train-developed prompt/schema/runtime were frozen before the same 30-example
development pilot used in Phases 3 and 4. Its low 10% overall accuracy and 13.33% coverage are retained
and reported; no post-development prompt tuning is permitted.
**Reason:** The result identifies semantic formalisation—not deterministic reasoning—as the current
bottleneck. Negative evidence is scientifically useful and must not be hidden or tuned away.

## D-024: Bound semantic correction and gate release on observable evidence

**Status:** accepted before the Phase 6 development pilot  
**Decision:** Phase 6 reuses raw Phase 5 candidates, allows at most one local neural replacement per
theory/query, revalidates deterministically, independently verifies reasoning, and exposes an answer
under P2 only when every mandatory evidence gate and semantic critic passes. `UNKNOWN`, `ABSTAIN`,
and `ERROR` remain distinct.  
**Reason:** A bounded, traceable controller measures recovery without hiding an open-ended
self-refinement agent. Observable evidence is auditable; model self-confidence and development gold
are not.
