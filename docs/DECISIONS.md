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
**Decision:** Source IDs are mandatory on facts, rules, rule literals, and queries; future proofs must replay against these links.
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
