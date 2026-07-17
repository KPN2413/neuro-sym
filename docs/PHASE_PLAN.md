# Phase Plan

Only the current phase may be implemented. Each phase begins with an explicit prompt and ends with tests, documentation, a verification report, and a cleanly described Git state.

## 1. Foundation

**Status:** completed at Git checkpoint `cac8f21`.

Create durable specifications, the versioned typed-AST schema and fixtures, FastAPI and Next.js shells, CI, Docker, and local verification. No solver, LLM call, dataset download, database, or experiment result.

**Gate:** all Phase 1 acceptance checks pass and the repository accurately reports limitations.

## 2. Dataset ingestion and evaluation harness

**Status:** completed on `phase/02-dataset-evaluation`; recorded by the branch's Phase 2 feature commit.

Add a versioned ProofWriter ingestion path, provenance/checksum manifest, normalized records, deterministic sampling without resplitting, JSONL run records, evaluation interfaces, and fixture-driven metrics tests. Do not call an LLM.

**Gate:** selected benchmark records ingest reproducibly and known fixture predictions produce exact expected metrics.

## 3. Direct and few-shot LLM baselines

**Status:** implementation pass; live pilot not authorized.

Implemented the provider-independent LLM port, official OpenAI Responses adapter, versioned/hash-frozen prompts and selections, bounded retries/timeouts/concurrency, circuit breaking, usage/cost accounting, content-addressed replay, direct/few-shot conditions, paid/data-transfer gates, and mocked contract tests. No live request or pilot metric has been produced.

**Gate:** baseline runs are reproducible, raw outputs and errors are recorded, and provider-free tests pass.

## 4. Symbolic reasoning engine

Implement semantic AST checks needed by the engine, deterministic finite forward chaining, unary/binary predicates, explicit negation, open-world decisions, multi-step derivations, inconsistency detection, and source-linked proof construction/replay.

**Gate:** unit, integration, and property-based logic/proof suites pass without any LLM dependency.

## 5. Neural semantic parser

Implement natural-language-to-AST prompting through the approved provider port, strict parsing, parser metadata, prompt/version tracking, and evaluation against reference formalizations where available.

**Gate:** parser outputs never bypass schema validation and parse/meaning errors are measured rather than hidden.

## 6. Validation, correction and abstention

Add structural/semantic meaning-preservation checks, limited solver-guided correction, confidence calibration/gating, explicit correction logs, and fail-closed abstention policies.

**Gate:** adversarial and ambiguity tests demonstrate bounded correction and safe rejection; coverage/selective-risk metrics are reproducible.

## 7. End-to-end neuro-symbolic integration

Connect ingestion, parser, validators, correction/gating, reasoner, proofs, and experiment records behind stable backend services and APIs.

**Gate:** representative cases complete end to end with replayable proofs and classified failure states.

## 8. Research frontend

Build the accessible research interface for theory entry, normalized AST inspection, decisions, proofs, validation/abstention explanations, runs, and baseline comparison. Use shadcn/ui when UI implementation begins.

**Gate:** critical UI states and API flows pass accessibility, browser, and error-path verification without fabricated data.

## 9. Full experiments and ablations

Freeze protocol/configurations, execute mandatory conditions and approved ablations, compute accuracy/proof/robustness/latency/cost metrics, and retain raw reproducible records.

**Gate:** aggregates reproduce from raw JSONL, denominators and failures are reported, and no unmeasured claim appears.

## 10. Deployment, report and presentation

Harden and deploy the separate frontend/backend, finalize operational documentation, capstone report, limitations, reproducibility package, and presentation artifacts.

**Gate:** deployment smoke tests, security checks, clean-room setup instructions, report figures, and presentation claims agree with recorded evidence.
