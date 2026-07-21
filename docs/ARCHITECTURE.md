# Architecture

## System context

VeriLogic-NS has two trust domains. Natural language, dataset records, provider responses, and user input are untrusted. The versioned typed AST becomes eligible for deterministic reasoning only after validation.

```text
Natural-language theory + query
        |
        v
Gold-isolated local semantic-parser adapter (Phase 5)
        |
        v
Restricted JSON AST -> structural validator -> semantic validator
        |                                      |
        | invalid/uncertain                    | valid raw candidate
        v                                      v
Typed feedback -> one local correction     semantic critic
        |                                      |
        +------------> full revalidation <-----+
                              |
                              v
                    evidence gate / abstain
                                                |
                                                v
                               deterministic forward-chaining engine
                                                |
                                                v
                ENTAILED / CONTRADICTED / UNKNOWN / INCONSISTENT
                              + source-linked proof or explanation
```

Phases 1-6 implement the contracts, service/UI shells, ProofWriter ingestion, deterministic sampling/leakage reporting, evaluation harness, direct/few-shot LLM baseline infrastructure, a gold-isolated local semantic parser, a bounded local critic/correction controller, semantic theory validation, the symbolic solver, and proof replay. They do not implement a production end-to-end API or research dashboard.

## Components

### Research frontend

`frontend/` is a Next.js App Router application. In Phase 1 it renders project identity, phase status, and live backend-health loading/success/failure states. Later it will submit theories, display normalized ASTs and proof traces, and expose research comparisons. It must not fabricate reasoning output.

### API backend

`backend/` is a FastAPI service created through an application factory. Settings come from environment variables, and CORS is limited to configured origins. Phase 1 exposes only `GET /health`. Later routers will orchestrate parsing, validation, reasoning, and experiments without embedding provider logic in route handlers.

### Semantic parser port

`verilogic_ns_api.semantic_parsing` accepts dedicated gold-free theory/query views and returns
untrusted schema-constrained candidates. The Phase 5 provider is loopback-only Ollama pinned by tag,
digest, and version. Neutral `sentN` IDs prevent ProofWriter formal keys from entering prompts;
internal mappings restore provenance only after inference. Source coverage and the complete Phase 4
AST validator must pass before reasoning. Parser failures become typed evaluation errors, never
`UNKNOWN`, and no generated code is executed.

### Validation boundary

The JSON Schema provides structural validation: known fields, identifier patterns, literal shape, arity bounds, source fields, and version. Phase 4 strict Pydantic models enforce cross-object constraints such as declared-predicate arity, reference existence, variable safety, type compatibility, and source-ID integrity. Any unresolved error fails closed. Natural-language meaning preservation remains Phase 5/6 work.

### Validation/correction controller

`verilogic_ns_api.validation_correction` turns Phase 5 validation outcomes into stable typed feedback,
applies a separate local fidelity critic, and permits one schema-constrained replacement per
theory/query. Corrected candidates cross the complete validation boundary again. P1 releases every
deterministically valid, independently verified result for diagnostic comparison; P2 additionally
requires critic acceptance. State transitions, hashes, decisions, abstention reasons, and aggregate
telemetry are immutable and replayable without storing chain-of-thought.

### Symbolic engine

`verilogic_ns_api.reasoning` grounds safe conjunctive rules and applies deterministic delta-based forward chaining to a fixed point. Positive and explicitly negative literals are separate signed atoms. Canonical derivations preserve source provenance; a separately implemented naive closure validates proof status. Configurable limits prevent partial computations from being mislabeled as complete.

### Experiment harness

`verilogic_ns_api.datasets` safely downloads or streams ProofWriter from ZIP, preserves official splits, normalizes one validated `BenchmarkExample` per question, and reports rather than changes overlap. Raw CWA false labels are rejected as ambiguous for the three-way contract.

`verilogic_ns_api.evaluation` passes a gold-redacted `PredictionInput` to a provider-independent predictor protocol. The runner records typed predictions, continues after per-example failures, aborts on systemic authentication/configuration failures, computes metrics, writes an incomplete manifest first, and atomically publishes JSONL/JSON outputs only when complete.

`verilogic_ns_api.baselines` plugs direct and fixed few-shot predictors into that boundary. A small `LLMProvider` protocol isolates the official OpenAI Responses adapter. Requests use the same strict three-label Pydantic output, with no tools, browsing, temperature, or rationale request. Content-addressed local caching sits outside bounded retry and circuit breaking so interrupted work can resume and cache replay cannot reach a provider. Usage, model IDs, request IDs, latency, retries, refusals, cache state, and timestamped cost estimates extend the existing prediction/metric contracts compatibly.

The live CLI validates frozen prompt/schema/sample hashes and the local archive before constructing an SDK client. Paid execution requires three deliberate flags; mere credential presence is inert. Raw provider payloads are confined to ignored cache files, while aggregate and gold-free prediction evidence uses the Phase 2 runner.

### Dataset trust boundary

Network bytes are streamed to an ignored `.part` file with configured timeouts and size limits. A ZIP integrity check and optional expected SHA-256 must pass before atomic rename. The observed SHA-256 is not described as publisher-verified. Optional extraction prevalidates every path, rejects symlinks and escape paths, bounds total entries/expanded bytes, and writes only beneath a content-addressed local raw directory.

## Decision semantics

- `ENTAILED`: the query literal is derivable and its explicit opposite is not.
- `CONTRADICTED`: the explicit opposite is derivable and the query is not.
- `UNKNOWN`: neither polarity is derivable under open-world semantics.
- `INCONSISTENT`: both polarities are derivable for the query; unrelated conflicts remain telemetry and do not cause explosion.
- `INVALID`: the input cannot safely cross the validation boundary.

## Proof architecture

Every asserted fact and rule has a `source_id`. The versioned proof DAG contains exact source facts, grounded rule applications, antecedent roots, source text, signed conclusions, depths, and canonical hashes. The independent verifier checks graph integrity and replays the claimed status against a naive closure rather than trusting producer output or provider prose.

## Deployment boundaries

Docker Compose runs the frontend and backend as separate services. The browser uses a public API base URL; server-side service names are not exposed as browser URLs. No database or provider credential exists in Phase 1. Future deployments must retain explicit origin allowlists, environment-only secrets, resource limits, and non-root containers where practical.
