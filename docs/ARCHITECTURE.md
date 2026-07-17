# Architecture

## System context

VeriLogic-NS has two trust domains. Natural language, dataset records, provider responses, and user input are untrusted. The versioned typed AST becomes eligible for deterministic reasoning only after validation.

```text
Natural-language theory + query
        |
        v
Provider-independent semantic-parser adapter (future)
        |
        v
Restricted JSON AST -> structural validator -> semantic validator
        |                                      |
        | invalid/uncertain                    | approved limited correction (future)
        v                                      v
Fail-closed INVALID/abstain             confidence gate
                                                |
                                                v
                               deterministic forward-chaining engine (future)
                                                |
                                                v
                       ENTAILED / CONTRADICTED / UNKNOWN
                              + source-linked proof or explanation
```

Phases 1–3 implement the contracts, service/UI shells, ProofWriter ingestion, deterministic sampling/leakage reporting, evaluation harness, and direct/few-shot LLM baseline infrastructure. They do not implement a semantic parser, symbolic solver, semantic AST validator, correction loop, or research dashboard.

## Components

### Research frontend

`frontend/` is a Next.js App Router application. In Phase 1 it renders project identity, phase status, and live backend-health loading/success/failure states. Later it will submit theories, display normalized ASTs and proof traces, and expose research comparisons. It must not fabricate reasoning output.

### API backend

`backend/` is a FastAPI service created through an application factory. Settings come from environment variables, and CORS is limited to configured origins. Phase 1 exposes only `GET /health`. Later routers will orchestrate parsing, validation, reasoning, and experiments without embedding provider logic in route handlers.

### Semantic parser port

A future provider-independent interface will accept versioned prompts and natural language and return untrusted JSON candidates plus allowed metadata. Provider output never reaches the reasoner directly and no generated code is executed.

### Validation boundary

The JSON Schema provides structural validation: known fields, identifier patterns, literal shape, arity bounds, source fields, and version. Future semantic validation will enforce cross-object constraints such as declared-predicate arity, reference existence, variable safety, type compatibility, and source-ID integrity. Any unresolved error fails closed.

### Symbolic engine

A future Datalog-style engine will ground safe conjunctive rules and apply deterministic forward chaining to a fixed point. Positive and explicitly negative literals are separate atoms. The engine will maintain derivation provenance and detect when both polarities are derivable.

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
- `INCONSISTENT`: both polarities are derivable; this is an internal safety state and must not be collapsed into a supported answer.
- `INVALID`: the input cannot safely cross the validation boundary.

## Proof architecture

Every asserted fact and rule has a `source_id`. A future proof node will identify the derived literal, the supporting fact or rule, antecedent proof nodes, and source IDs. Proof verification must replay against the normalized AST rather than trust provider prose.

## Deployment boundaries

Docker Compose runs the frontend and backend as separate services. The browser uses a public API base URL; server-side service names are not exposed as browser URLs. No database or provider credential exists in Phase 1. Future deployments must retain explicit origin allowlists, environment-only secrets, resource limits, and non-root containers where practical.
