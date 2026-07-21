# VeriLogic-NS

VeriLogic-NS is an explainable neuro-symbolic research framework that will combine an existing large language model with a deterministic symbolic reasoner. The language model is limited to converting natural-language premises and queries into a restricted typed AST; validation and formal reasoning remain deterministic and auditable.

The project answers whether a conclusion follows from supplied premises. It does **not** establish that those premises are factually true.

## Current implementation: Phases 1-5

The repository provides:

- durable project, architecture, logic, security, testing, and research specifications;
- the versioned `theory.v1` JSON Schema with valid and intentionally invalid examples;
- a FastAPI service with `GET /health`;
- a minimal accessible Next.js App Router page with live backend-health states;
- backend and schema tests;
- Ruff, ESLint, TypeScript, Docker Compose, and GitHub Actions configuration;
- safe, streamed ProofWriter V2020.12.3 acquisition with observed provenance;
- OWA main-file normalization into a versioned `BenchmarkExample` contract;
- deterministic train/development sampling and explicit test-split protection;
- cross-split duplicate, content-hash, question, and theory-overlap reports;
- a gold-redacted predictor protocol, deterministic development predictors, and an atomic JSONL evaluation runner;
- accuracy, coverage, selective-risk, three-class macro metrics, confusion matrices, and per-depth metrics.
- provider-neutral direct and fixed six-example few-shot LLM predictors;
- an exact-pinned local Ollama native-chat adapter and optional OpenAI Responses adapter, both
  using strict three-label Structured Outputs;
- frozen prompts, output schema, demonstrations, and a balanced 30-example OWA development pilot;
- bounded retry, circuit-breaker, cost-cap, response-cache, replay, telemetry, and paired-comparison support.
- a strict semantic validator for the typed theory contract;
- a deterministic finite Datalog-style forward-chaining engine with explicit negation;
- four-way `ENTAILED`, `CONTRADICTED`, `UNKNOWN`, and `INCONSISTENT` query decisions;
- canonical source-linked proof DAGs, SHA-256 identities, and independent proof replay;
- resource-bounded reasoning CLIs and an oracle-structure ProofWriter conformance adapter.
- a gold-isolated, loopback-only Ollama semantic parser with separate theory/query prompts;
- strict neural fact/rule/query output schemas, neutral source IDs, and deterministic AST conversion;
- parser-specific atomic cache/replay, typed fail-closed errors, and detailed parsing metrics.

A zero-cost local Ollama baseline pilot and a correction-free semantic-parser pilot have completed and
were replayed from cache. Generated records, caches, and raw metrics remain ignored local artifacts;
no hosted provider was called. The Phase 5 aggregate result is documented honestly in
`docs/PHASE5_PILOT_RESULTS.md`: the small local parser is the current bottleneck. There is no Phase 6
correction/confidence system, production end-to-end API, database, authentication, or deployment.

## Prerequisites

- Git
- Python 3.11 or newer
- Node.js 20.9 or newer (Node.js 22+ recommended)
- pnpm 9.15.4
- Docker with Compose v2, optionally, for container execution

## Backend setup

From `backend/`:

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m uvicorn verilogic_ns_api.main:app --reload --host 127.0.0.1 --port 8000
```

`GET http://localhost:8000/health` returns the service name, status, and version. Copy `backend/.env.example` to `backend/.env` only when local overrides are needed.

## Frontend setup

From `frontend/`:

```bash
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://localhost:3000`. `NEXT_PUBLIC_API_BASE_URL` defaults to `http://localhost:8000`; copy `frontend/.env.example` to `frontend/.env.local` to override it. pnpm is the only supported frontend package manager.

## Verification

Backend, from `backend/`:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

Frontend, from `frontend/`:

```bash
pnpm lint
pnpm type-check
pnpm build
```

Schema fixture behavior is covered by `backend/tests/test_schema.py` and runs as part of pytest.

## ProofWriter dataset commands

Activate the backend virtual environment and run from the repository root:

```bash
python -m verilogic_ns_api.datasets --help
python -m verilogic_ns_api.datasets download proofwriter
python -m verilogic_ns_api.datasets inspect proofwriter --variant depth-1
python -m verilogic_ns_api.datasets prepare proofwriter --variant depth-1
```

The download defaults to archive-only because the verified 214 MB ZIP expands to about 3.41 GB. The loader streams from the ZIP directly. Use `--extract` only when an extracted copy is required; safe extraction remains inside the configured dataset root. Existing archives are reused only after ZIP/SHA validation, and replacement requires `--force`.

The archive README does not state a dataset licence and contains no licence file. The observed checksum in `datasets/proofwriter/provenance.observed.json` was computed locally and is not publisher-verified. See `datasets/proofwriter/DATASET_CARD.md` before reuse or redistribution.

## Evaluation smoke run

```bash
python -m verilogic_ns_api.evaluation --help
python -m verilogic_ns_api.evaluation run --config experiments/configs/proofwriter-smoke.yaml
```

The tracked smoke configuration uses six clearly synthetic train/development examples and `ConstantUnknownPredictor`. Each run creates a unique ignored directory under `results/runs/` containing `predictions.jsonl`, `run-manifest.json`, and `metrics.json`. A run is first marked incomplete and is atomically promoted only when all outputs are complete.

## LLM baseline workflow

Run from the repository root with the backend environment active:

```bash
python -m verilogic_ns_api.baselines --help
python -m verilogic_ns_api.baselines smoke --condition direct
python -m verilogic_ns_api.baselines smoke --condition few_shot
python -m verilogic_ns_api.baselines ollama-smoke
python -m verilogic_ns_api.baselines plan --config experiments/configs/ollama-direct-pilot.yaml
python -m verilogic_ns_api.baselines plan --config experiments/configs/ollama-few-shot-pilot.yaml
```

These smoke and plan commands need no key and make no real inference request. The operational pilot
uses a digest-pinned model through loopback-only Ollama with cloud features disabled, so ProofWriter
data stays local and API cost is USD 0.00. The optional OpenAI path remains implemented and mocked
but operationally unverified; its paid live gate is unchanged. See `docs/LOCAL_LLM_BASELINE.md` and
`docs/LLM_BASELINES.md` for the separate local and hosted-provider protocols.

## Symbolic reasoning

Run from the repository root with the backend environment active:

```bash
python -m verilogic_ns_api.reasoning --help
python -m verilogic_ns_api.reasoning reason --input examples/theories/entailed.json --human
python -m verilogic_ns_api.reasoning saturate --input examples/theories/binary-join.json
python -m verilogic_ns_api.reasoning inspect-closure --input examples/theories/inconsistent.json
```

The engine consumes only validated `theory.v1` JSON. It supports unary/binary predicates,
conjunctive rules, variables and constants, explicit positive/negative literals, multi-step
reasoning, positive recursion, and query-specific inconsistency. It uses open-world semantics:
missing evidence produces `UNKNOWN`, never an inferred negative. It does not use an LLM, parse
natural language, perform contraposition, or establish that source premises are factually true.

Proofs use [the versioned proof contract](docs/PROOF_FORMAT.md) and can be independently replayed.
The implementation and formal semantics are documented in
[the symbolic-engine guide](docs/SYMBOLIC_ENGINE.md).

The observed Phase 4 oracle-structure checks were 300/300 on a balanced ProofWriter OWA development
sample and 30/30 on the frozen Phase 3 development sample, with every proof verified. This is a
symbolic ceiling using dataset-provided formal fields, not natural-language parsing performance or a
final research result. Raw conformance records remain ignored locally, and the ProofWriter licence
status remains unverified.

## Neural semantic parser

```bash
python -m verilogic_ns_api.semantic_parsing --help
python -m verilogic_ns_api.semantic_parsing plan --config experiments/configs/ollama-semantic-parser-pilot.yaml
python -m verilogic_ns_api.semantic_parsing replay --config experiments/configs/ollama-semantic-parser-pilot.yaml --dataset pilot --run-id REPLAY_ID
```

The parser uses the exact local model/digest from Phase 3, sends no data externally, and never sees
gold labels or ProofWriter formal fields. Parser errors fail closed as `ERROR`; they are never changed
to `UNKNOWN`. See `docs/NEURAL_SEMANTIC_PARSER.md` for the architecture, commands, security boundary,
and frozen protocol.

## Validation-guided correction and abstention

```bash
python -m verilogic_ns_api.validation_correction --help
python -m verilogic_ns_api.validation_correction plan --config experiments/configs/ollama-validation-correction-pilot.yaml
python -m verilogic_ns_api.validation_correction replay --config experiments/configs/ollama-validation-correction-pilot.yaml --run-id REPLAY_ID
```

Phase 6 converts deterministic validation failures into bounded typed feedback, asks the same
loopback-only local model for at most one complete replacement per theory/query, revalidates the
replacement, and applies a separate structured fidelity critic. The final selective policy answers
only when deterministic validation, critic acceptance, reasoning, and independent verification all
pass. A valid `UNKNOWN` is a logical answer; `ABSTAIN` is a deliberate reliability decision and
`ERROR` is an infrastructure failure. See `docs/VALIDATION_CORRECTION.md` and
`docs/ABSTENTION_POLICY.md`.

## Docker

From the repository root:

```bash
docker compose config
docker compose up --build
```

The frontend is exposed on port 3000 and the backend on port 8000 by default. Values in the root `.env.example` document safe local overrides.

## Structure

```text
.
|-- backend/              FastAPI service and tests
|-- frontend/             Next.js research interface foundation
|-- schemas/              Versioned typed AST contracts
|-- examples/theories/    Valid and invalid schema fixtures
|-- docs/                 Project and research specifications
|-- datasets/             Dataset cards/provenance; raw/processed data ignored
|-- experiments/          Versioned evaluation configurations and documentation
|-- prompts/              Frozen, versioned prompt templates
|-- results/              Reserved for generated experiment outputs
`-- .github/workflows/    Continuous integration
```

## Research roadmap

1. Foundation
2. Dataset ingestion and evaluation harness
3. Direct and few-shot LLM baselines
4. Symbolic reasoning engine
5. Neural semantic parser
6. Validation, correction and abstention
7. End-to-end neuro-symbolic integration
8. Research frontend
9. Full experiments and ablations
10. Deployment, report and presentation

See `docs/PHASE_PLAN.md` for phase gates and `docs/PROJECT_CHARTER.md` for scope.
