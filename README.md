# VeriLogic-NS

VeriLogic-NS is an explainable neuro-symbolic research framework that will combine an existing large language model with a deterministic symbolic reasoner. The language model is limited to converting natural-language premises and queries into a restricted typed AST; validation and formal reasoning remain deterministic and auditable.

The project answers whether a conclusion follows from supplied premises. It does **not** establish that those premises are factually true.

## Current implementation: Phases 1–3 infrastructure

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

No live provider call or research pilot result is committed. The provider path requires explicit paid-use, external-transfer, and cost-cap flags even when a key exists. There is no semantic parser, symbolic solver, database, authentication, or deployment. Generated smoke outputs use a fake provider, are ignored, and are not research results.

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
