# VeriLogic-NS Engineering Rules

These rules apply to every change in this repository. A phase-specific prompt may narrow the work, but it must explicitly authorize any change to public contracts or approved scope.

## Purpose and architecture

VeriLogic-NS is an explainable neuro-symbolic research system. A provider-independent LLM adapter will translate natural-language facts, rules, and a query into the restricted, versioned JSON AST in `schemas/`. Structural and semantic validators must accept the AST before a deterministic Datalog-style forward-chaining engine can inspect it. The system returns `ENTAILED`, `CONTRADICTED`, or `UNKNOWN`, with `INVALID` and `INCONSISTENT` reserved as internal safety states. Accepted conclusions must have source-linked proofs.

The LLM is a semantic parser, never the authority for logical entailment. The system verifies conclusions relative to supplied premises; it does not establish that those premises are factually true.

Approved application boundaries:

- `frontend/`: Next.js App Router, TypeScript, Tailwind CSS, pnpm only.
- `backend/`: Python 3.11+, FastAPI, Pydantic v2, pytest, Hypothesis, and Ruff.
- `schemas/`: versioned, non-executable JSON contracts.
- `datasets/`, `experiments/`, `prompts/`, and `results/`: research artifacts introduced only in their approved phases.
- `docs/`: architecture, research, testing, security, and decision records.

Do not add stretch features unless a phase explicitly requests them. Stretch features include FOLIO, Z3 verification, fine-tuning, RAG, knowledge graphs, multiple LLM providers, multi-agent architecture, authentication, payments, and mobile applications.

## Commands

Run commands from the indicated directory. Use an isolated virtual environment for Python and never install project packages globally.

### Backend

```text
cd backend
python -m venv .venv
# Activate .venv using the command appropriate for the local shell.
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m uvicorn verilogic_ns_api.main:app --reload --host 127.0.0.1 --port 8000
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

Manage all Python runtime and development dependencies in `backend/pyproject.toml`, with bounded compatible version ranges. Do not introduce a second dependency manager without an explicit architecture decision. Regenerate any future lock file through its declared tool; do not hand-edit it.

### Frontend

```text
cd frontend
pnpm install --frozen-lockfile
pnpm dev
pnpm lint
pnpm type-check
pnpm build
```

Use pnpm exclusively. Do not run npm, Yarn, or Bun to add, remove, install, or update frontend dependencies. Update `package.json` and `pnpm-lock.yaml` together through pnpm.

### Dataset and evaluation commands

Activate `backend/.venv`, then run these from the repository root so configuration paths remain reproducible:

```text
python -m verilogic_ns_api.datasets --help
python -m verilogic_ns_api.datasets download proofwriter
python -m verilogic_ns_api.datasets inspect proofwriter --variant depth-1
python -m verilogic_ns_api.datasets prepare proofwriter --variant depth-1
python -m verilogic_ns_api.evaluation --help
python -m verilogic_ns_api.evaluation run --config experiments/configs/proofwriter-smoke.yaml
python -m verilogic_ns_api.baselines --help
python -m verilogic_ns_api.baselines smoke --condition direct
python -m verilogic_ns_api.baselines smoke --condition few_shot
python -m verilogic_ns_api.baselines plan --config experiments/configs/openai-direct-pilot.yaml
```

Dataset download, extraction, preparation, samples, and evaluation outputs are local generated artifacts and must remain ignored. Track only acquisition/normalization code, schemas, documentation, safe aggregate provenance, configurations, and small explicitly synthetic fixtures.

### Whole stack

```text
docker compose config
docker compose up --build
```

## Contract and safety rules

- Treat every model response and dataset record as untrusted input.
- Stream dataset downloads to partial files, enforce limits/timeouts/checksums, validate ZIPs before rename, and reject traversal or symbolic-link entries during extraction.
- Never execute model-generated Python, JavaScript, Prolog, SQL, shell, templates, or other code.
- The versioned typed AST is the only boundary between language parsing and symbolic reasoning. Do not bypass it with provider-specific payloads or free-form expressions.
- Reject unknown fields, unsafe identifiers, unsupported operators, missing source links, arity errors, type errors, and unsafe rules.
- Fail closed. If parsing, validation, correction, confidence gating, or consistency checks are uncertain, return or record an abstention/invalid state rather than guessing.
- Limit correction to typed, auditable transformations; never silently change premise meaning.
- Preserve explicit negation and open-world semantics. Absence of evidence is not evidence of the opposite.
- Never add real credentials, tokens, private endpoints, personal data, or secrets. Use documented placeholders in `.env.example`; local `.env*` files remain ignored.
- Do not log secrets, full provider payloads containing sensitive data, or hidden model reasoning.
- Do not expose stack traces or internal validation details through public production responses.

## Change discipline

- Work only on the current approved phase and avoid unrelated refactoring.
- Preserve public APIs, schema versions, result meanings, CLI commands, file formats, and documented behavior unless the phase explicitly changes them.
- A breaking contract change requires a new version or an explicit migration decision in `docs/DECISIONS.md`.
- Add or update automated tests with every behavior change. Bug fixes require a regression test.
- Update relevant documentation whenever behavior, configuration, commands, architecture, research protocol, or security assumptions change.
- Keep generated dependencies and build outputs out of Git.
- Do not fabricate benchmarks, costs, proofs, measurements, or experiment results.
- Preserve official dataset splits. Test data requires an explicit opt-in and may not be used for development or tuning.
- Never map CWA false to `CONTRADICTED` without explicit opposite-proof semantics. Current ProofWriter normalization accepts OWA main question files only.
- Genuine predictor interfaces receive the gold-redacted `PredictionInput`, never `gold_label` or `original_raw_label`.
- Never make a paid provider call based on credential presence. Baseline live runs require explicit paid-use and external-transfer approvals plus a validated positive cost cap.
- Never request hidden chain-of-thought. Keep real provider payloads and rendered benchmark text in ignored local caches/results only.

## Test requirements

At minimum, run the checks affected by a change. Before declaring a phase complete, run the full applicable suite: Ruff lint and format check, pytest, schema fixture validation, dataset/evaluation CLI help, synthetic inspection and smoke evaluation, frontend lint, frontend type-check, frontend production build, and Docker configuration validation when Docker is available. Dataset changes require mocked acquisition failures, archive-safety tests, label-mapping tests, split/leakage tests, and atomic-output tests. Future reasoning changes also require unit, integration, proof-integrity, and property-based tests. Security boundaries require negative tests.

Tests must be deterministic by default. Record seeds for randomized experiments and mock paid or external services in routine tests. A test may not assert a result that was not produced by the implementation.

## Definition of done

A task is done only when:

1. Its requested behavior and artifacts are complete without out-of-scope features.
2. New and existing relevant tests pass, including failure-path tests.
3. Lint, formatting, type checks, and builds pass for affected components.
4. Contracts, source links, fail-closed behavior, and secret handling have been reviewed.
5. Documentation and examples match actual behavior.
6. No unrelated user work or public API was changed.
7. `git status` has been inspected and generated or secret files are not staged.
8. The final report lists every command run, its exact result, remaining blockers, and the resulting Git state.
