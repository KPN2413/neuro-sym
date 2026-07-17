# LLM Baselines

## Scope and status

Phase 3 adds reproducible direct and fixed few-shot ProofWriter classifiers. The zero-cost
operational condition is the fully local Ollama baseline documented in
[`LOCAL_LLM_BASELINE.md`](LOCAL_LLM_BASELINE.md). The OpenAI boundary remains implemented and
mocked, but it is not operationally verified and made zero live requests. Neither condition is the
future semantic parser or neuro-symbolic system.

Both conditions use the OpenAI Responses API, `gpt-5.6-terra`, low reasoning effort, 512 maximum output tokens, no tools, no browsing, no temperature, and strict `{"label": "ENTAILED" | "CONTRADICTED" | "UNKNOWN"}` output. `UNKNOWN` is an answer. Explicit refusal maps to `ABSTAIN`; exhausted transport or invalid output maps to `ERROR`. No prompt asks for rationale, proof, or chain-of-thought.

The implementation was checked against the official [model reference](https://developers.openai.com/api/docs/models/gpt-5.6-terra), [Responses text guide](https://developers.openai.com/api/docs/guides/text), [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs), and [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning). The official SDK is pinned to `openai==2.45.0`; SDK retries are disabled so the repository's bounded retry/circuit policy remains authoritative.

## Frozen scientific protocol

- Dataset: observed ProofWriter V2020.12.3 archive, OWA `depth-5` only; licence remains unverified.
- Pilot: 30 development records, two per label at each depth 0, 1, 2, 3, and 5; no test records.
- Demonstrations: six training records, two per label, each label represented at depths 0 and 2.
- Equality: model/settings, task definition, output schema, ordered pilot IDs, and evaluator are identical. Few-shot adds only the demonstrations.
- Isolation: renderer/provider APIs accept `PredictionInput`, which has no gold label, raw label, or gold proof. Runtime sends context/query only; manifests retain IDs/hashes/selection metadata locally.

The prompts, schema, manifests, configs, seed, and archive are hash-checked before planning or execution. Do not tune after inspecting the pilot. A behavior-changing defect fix requires versioning and a documented rerun while preserving prior local evidence. Thirty records do not support significance or state-of-the-art claims.

## OpenAI network-free workflow

From the repository root with `backend/.venv` active:

```text
python -m verilogic_ns_api.baselines --help
python -m verilogic_ns_api.baselines smoke --condition direct
python -m verilogic_ns_api.baselines smoke --condition few_shot
python -m verilogic_ns_api.baselines plan --config experiments/configs/openai-direct-pilot.yaml
python -m verilogic_ns_api.baselines plan --config experiments/configs/openai-few-shot-pilot.yaml
python -m verilogic_ns_api.baselines canary-plan --direct-config experiments/configs/openai-direct-pilot.yaml --few-shot-config experiments/configs/openai-few-shot-pilot.yaml
```

Smoke uses a deterministic fake and immediately proves cache-only replay. Plan loads real local records to validate IDs/hashes and estimate tokens/cache hits/cost, but cannot construct a provider. CLI help, tests, CI, smoke, planning, comparison, and replay require no key.

Configure `OPENAI_API_KEY` only through an ignored local environment or secret manager. Do not paste it into chat or place it in YAML, source, Docker, frontend variables, logs, results, or CI. Its presence alone is inert. Every live command separately requires `--allow-paid-api`, `--confirm-external-data-transfer`, and `--max-cost-usd VALUE`; the cap must cover the preflight worst case and is checked before each dispatch. A synthetic two-condition canary must pass before benchmark runs.

## Cache, retry, and telemetry

`results/cache/llm-responses/` is an ignored content-addressed cache keyed by provider/API/model parameters, prompt/schema versions and hashes, demonstration-manifest hash, example ID, and rendered-request hash. Entries are strict and atomically replaced; metadata mismatch aborts, corrupt/truncated files are quarantined, concurrent identical requests serialize, and stale locks can be recovered. Replay has no inner provider and fails when any entry is missing.

Only rate limits, selected timeouts/connections, and server failures receive bounded exponential-backoff retries with jitter. Authentication, permission, unsupported model, invalid request, and schema failures do not retry or fall back to another model. Repeated exhausted failures open a circuit breaker. A successful response records configured/returned model, provider request ID, timestamps, latency, input/output/reasoning/cached tokens, retry count, cache state, status, and estimated cost. Raw payloads remain local. An ambiguous timeout can still have reached the provider, so duplicate billing cannot be eliminated; its reservation is conservatively retained.

Pricing is a timestamped estimate, not an invoice. Phase 3 configurations record the official pricing URL and 2026-07-13 standard rates. Reverify prices before later execution. The ProofWriter archive is publicly downloadable but has no verified dataset licence; external transfer still requires explicit user and institutional approval.

## Replay and comparison

After an authorized complete cache exists:

```text
python -m verilogic_ns_api.baselines run --config experiments/configs/openai-direct-pilot.yaml --mode replay
python -m verilogic_ns_api.baselines run --config experiments/configs/openai-few-shot-pilot.yaml --mode replay
python -m verilogic_ns_api.baselines compare --direct-run DIRECT_RUN --few-shot-run FEW_RUN --selection-manifest experiments/manifests/proofwriter-owa-depth5-dev-pilot.v1.json --output results/comparisons/pilot.json
```

Comparison rejects mismatched model/dataset/seed settings or example order and reports paired accuracy/coverage, per-depth/per-label deltas, correctness quadrants, and a disagreement matrix. Final test-split evaluation is deferred until the complete capstone protocol is frozen.
