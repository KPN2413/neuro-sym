# Experiments

The model-independent Phase 2 harness now hosts Phase 3 direct and fixed few-shot LLM predictors. Every predictor still receives only a gold-redacted `PredictionInput` and returns the same typed `PredictionRecord`.

## Configuration

`configs/proofwriter-smoke.yaml` contains:

- dataset source, version, OWA variant, official splits, and manifest reference;
- deterministic sampling seed, maximum count, allowed splits, filters, and random/balanced/stratified strategy;
- predictor kind/version;
- output directory plus optional safe run ID or generated run-ID prefix.

Run from the repository root after activating `backend/.venv`:

```text
python -m verilogic_ns_api.evaluation run --config experiments/configs/proofwriter-smoke.yaml
```

The smoke configuration uses six synthetic train/development examples and never selects test. It produces an ignored unique directory under `results/runs/` with:

- `predictions.jsonl`: one gold-free typed record per example;
- `run-manifest.json`: configuration, seed, predictor, safe environment/package metadata, counts, timestamps, and completion state;
- `metrics.json`: accuracy, answered-only accuracy, coverage, selective risk, three-class macro/per-label metrics, five-column confusion matrix, per-depth metrics, and invalid-prediction count.

Output files are written atomically. Existing complete or incomplete run IDs are never overwritten, and unsafe resume is not supported. A smoke run validates plumbing only and must not be cited as research performance.

## Phase 3 frozen pilot

`openai-direct-pilot.yaml` and `openai-few-shot-pilot.yaml` select the same 30 OWA `depth-5` development examples: two examples for each label at depths 0, 1, 2, 3, and 5. The committed pilot manifest stores IDs, normalized hashes, depths, labels, seed, source archive hash, and sampler version—not ProofWriter text. The six-example manifest selects two training examples per label with shallow/depth-2 coverage and no ID/content overlap.

```text
python -m verilogic_ns_api.baselines plan --config experiments/configs/openai-direct-pilot.yaml
python -m verilogic_ns_api.baselines plan --config experiments/configs/openai-few-shot-pilot.yaml
python -m verilogic_ns_api.baselines run --config CONFIG --mode replay
python -m verilogic_ns_api.baselines compare --direct-run DIRECT --few-shot-run FEW --selection-manifest experiments/manifests/proofwriter-owa-depth5-dev-pilot.v1.json --output results/comparisons/pilot.json
```

Planning validates the 214 MB archive and every frozen hash but makes no network call. Replay
refuses an incomplete cache. Live OpenAI execution is intentionally omitted here; it additionally
requires the documented explicit approvals and cost cap in `docs/LLM_BASELINES.md`.

The operational zero-cost condition uses `ollama-direct-pilot.yaml` and
`ollama-few-shot-pilot.yaml`. They preserve the same prompts, demonstrations, pilot, schema, seed,
and evaluator while pinning the local Ollama version, loopback endpoint, exact model digest, runtime
options, and CPU execution. Both use `results/cache/ollama-responses/`, separate from OpenAI.

```text
python -m verilogic_ns_api.baselines ollama-smoke
python -m verilogic_ns_api.baselines plan --config experiments/configs/ollama-direct-pilot.yaml
python -m verilogic_ns_api.baselines plan --config experiments/configs/ollama-few-shot-pilot.yaml
python -m verilogic_ns_api.baselines run --config experiments/configs/ollama-direct-pilot.yaml --mode live
python -m verilogic_ns_api.baselines run --config experiments/configs/ollama-few-shot-pilot.yaml --mode live
```

Local execution requires no API key, paid-use flag, or external data transfer. See
`docs/LOCAL_LLM_BASELINE.md` for the signed installation, cloud-disable, hardware/model selection,
canary, replay, and interpretation protocol. The existing OpenAI configurations remain optional,
implemented, and mocked, but not operationally verified.
