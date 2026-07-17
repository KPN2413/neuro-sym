# Results

Generated experiment outputs are written beneath `results/runs/` and ignored by Git. Each Phase 2 run is first written to a unique `.incomplete` directory and promoted atomically only after predictions, metrics, and the final manifest are durable.

Phase 3 local Ollama payloads remain only in ignored `results/cache/ollama-responses/`
entries; the unexecuted OpenAI cache has its separate `results/cache/llm-responses/` root.
Rendered requests are reconstructed locally from frozen prompts and the ignored dataset. Synthetic
smoke, canary, aggregate, run, replay, and comparison evidence stays ignored. Cache references in
prediction rows are relative and contain no credential. Replay validates the full request identity
and has no inner provider, so it cannot make an inference call.

The repository contains no fabricated or placeholder measurements. Real local Phase 3 measurements
are retained only in the ignored results tree according to repository policy; no raw cache, model
weight, or record-level result is committed. Reproduce local checks with the commands in
`docs/LOCAL_LLM_BASELINE.md`. A local run requires the locally acquired ProofWriter archive and the
exact pinned Ollama model, but no paid-use flag, provider account, external-transfer approval, or API
key. Those explicit gates still apply to the optional, operationally unverified OpenAI path.
