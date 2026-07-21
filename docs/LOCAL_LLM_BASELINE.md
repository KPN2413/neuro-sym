# Local Open-Weight LLM Baseline

## Operational status

Phase 3 uses Ollama for the operational direct and fixed few-shot pilot. Inference is fully
local, requires no API key or provider account, incurs an API cost of USD 0.00, and sends no
ProofWriter material to a hosted provider. The existing OpenAI Responses implementation remains
available for engineering comparison, but it is implemented and mocked, not operationally
verified. It was not called during the local pilot.

This is a **local open-weight LLM baseline**, not a GPT-5.6 Terra result and not the later
neuro-symbolic system.

## Hardware-aware selection

The qualification machine has an 11th-generation Intel Core i5-1135G7 with eight logical
processors, 7.79 GiB RAM, and integrated Intel Iris Xe graphics with shared memory. No NVIDIA
device or CUDA runtime is available. Before download, the system drive had 17.23 GiB free. Ollama
process metadata reports `size_vram: 0`, so the frozen execution device is CPU.

The selection policy rejected:

- `gpt-oss:20b`: below the 24 GiB RAM, approximately 16 GiB accelerator-memory, and 25 GiB
  pre-download disk thresholds;
- `qwen3.5:9b`: below the approximately 16 GiB RAM threshold.

The strongest safe tier was `qwen3.5:4b-q4_K_M`, which fits the approximately 8 GiB RAM and 8 GiB
free-disk policy. Only this one model was downloaded.

## Frozen runtime identity

- Ollama: `0.32.1`, installed from the official signed Windows distribution
- Endpoint: `http://127.0.0.1:11434`
- Cloud mode: disabled with `OLLAMA_NO_CLOUD=1`
- Model tag: `qwen3.5:4b-q4_K_M`
- Model digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- Local model size: 3,389,983,735 bytes
- Model family/quantization: Qwen 3.5, 4.7B reported locally, Q4_K_M
- Licence: Apache License 2.0, verified in the official model registry and local model metadata
- Context: 4096 tokens
- Output limit: 128 tokens
- Temperature: 0
- Sampling seed: 20260713
- Thinking: disabled; non-empty thinking content is rejected before caching
- Concurrency: 1
- Keep-alive: 30 minutes
- Timeout: 600 seconds
- Execution device: CPU

The official sources used were the [Ollama Windows guide](https://docs.ollama.com/windows),
[native chat API](https://docs.ollama.com/api/chat),
[structured-output guide](https://docs.ollama.com/capabilities/structured-outputs),
[local-only FAQ](https://docs.ollama.com/faq), and the
[`qwen3.5:4b-q4_K_M` registry entry](https://ollama.com/library/qwen3.5:4b-q4_K_M).

The server is started with an explicit IPv4-loopback binding and local-only cloud setting. The
provider independently rejects HTTPS, remote IP addresses, public names, proxies, cloud tags,
runtime-version drift, model-tag/digest drift, multiple installed models, and execution-device
drift. `httpx` environment proxy inheritance is disabled for local requests.

## Structured request and cache contract

The adapter uses native `POST /api/chat` with `stream: false`, no tools, no web access, the existing
strict JSON Schema in `schemas/llm-baseline-output.v1.schema.json`, and a second Pydantic validation
after receipt. Only `ENTAILED`, `CONTRADICTED`, or `UNKNOWN` is accepted. Prompts request the label
only. They do not request a proof, rationale, or chain of thought.

The local cache key includes provider, loopback endpoint identity, Ollama version, exact model tag
and digest, all model options, prompt and rendered-request hashes, schema hash, demonstration
manifest hash, example ID, and pilot-manifest hash. Ollama uses a separate ignored cache root from
OpenAI. Cache writes remain atomic, corrupt entries are quarantined, and replay has no inner
provider. Raw caches and model weights are never committed.

Ollama token counts and nanosecond durations are mapped into prediction telemetry: prompt tokens,
generated tokens, model-load time, prompt-evaluation time, generation time, total time, derived
generation throughput, retries, cache state, model digest/version, and execution device. Thinking
text is neither logged nor persisted.

## Reproduction

Install the official Windows application, configure Ollama to bind `127.0.0.1:11434`, set
`OLLAMA_NO_CLOUD=1`, restart Ollama, and pull exactly `qwen3.5:4b-q4_K_M`. Confirm the tag and digest
before running the repository commands below. No `OPENAI_API_KEY` is needed or used.

From the repository root with `backend/.venv` active:

```text
python -m verilogic_ns_api.baselines ollama-smoke
python -m verilogic_ns_api.baselines plan --config experiments/configs/ollama-direct-pilot.yaml
python -m verilogic_ns_api.baselines plan --config experiments/configs/ollama-few-shot-pilot.yaml
python -m verilogic_ns_api.baselines canary-plan --direct-config experiments/configs/ollama-direct-pilot.yaml --few-shot-config experiments/configs/ollama-few-shot-pilot.yaml
python -m verilogic_ns_api.baselines canary --direct-config experiments/configs/ollama-direct-pilot.yaml --few-shot-config experiments/configs/ollama-few-shot-pilot.yaml
python -m verilogic_ns_api.baselines run --config experiments/configs/ollama-direct-pilot.yaml --mode live
python -m verilogic_ns_api.baselines run --config experiments/configs/ollama-few-shot-pilot.yaml --mode live
python -m verilogic_ns_api.baselines run --config experiments/configs/ollama-direct-pilot.yaml --mode replay
python -m verilogic_ns_api.baselines run --config experiments/configs/ollama-few-shot-pilot.yaml --mode replay
```

The direct/few-shot config hashes are respectively
`13b53cb22b7a2f82f0de298771872bfea913dce2fcef93896e655a199392019b` and
`18a78122765bb603868542e416a610a775cafbabb9ded126d24a37419979bf6e`.

## Interpretation limits

CPU inference is slow: qualification measured about 9.5 seconds for the synthetic direct request
and 60.7 seconds for the synthetic few-shot request. The 30-record development pilot is a small,
preselected engineering/research pilot. It does not support significance, state-of-the-art, or
final test-set claims, and prompts/examples must not be tuned after observing it. ProofWriter's
dataset licence remains unverified; the model licence does not resolve that separate limitation.

Sanitized hardware, installation, canary, cache, and aggregate result evidence is kept under the
ignored `results/` tree. It contains no raw contexts, queries paired with IDs/labels, thinking text,
machine identity, personal path, credentials, or model weights.
