# Local Neural Semantic Parser

## Scope

Phase 5 adds a research semantic parser that converts restricted ProofWriter-style English into the
existing Phase 4 typed AST. It uses only the locally installed Ollama model
`qwen3.5:4b-q4_K_M` at digest
`2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` through
`http://127.0.0.1:11434`. API cost is USD 0.00 and no benchmark text leaves the machine.
The model licence is Apache License 2.0, as already verified from the official registry and local
model metadata during Phase 3 qualification.

The model is a parser, not a reasoner. It never receives or predicts benchmark labels and never
produces a proof. A parsed candidate must pass strict Pydantic structure, complete source coverage,
predicate arity, entity/variable, safe-rule, and full Phase 4 `Theory` validation before the
deterministic forward chainer can use it.

## Gold isolation

The public theory request contains only `sent1`, `sent2`, ... and the corresponding natural-language
sentences. The public query request contains only query text. Separate internal bindings restore the
original source IDs after inference. Prompt renderers cannot accept `BenchmarkExample`,
`PredictionInput`, formal representations, labels, proofs, reasoning depth, record paths, or raw
ProofWriter keys. Tests assert this boundary.

One theory request is made per unique theory and one query request per question. Theory responses
are reused across questions from the same theory. The provider is configured with `think: false`;
any returned thinking content is a structured-output error and is never logged or cached.

## Contracts and failure semantics

The versioned output schemas are:

- `schemas/neural-theory-output.v1.schema.json`: separate fact and rule arrays with exactly one
  source-linked candidate for every input sentence;
- `schemas/neural-query-output.v1.schema.json`: one ground signed unary/binary query literal.

Candidate terms reuse Phase 4 `EntityTerm`, `VariableTerm`, and `VariableDefinition` types. The
deterministic converter creates Phase 4 facts, rules, declarations, source links, and parser metadata.
Every accepted reasoning result is independently checked by the Phase 4 proof verifier before its
label is reported. No model-generated code or solver expression is executed.

Outcomes are `PARSED`, `PROVIDER_ERROR`, `STRUCTURED_OUTPUT_ERROR`, `STRUCTURAL_INVALID`,
`SEMANTIC_INVALID`, `SOURCE_COVERAGE_ERROR`, `RESOURCE_LIMIT`, and `TIMEOUT`. Every parser failure
becomes evaluation `ERROR`. It is never silently mapped to the valid open-world answer `UNKNOWN`.

Phase 5 contains no repair prompt, self-reflection, majority vote, confidence gate, or solver
feedback. Only an identical request may be retried for a transient local transport failure. Those
features remain Phase 6 work.

## Reproducible commands

Run from the repository root with `backend/.venv` active:

```text
python -m verilogic_ns_api.semantic_parsing --help
python -m verilogic_ns_api.semantic_parsing plan --config experiments/configs/ollama-semantic-parser-pilot.yaml
python -m verilogic_ns_api.semantic_parsing run --config experiments/configs/ollama-semantic-parser-pilot.yaml --dataset calibration --run-id RUN_ID
python -m verilogic_ns_api.semantic_parsing run --config experiments/configs/ollama-semantic-parser-pilot.yaml --dataset pilot --run-id RUN_ID
python -m verilogic_ns_api.semantic_parsing replay --config experiments/configs/ollama-semantic-parser-pilot.yaml --dataset pilot --run-id REPLAY_ID
python -m verilogic_ns_api.semantic_parsing evaluate --run results/semantic-parsing/RUN_ID
```

`plan` validates the archive, manifests, prompts, schemas, model identity, and request count without
calling Ollama. `replay` constructs no provider and fails closed on a cache miss. Parser cache entries
live only under ignored `results/cache/semantic-parser/`; raw candidates and per-record outputs remain
under ignored `results/semantic-parsing/`.

## Frozen protocol and limitations

The train-developed protocol is recorded in
`experiments/manifests/semantic-parser-freeze.v1.json`. Calibration used six training records only.
The frozen pilot reused the exact 30 OWA development examples from Phase 3 (two per label at depths
0, 1, 2, 3, and 5) and never used the test split. Prompts and examples were not changed after the
freeze.

The 4B CPU model often emits structurally valid but semantically wrong rules. Phase 5 reports these
errors rather than repairing them. Consequently, low coverage and low overall accuracy are expected
and are an important result, not a reason to hide failures. ProofWriter's dataset licence remains
unverified, and the pilot is too small for significance or state-of-the-art claims.
