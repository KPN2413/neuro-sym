# Prompt Protocol

Phase 3 prompt v1 is a frozen task definition stored as repository text, never a Python literal. Direct and few-shot copies are byte-identical. It defines `ENTAILED` as derivability of the query, `CONTRADICTED` as derivability of the explicit opposite, and `UNKNOWN` as derivability of neither under open-world semantics. Absence is never contradiction.

The renderer places each context/query JSON object inside `<untrusted_proofwriter_example>` delimiters and tells the model that benchmark text is data, not instructions. It excludes example gold, proofs, IDs, paths, host data, and secrets. The requested response is the strict versioned schema only; prompts forbid rationale, proof, explanation, hidden chain-of-thought, tools, browsing, and external knowledge.

Direct rendering appends no demonstrations. Few-shot rendering loads the six manifest-approved training examples from the local verified archive and encloses them in `<approved_training_demonstrations>`. Each demonstration contains input plus correct label only—never proof or reasoning. The committed manifest contains only selection/provenance metadata and normalized hashes, not ProofWriter text.

Prompt file SHA-256, schema SHA-256, demonstration-manifest hash, and the final rendered-request hash are part of every cache identity. Editing a prompt without updating its versioned config hash fails before dispatch. Development uses synthetic fixtures and training records; frozen pilot results must never be used to silently revise v1.

## Phase 5 parser prompts

The theory and query parser prompts are separately versioned under `prompts/semantic_parsing/`.
The theory renderer accepts only neutral `sentN` plus natural-language text; the query renderer
accepts only query text. Both wrap content as untrusted benchmark data. Neither renderer can accept
formal facts/rules, labels, proofs, depth, raw IDs, host data, or a complete benchmark record.

The theory output separates fact and rule arrays while preserving every neutral source exactly once.
The query output is one ground signed literal. Both reuse the Phase 4 term vocabulary, forbid prose,
answers, proofs, rationales, and chain-of-thought, and run with Ollama `think: false`. Prompt and schema
hashes are part of the parser cache identity. Calibration used training data only; the final identities
were recorded in `semantic-parser-freeze.v1.json` before development inference.

## Phase 6 critic and correction prompts

Phase 6 has four versioned prompt contracts: theory/query fidelity critics and theory/query complete
replacement correctors. Each receives a dedicated gold-free view wrapped as untrusted data. Critics
return only `ACCEPT`/`REVISE` with bounded source-linked issues. Correctors receive the original
neutral source, prior candidate, deterministic feedback, and optional critic issues, then return the
unchanged Phase 5 strict candidate schema. No prompt requests or persists rationale or
chain-of-thought. Prompt/schema/runtime hashes and the train-only calibration manifest are frozen in
`experiments/manifests/phase6-freeze.v1.json` before development evaluation.
