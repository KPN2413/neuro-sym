# Prompt Protocol

Phase 3 prompt v1 is a frozen task definition stored as repository text, never a Python literal. Direct and few-shot copies are byte-identical. It defines `ENTAILED` as derivability of the query, `CONTRADICTED` as derivability of the explicit opposite, and `UNKNOWN` as derivability of neither under open-world semantics. Absence is never contradiction.

The renderer places each context/query JSON object inside `<untrusted_proofwriter_example>` delimiters and tells the model that benchmark text is data, not instructions. It excludes example gold, proofs, IDs, paths, host data, and secrets. The requested response is the strict versioned schema only; prompts forbid rationale, proof, explanation, hidden chain-of-thought, tools, browsing, and external knowledge.

Direct rendering appends no demonstrations. Few-shot rendering loads the six manifest-approved training examples from the local verified archive and encloses them in `<approved_training_demonstrations>`. Each demonstration contains input plus correct label only—never proof or reasoning. The committed manifest contains only selection/provenance metadata and normalized hashes, not ProofWriter text.

Prompt file SHA-256, schema SHA-256, demonstration-manifest hash, and the final rendered-request hash are part of every cache identity. Editing a prompt without updating its versioned config hash fails before dispatch. Development uses synthetic fixtures and training records; frozen pilot results must never be used to silently revise v1.
