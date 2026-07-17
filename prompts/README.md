# Prompts

Phase 3 stores immutable task definitions at `baselines/direct/v1.md` and `baselines/few_shot/v1.md`. Their byte-identical v1 text defines the three OWA labels, declares benchmark material untrusted, and requests only the strict label—never a rationale or chain-of-thought. The direct renderer adds no example; the few-shot renderer loads exactly six approved training examples locally from the hash-checked manifest. No real ProofWriter text is committed. See `docs/PROMPT_PROTOCOL.md`.
