# Phase 6 Abstention Policy

`UNKNOWN`, `ABSTAIN`, `ERROR`, and `INCONSISTENT` are separate states:

- `UNKNOWN` is a valid open-world symbolic conclusion: neither query polarity is derivable.
- `ABSTAIN` is a deliberate system decision not to expose a benchmark answer because required evidence is missing.
- `ERROR` is an infrastructure, provider, software, or independent-verifier failure.
- `INCONSISTENT` remains supported by the general reasoner; under the three-label ProofWriter policy it becomes `ABSTAIN` with `UNEXPECTED_INCONSISTENCY`.

P2 answers only when theory and query structured outputs are valid, source coverage is complete,
semantic validation passes, the critic accepts both components, reasoning completes, and independent
verification succeeds. The evidence vector is observable and auditable; model self-confidence is
not a feature and no numeric probability is claimed.

Typed abstention reasons include invalid theory/query, incomplete coverage, critic rejection,
correction failure/limit/no-progress, resource limit, unexpected inconsistency, and a failed
reliability gate. Provider or verifier internal failures remain `ERROR`. A valid `UNKNOWN` must never
be converted to abstention merely because it lacks a positive proof.

Report overall accuracy together with coverage, answered-only accuracy, selective risk, ABSTAIN and
ERROR confusion columns, and the reason distribution. This prevents a conservative system from
appearing reliable merely by answering very little.
