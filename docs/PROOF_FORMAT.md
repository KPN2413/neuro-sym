# Proof Format

## Contract

`schemas/proof.v1.schema.json` is generated from the strict `ProofDAG` Pydantic model. A proof is a
versioned, canonical JSON DAG bound to the SHA-256 hash of one normalized theory. It records the
query, result status, optional support/opposition roots, sorted proof nodes, and a hash of the entire
proof payload. `schemas/reasoning-result.v1.schema.json` wraps the decision, conflict metadata,
proof, and execution telemetry.

There are three node kinds:

- `source_fact`: exact signed ground literal, source ID/text, and depth zero;
- `rule_application`: rule/source identity, sorted substitution, ordered premise-node IDs, grounded
  conclusion, and derived depth;
- `derived_literal`: the signed conclusion and the rule-application node that produced it.

An entailed result has a support root. A contradicted result has an opposition root proving the
query's explicit complement. An inconsistent result has both. Unknown has neither root and an empty
node set; it never invents a proof of non-derivability.

## Canonical hashing

All hashes use SHA-256 over UTF-8 JSON serialized with sorted object keys and compact separators.
Theory hashing additionally sorts source statements, entities, predicates, facts, rules, variables,
and rule-body literals without changing ordered predicate arguments. Node IDs hash the complete
node payload excluding `node_id`; the proof hash covers every field except `proof_hash`.

Changing a source sentence, polarity, argument, rule ID, substitution, premise, result root, or
theory changes a protected hash. Hashes provide deterministic identity and tamper evidence, not a
digital signature or proof of premise truth.

## Independent verification

`ProofVerifier` does not trust the producing engine. It checks:

- exact theory/query/hash binding and canonical node order;
- unique node IDs, node hashes, and proof hash;
- root presence and signed conclusion required by the claimed status;
- exact source-fact and source-text correspondence;
- rule/source identity, complete declared substitution, entity validity, premise count/order,
  grounded premises/head, and depth;
- referenced-node existence, acyclicity, and absence of unreachable injected nodes;
- conflict counts and the claimed result against a separately implemented naive least-fixpoint
  closure.

Any mismatch raises a typed `ProofVerificationError`; malformed or partially valid proofs never
become accepted conclusions. Tests cover outer-hash tampering, node tampering, semantic source
tampering with recomputed hashes, theory substitution, and fabricated `UNKNOWN` claims.
