# Deterministic Symbolic Engine

## Scope

Phase 4 implements a finite, function-free, Datalog-style forward chainer over validated
`theory.v1` objects. It supports unary and binary predicates, named entities, universally scoped
rule variables, conjunctive non-empty bodies, constants in rules, positive literals, explicit
negative literals, multi-step derivations, positive recursion, and open-world query classification.
It does not parse ordinary language, call an LLM, use negation as failure, or perform
contraposition.

## Validation boundary

`Theory` Pydantic models mirror the versioned JSON contract and add semantic checks that JSON
Schema cannot express. They reject duplicate identifiers, dangling sources, unknown entities or
predicates, predicate arity conflicts, undeclared variables, optional type conflicts, and head
variables not bound in the body. Facts and queries must be ground. Invalid inputs never enter the
engine and are not silently repaired.

Explicitly negative body literals are ordinary signed premises. They bind variables only when a
matching negative fact or conclusion is present. Missing positive evidence is not negative
evidence.

## Fixpoint algorithm

The engine starts from the set of validated ground facts. In each round it indexes the complete
closure by predicate, arity, and polarity, then evaluates canonicalized rule bodies. At least one
premise of a rule instance must come from the preceding delta, which avoids needlessly repeating
old combinations while preserving the least fixpoint. New ground heads are added monotonically.
The finite entity domain and absence of function symbols guarantee termination under normal
limits, including for positive recursive cycles.

Matching preserves ordered arguments and shared-variable equality. A repeated variable such as
`related(X, X)` matches only facts with identical arguments. Constants and variables may be mixed,
and substitutions are sorted before proof construction.

## Result semantics

For normalized query `q`, the engine separately tests `q` and the literal obtained by flipping only
its explicit polarity:

| `q` derivable | opposite derivable | Status |
|---|---|---|
| yes | no | `ENTAILED` |
| no | yes | `CONTRADICTED` |
| no | no | `UNKNOWN` |
| yes | yes | `INCONSISTENT` |

This is query-specific, paraconsistent behavior. A conflict elsewhere is reported in telemetry but
does not make an unrelated query inconsistent, and a contradiction does not explode into arbitrary
conclusions. `UNKNOWN` is a complete least-fixpoint result, not falsity or abstention. The Phase 2
benchmark metric vocabulary remains three-class; `INCONSISTENT` is the symbolic engine's fourth
status for inputs containing support for both sides.

## Determinism and limits

Facts, rules, body literals, matches, closure entries, proof nodes, and substitutions have explicit
canonical ordering. Duplicate facts are idempotent. When several proofs derive the same literal,
selection prefers, in order: lower depth, fewer proof nodes, lower rule ID, lexicographically sorted
substitution and premise IDs, then a canonical hash. Reordering equivalent input arrays therefore
does not change the logical result or chosen proof.

Configurable safeguards bound derived literals, considered rule instances, productive rounds,
proof nodes, and optionally elapsed execution time. Exceeding a limit raises a typed
`ResourceLimitError`; the engine never converts an incomplete computation to `UNKNOWN`.

## Commands

With the backend environment active, run from the repository root:

```text
python -m verilogic_ns_api.reasoning --help
python -m verilogic_ns_api.reasoning reason --input examples/theories/entailed.json --human
python -m verilogic_ns_api.reasoning saturate --input examples/theories/binary-join.json
python -m verilogic_ns_api.reasoning inspect-closure --input examples/theories/inconsistent.json
python -m verilogic_ns_api.reasoning verify-proof --theory examples/theories/entailed.json --proof results/reasoning/entailed.json
python -m verilogic_ns_api.reasoning conformance-run --data-source datasets/proofwriter/raw/archives/proofwriter-dataset-V2020.12.3.zip --variant depth-5 --per-cell 20 --seed 20260713 --output results/reasoning/conformance300.json
```

JSON outputs are written atomically, may only be placed beneath the current working directory, and
cannot replace an existing file without `--force`. Exit codes distinguish invalid input (`2`),
resource limits (`3`), invalid proofs (`4`), and other reasoning errors (`5`).

## ProofWriter conformance

The deterministic adapter reads ProofWriter's existing formal S-expression fields. It does not
infer logic from English. It maps `+` to positive polarity and the dataset's `-`/`~` markers to
explicit negative polarity, converts only the OWA development split, and refuses the test split.

The Phase 4 balanced conformance set has 300 examples: 20 for every combination of depths
0/1/2/3/5 and labels ENTAILED/CONTRADICTED/UNKNOWN. The observed result was 300/300 classifications
and 300/300 independently verified proofs. The same 30 development examples frozen for Phase 3
also produced 30/30 and 30/30. These are oracle-structure symbolic-ceiling checks, not evidence that
the future natural-language semantic parser is accurate and not a final capstone comparison.

ProofWriter's dataset licence remains unverified, and raw records/results remain ignored locally.
