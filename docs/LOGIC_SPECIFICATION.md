# Logic Specification

## Supported language

The MVP logic is a finite, function-free Datalog-style fragment:

- constants are declared entity identifiers;
- variables are declared within a rule;
- predicates have arity one or two;
- literals are a predicate applied to ordered terms, with an explicit `negated` flag;
- facts are ground literals;
- rules have one or more conjunctive body literals and exactly one head literal;
- queries are ground literals.

Functions, arithmetic, equality, disjunction, existential rule heads, default negation, negation-as-failure, quantifier syntax, aggregation, side effects, and executable expressions are unsupported.

## Terms, literals, and polarity

An entity term refers to a declared entity. A variable term refers to a variable declared by its containing rule. The ordered number of arguments must match the predicate definition. Positive `blue(ava)` and explicitly negative `not blue(ava)` are distinct, opposing literals.

Explicit negation is classical evidence supplied or derived by the theory. It is not inferred from absence. This distinction is mandatory for open-world reasoning.

## Rules

A rule is interpreted as a universally quantified Horn implication over its declared variables:

```text
body_1 AND ... AND body_n -> head
```

Rules must be range-restricted before execution: every variable in the head must occur in at least one
body literal, and every variable use must be declared. Positive and explicitly negative body
literals are both ordinary signed relations and may bind variables only through matching known
literals. This is not negation as failure. Unsafe rules are invalid rather than guessed.

## Forward-chaining semantics

The engine begins with all validated facts, repeatedly applies every rule under substitutions whose body literals are already known, and adds new ground head literals until no new literal can be added. A delta set ensures each considered instance has a newly available premise. The finite domain and lack of function symbols guarantee termination for validated finite theories.

Derivations are monotonic within a theory: adding a derived literal does not retract earlier literals. Provenance is accumulated independently of traversal order so the decision is deterministic even when multiple proofs exist.

## Query classification

Let `q` be the normalized query and `opposite(q)` be the same atom with the `negated` flag reversed.

| Derivable `q` | Derivable opposite | State |
|---|---|---|
| yes | no | `ENTAILED` |
| no | yes | `CONTRADICTED` |
| no | no | `UNKNOWN` |
| yes | yes | `INCONSISTENT` |

Structural or semantic rejection occurs before reasoning. Parser uncertainty in a future phase can
also cause abstention without invoking the engine. Resource exhaustion is a typed failure and is
never reported as `UNKNOWN`.

`UNKNOWN` never means false. `CONTRADICTED` requires a derivation of the explicit opposite.

## Source-linked proof requirements

Facts, rules, their literals, and the query carry source IDs. A proof for an accepted answer contains only validated facts and rule applications, preserves literal polarity and argument order, links supporting leaves and rules to their source statements, and is replayable deterministically. `UNKNOWN` has no support/opposition root and an empty DAG rather than a fabricated proof. An independent naive closure verifies that neither side is derivable.

## Meaning preservation

Structural validity does not prove that an AST correctly represents its source sentence. Future semantic checks will compare predicates, entities, polarity, argument order, variable binding, rule direction, and query intent. A correction may not invent a fact, reverse a rule, change polarity, or silently substitute an entity. Unresolved ambiguity must abstain.
