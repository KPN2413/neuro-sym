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

Rules must be range-restricted before execution: every variable in the head must occur in a positive body literal, and every variable use must be declared. Later semantic validation will define the exact safe treatment of variables appearing only in explicitly negative body literals. Until then, unsafe rules are invalid rather than guessed.

## Forward-chaining semantics

The future engine begins with all validated facts, repeatedly applies every rule under substitutions whose body literals are already known, and adds new ground head literals until no new literal can be added. The finite domain and lack of function symbols guarantee termination for validated finite theories.

Derivations are monotonic within a theory: adding a derived literal does not retract earlier literals. Provenance is accumulated independently of traversal order so the decision is deterministic even when multiple proofs exist.

## Query classification

Let `q` be the normalized query and `opposite(q)` be the same atom with the `negated` flag reversed.

| Derivable `q` | Derivable opposite | State |
|---|---|---|
| yes | no | `ENTAILED` |
| no | yes | `CONTRADICTED` |
| no | no | `UNKNOWN` |
| yes | yes | internal `INCONSISTENT` |

Structural or semantic rejection yields internal `INVALID` before reasoning. Parser uncertainty can also cause abstention without invoking the engine.

`UNKNOWN` never means false. `CONTRADICTED` requires a derivation of the explicit opposite.

## Source-linked proof requirements

Facts, rules, their literals, and the query carry source IDs. A proof for an accepted answer must contain only validated facts and rule applications, preserve literal polarity and argument order, link supporting leaves and rules to their source statements, and be replayable deterministically. `UNKNOWN` requires a structured explanation of non-derivability boundaries rather than a fabricated proof.

## Meaning preservation

Structural validity does not prove that an AST correctly represents its source sentence. Future semantic checks will compare predicates, entities, polarity, argument order, variable binding, rule direction, and query intent. A correction may not invent a fact, reverse a rule, change polarity, or silently substitute an entity. Unresolved ambiguity must abstain.
