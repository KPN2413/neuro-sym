# Typed AST Schema

## Contract

`schemas/theory.v1.schema.json` is the Phase 1 structural contract for a normalized theory. It uses JSON Schema Draft 2020-12, fixes `schema_version` to `1.0`, rejects unknown properties, and contains no field capable of requesting executable code.

Top-level fields are:

- `schema_version`: contract version, currently `1.0`;
- `theory_id`: safe stable identifier;
- `source_statements`: source IDs and non-empty natural-language text;
- `entities`: entity IDs, display labels, and optional types;
- `predicates`: predicate names, arities, and optional argument types;
- `facts`: ground positive or explicitly negative literals;
- `rules`: declared variables, conjunctive body, one head, and source link;
- `query`: one ground literal with a source link;
- `parser_metadata`: optional restricted provenance/confidence metadata.

## Terms and identifiers

Identifiers use conservative ASCII patterns. Theory, source, entity, rule, and optional type identifiers begin with a letter and contain only letters, digits, `_`, `-`, or `.` where the relevant definition permits them. Predicate names are lower snake-style identifiers. Variables begin with an uppercase letter.

Terms are tagged objects rather than free-form expressions:

```json
{ "kind": "entity", "id": "bob" }
```

```json
{ "kind": "variable", "name": "X" }
```

Facts and queries accept only entity terms. Rule body and head literals may use entity or variable terms. Literal arguments are ordered and limited to one or two items. The `negated` Boolean represents explicit negation; omission and null are not alternative meanings.

## Source integrity

Every fact, rule, rule-body literal, rule-head literal, and query requires a `source_id`. The schema ensures the field is present and syntactically safe. JSON Schema cannot portably enforce that an arbitrary source ID appears in the root array, so the future semantic validator must check referential integrity and reject dangling references.

## Structural versus semantic validation

The schema enforces supported shapes, fixed operators, allowed fields, basic arity bounds, and identifier safety. Semantic validation must additionally enforce:

- uniqueness and existence of all IDs and references;
- declared predicate existence and exact arity consistency;
- optional argument-type compatibility;
- entity and variable declaration before use;
- rule range restriction and variable safety;
- source-statement reference integrity;
- preservation of source meaning and query direction;
- future schema-version compatibility policy.

Passing the JSON Schema alone never authorizes reasoning when semantic validation is required.

## Fixtures

Valid examples are in `examples/theories/entailed.json`, `contradicted.json`, and `unknown.json`. They demonstrate the three public outcomes conceptually but Phase 1 does not execute a reasoner. Intentionally invalid fixtures cover an argument list outside supported arity, a missing source-link field, an unsafe identifier, and an unsupported executable-code property.

Backend tests load the schema with a Draft 2020-12 validator, require every valid fixture to pass, and require every invalid fixture to fail for its intended structural reason.

## Versioning

Existing versioned schemas are immutable public contracts. A breaking field, meaning, or validation change requires a new schema file and a decision record describing migration. Additive changes are allowed only if they preserve fail-closed behavior; because unknown properties are rejected, most additions still require a deliberate version review.

## Phase 2 benchmark-example contract

`schemas/benchmark-example.v1.schema.json` is a separate Draft 2020-12 contract for normalized benchmark evidence. It is generated from the strict Pydantic `BenchmarkExample` model and checked byte-for-byte in tests so model/schema drift fails CI.

The contract preserves a stable example identifier, dataset version and variant, official split, theory and question identifiers, reasoning depth, source text, normalized query and context, structured source fields when present, the three-way gold label, raw label, world assumption, source-relative provenance, and opaque gold proof payload. Unknown fields are rejected.

This research-data contract does not replace `theory.v1.schema.json`: the benchmark example records what the dataset supplied, while the typed AST remains the future parser/reasoner boundary. Before prediction, the evaluator derives a gold-redacted `PredictionInput`; that narrower runtime view deliberately has no gold label, raw label, or proof field.
