# Symbolic theory examples

The top-level JSON files are valid `theory.v1` inputs covering entailment, contradiction,
open-world unknown, inconsistency, multi-step unary reasoning, binary joins, explicit-negative
premises, recursion, and canonical selection between multiple derivations.

Files under `invalid/` intentionally violate either the structural JSON Schema or the Phase 4
semantic validation boundary. They are test inputs and must never reach the reasoning engine.

Expected valid-fixture decisions:

| Fixture | Expected status | Main behavior |
|---|---|---|
| `entailed.json` | `ENTAILED` | positive support |
| `contradicted.json` | `CONTRADICTED` | explicit opposite support |
| `unknown.json` | `UNKNOWN` | open-world absence |
| `inconsistent.json` | `INCONSISTENT` | both query polarities |
| `unary-multistep.json` | `ENTAILED` | depth-two chain |
| `binary-join.json` | `ENTAILED` | shared-variable join |
| `explicit-negative-premise.json` | `ENTAILED` | signed negative body |
| `recursive-cycle.json` | `ENTAILED` | finite positive recursion |
| `multiple-derivations.json` | `ENTAILED` | canonical proof selection |

Run a fixture from the repository root with the backend environment active:

```text
python -m verilogic_ns_api.reasoning reason --input examples/theories/unary-multistep.json --human
```
