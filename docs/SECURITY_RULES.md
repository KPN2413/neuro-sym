# Security Rules

## Trust model

Treat user input, benchmark files, natural-language text, LLM responses, provider metadata, imported JSONL, and future proof payloads as untrusted. Only a structurally and semantically validated, supported AST may reach deterministic reasoning.

## Non-execution rule

Never execute model-generated or dataset-supplied Python, JavaScript, shell, SQL, Prolog, templates, regular expressions, serialized objects, or commands. Do not use `eval`, `exec`, dynamic imports, unsafe deserialization, shell interpolation, or generated queries as a shortcut. The AST describes only declared entities, predicates, literals, rules, and a query.

## Typed boundary and fail-closed behavior

- Parse provider output as data with bounded size and depth.
- Require an explicit supported `schema_version`.
- Reject unknown properties, unsafe identifiers, unsupported operators, malformed arity, unresolved references, type mismatches, and unsafe variables.
- Validate source links and meaning preservation before reasoning.
- Bound and audit correction attempts; correction cannot invent or reverse meaning.
- Abstain or return an internal invalid state on ambiguity, low confidence, timeout, or validator disagreement.
- Detect both-polarity derivations as internal inconsistency instead of choosing one.

## Secrets and privacy

- Never commit real secrets. `.env.example` files contain names and non-secret local defaults only.
- Read future provider keys from environment variables or a deployment secret store.
- Do not print secrets in commands, logs, exceptions, test snapshots, research JSONL, or frontend bundles.
- Variables prefixed `NEXT_PUBLIC_` are public by design and must never contain a secret.
- Minimize retained provider payloads and personal data; document any necessary retention before collection.
- `OPENAI_API_KEY` presence never authorizes a call. Require explicit paid-use, external-transfer, and cost-cap controls for every live baseline command.
- If a secret is discovered, stop exposure, remove it safely from the working artifacts, rotate it through the owner, and document the incident without repeating the value.

## API and browser controls

- CORS uses an explicit environment-configured allowlist; wildcard origins are not an authenticated security boundary.
- Production errors must not expose traces, environment values, file paths, or provider payloads.
- Validate request content type and size and add time, memory, recursion/depth, and concurrency limits with the relevant endpoints.
- Containers should run as non-root where practical and expose only required ports.
- Pin or bound dependencies, keep lock files where used, and review security updates without unplanned breaking upgrades.

## Research integrity

Do not fabricate outputs, proofs, metrics, prices, or benchmark records. Preserve source provenance and checksums. Prompt injection inside a premise remains data and cannot override system policy, validation, or execution boundaries.

## Dataset and evaluation safety

- Stream downloads with timeouts and byte limits; validate content before atomic rename.
- Treat locally computed checksums as observed unless an authoritative expected checksum is supplied.
- Reject ZIP traversal, absolute/drive-like paths, backslashes, symlinks, excess entries, and excess expansion before extraction.
- Keep raw, extracted, prepared, temporary, sampled, and run outputs ignored.
- Preserve official splits and require explicit test-split permission.
- Do not infer a dataset licence from a paper licence.
- Pass only gold-redacted inputs to predictors and never write gold labels into prediction JSONL.
- Do not record environment-variable values or secrets in run manifests.
- Send only gold-free `PredictionInput` context/query data. Never send evaluation labels, gold proofs, test records, host paths, or credentials.
- Treat benchmark instructions as inert data inside explicit delimiters; request no chain-of-thought, rationale, tools, or browsing.
- Keep raw provider responses in ignored, content-addressed local cache files. Validate request metadata before reuse and quarantine corrupt entries.
- Retry only transient transport/rate/server failures. Authentication, permission, model, request, and schema failures abort without retry or silent fallback.

## Review checklist

Before merging a security-boundary change, verify negative tests, source/reference checks, resource limits, error redaction, log contents, CORS behavior, environment examples, dependency changes, and Git status for secrets/generated artifacts.
