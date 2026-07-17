# Experiment Protocol

## Status

Phase 3 implements frozen direct and fixed few-shot baseline infrastructure over the Phase 2 runner. No live provider pilot has been authorized or reported. Synthetic fake-provider metrics verify engineering behavior only.

## Units and data

ProofWriter is the primary benchmark. The Phase 2 adapter records source/version, observed HTTP and ZIP metadata, a locally observed SHA-256, and explicit licence uncertainty. Raw and normalized data remain ignored. Synthetic fixtures and training records may be used for implementation and prompt development. The frozen 30-record development pilot may be evaluated once only after prompts, demonstrations, settings, schema, and selection hashes are frozen. It is an engineering/research pilot, not final evidence. The test split remains untouched until a later final protocol is frozen.

Each unit contains source statements, a query, the benchmark label, available proof/depth metadata, and a stable example ID. Normalization must retain the original record or a verifiable reference to it.

Only main OWA `meta-{train,dev,test}.jsonl` files enter the three-way normalized contract. OWA `true` maps to `ENTAILED`; OWA `false` maps to `CONTRADICTED` only with the documented proof strategy and intermediate-proof evidence; OWA `Unknown` maps to `UNKNOWN`. CWA false is never treated as contradiction by default.

## Conditions

The mandatory conditions are direct LLM, few-shot LLM, and the end-to-end neuro-symbolic pipeline. Later ablations may disable one approved component at a time. Conditions must use the same eligible evaluation records and, where applicable, the same provider/model snapshot, decoding controls, and information budget. Few-shot demonstrations must come only from the training split. Phase 3 fixes exactly six: two per label, with one depth-0 and one depth-2 example per label. Direct and few-shot runs share model/settings, strict schema, ordered development IDs, and task definition; they differ only by the six demonstrations.

## Reproducibility record

Every run must record:

- run ID, timestamp, code commit, dirty-tree flag, and schema version;
- dataset manifest/checksum and split;
- condition, prompt version, model/provider identifiers, and decoding parameters;
- random seeds and retry policy;
- per-example raw decision, normalized decision, abstention/invalid state, latency, usage, and error class;
- AST and proof artifacts where applicable, with sensitive provider data excluded;
- pricing source/version used for any cost estimate.

JSONL is the canonical per-example output format. Aggregates must be reproducible from those records. SQLite may later index local run metadata but is not the canonical evidence source.

Predictors receive `PredictionInput`, a gold-redacted projection of `BenchmarkExample`. Run configurations include sampling seed/strategy, selected official splits, predictor identity/version, dataset manifest reference, and safe package/platform metadata. Environment-variable values are not captured.

## Execution controls

Use deterministic solver settings. Cache or replay provider outputs only when the condition and prompt identity match and the protocol marks the use. Bound retries, timeouts, concurrency, and maximum token budgets. Record failures rather than silently dropping examples. Never execute model-generated code.

Live baseline execution additionally requires explicit paid-API authorization, explicit external-data-transfer confirmation, and a preflight/during-run dollar cap. Only selected context/query text is sent for development predictions; evaluation gold labels, gold proofs, local paths, credentials, and test records are excluded. Transport retries resend the identical logical request. Ambiguous timeouts retain residual duplicate-billing risk because provider receipt may be unknowable.

## Metrics and analysis

Compute the metrics listed in `RESEARCH_QUESTIONS.md` from raw records. Phase 2 defines overall accuracy as correct answers divided by all examples, so abstentions/errors reduce it; answered-only accuracy uses answered examples only; coverage is answered/all; and selective risk is one minus answered-only accuracy. `ABSTAIN` and `ERROR` have distinct confusion-matrix columns and are never converted to `UNKNOWN`. Three-class macro precision/recall/F1 use explicit zero-division handling, with abstentions/errors contributing false negatives for their gold class. Report denominators, class distribution, invalid predictions, and missing data. Later research comparisons will use paired confidence intervals/tests declared before final results.

Cost values are estimates, not timeless facts. Store token counts separately from the price table so costs can be recomputed.

## Robustness

Approved perturbations must be meaning-preserving and versioned. Candidate groups include source-sentence permutation, bijective entity renaming, harmless formatting variation, and manually reviewed paraphrases. Evaluate original and transformed instances as pairs and report decision-invariance and accuracy deltas.

## Proof evaluation

Proofs are scored only after replay against the accepted AST. Planned checks include conclusion match, valid rule applications, antecedent availability, source-ID validity, polarity preservation, and source completeness. Provider-authored proof prose is never accepted as formal evidence.

## Reporting

Separate confirmatory from exploratory analyses. Publish configuration and failed-run counts alongside successful results. Do not infer factual truth from logical entailment, do not hide null or negative findings, and do not claim causation beyond controlled ablations.
