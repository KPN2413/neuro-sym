# Experiment Protocol

## Status

Phase 3 completed a frozen zero-cost local direct/few-shot pilot over the Phase 2 runner; the hosted OpenAI path remains mocked but operationally unverified. Phase 4 adds deterministic symbolic oracle-structure conformance. Phase 5 adds a frozen local natural-language-to-AST pilot with no correction. These pilot and conformance measurements are engineering evidence, not final capstone results.

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

Proofs are accepted only after replay against the validated AST. Phase 4 checks conclusion/status,
exact source facts and text, grounded rule applications, complete substitutions, antecedent
availability/order, source-ID validity, polarity, depth, hash integrity, reachability, and
acyclicity. A separately implemented naive closure validates even empty `UNKNOWN` proofs.
Provider-authored proof prose is never accepted as formal evidence.

The Phase 4 conformance protocol uses only ProofWriter's existing formal S-expressions, only the OWA
development split, and no natural-language parsing. The balanced oracle-structure sample contains 20
examples in every depth (0/1/2/3/5) by label (ENTAILED/CONTRADICTED/UNKNOWN) cell, for 300 total.
The same-30 check reuses the exact frozen Phase 3 development IDs. Report these as symbolic-ceiling
results, never semantic-parser or end-to-end results. Raw records and reports remain ignored locally.

## Reporting

Separate confirmatory from exploratory analyses. Publish configuration and failed-run counts alongside successful results. Do not infer factual truth from logical entailment, do not hide null or negative findings, and do not claim causation beyond controlled ablations.

## Phase 5 semantic-parser protocol

Prompt development used synthetic inputs and six training examples only. The theory prompt, query
prompt, output schemas, runtime, calibration manifest, and exact Phase 3 30-example development
manifest were hash-frozen before any Phase 5 development call. The parser receives only neutral
source IDs plus natural-language statements/query. Formal fields, labels, proofs, depth, raw keys,
and test records are excluded. One theory request is reused across questions; query parsing is
separate. Phase 5 permits no semantic repair, feedback, reflection, voting, or solver-guided retry.

Report structural validity, source coverage, exact theory/query accuracy, canonical statement and
closure precision/recall/F1, component/construction accuracy, complete-pipeline classification,
coverage/selective risk, error taxonomy, tokens, latency, and cache use. Parser failures count as
`ERROR`, never `UNKNOWN`. The frozen result may not be used to revise Phase 5 prompts.

## Phase 6 validation/correction protocol

Phase 6 replays the frozen Phase 5 raw cache, converts deterministic failures to bounded typed
feedback, applies a local semantic critic, and permits one complete replacement correction per
theory and per query. Corrected outputs re-enter the unchanged Phase 5 validators and the Phase 4
reasoner/verifier. P0 reproduces raw Phase 5, P1 answers deterministically valid corrected outputs,
and P2 additionally requires critic acceptance. P1/P2 share candidates and calls.

Critic/correction prompt development and controlled corruptions use synthetic or training data only.
All controller artifacts and the evidence-gate policy are hash-frozen before development evaluation.
Gold ASTs, labels, proofs, depth, closure, and oracle results cannot enter requests or decisions.
Evaluation against gold happens only after final component decisions. The valid logical label
`UNKNOWN` remains distinct from deliberate `ABSTAIN` and infrastructure `ERROR`.
