# Research Questions

No answer or improvement is assumed in advance. These questions define what later controlled experiments will measure.

## Primary questions

### RQ1: Decision accuracy

How does end-to-end VeriLogic-NS accuracy on the selected ProofWriter evaluation set compare with direct and few-shot LLM baselines under matched model and data conditions?

### RQ2: Logical reliability

How often does each condition produce unsupported, contradictory, or malformed answers, and how do error rates vary with proof depth and query polarity?

### RQ3: Proof quality

For decisions that should be supported or contradicted, what proportion of generated symbolic proofs are structurally valid, source-complete, and replayable against the accepted AST?

### RQ4: Validation and abstention

How do structural validation, semantic validation, limited correction, and confidence-gated abstention change coverage, selective accuracy, and the severity of remaining errors?

### RQ5: Robustness

How sensitive are baselines and the neuro-symbolic pipeline to meaning-preserving perturbations such as sentence order, entity renaming, and approved paraphrase sets?

### RQ6: Efficiency

What latency, token usage, and estimated provider cost does each condition incur, and how are these quantities distributed rather than only averaged?

## Planned comparisons

- direct LLM answer;
- few-shot LLM answer;
- validated semantic parsing plus deterministic reasoning;
- later ablations that remove one approved validation, correction, or abstention control at a time.

The primary benchmark is ProofWriter. FOLIO and multiple-provider comparisons are out of scope unless explicitly approved.

## Metrics

Planned metrics include exact decision accuracy, macro-F1 where class balance warrants it, per-class precision/recall, accuracy by proof depth, malformed-output rate, contradiction/inconsistency rate, abstention coverage, selective accuracy/risk, proof validity and source-link completeness, robustness deltas, latency percentiles, token counts, and cost estimates based on recorded pricing metadata.

Confidence intervals and paired comparisons will be selected in `EXPERIMENT_PROTOCOL.md` before observing final results. Negative and null findings must be reported.

## Threats to validity

Key risks include benchmark contamination, semantic-parser meaning drift, prompt sensitivity, provider nondeterminism, unequal information across conditions, cost-price changes, incomplete proof scoring, and conclusions that do not generalize beyond the benchmark fragment. Logical correctness relative to premises must never be presented as factual-world verification.

## Phase 6 pre-registered questions and hypotheses

These questions and directional hypotheses were frozen after synthetic/train-only calibration and
before the Phase 6 development pilot was inspected. They are research expectations, not acceptance
criteria, and null or negative findings must be retained.

- **RQ6.1:** Can deterministic validator feedback improve neural semantic-parser validity?
- **RQ6.2:** Can a semantic critic detect meaning mismatches that structural validation misses?
- **RQ6.3:** Can correction increase end-to-end coverage without reducing answered-only reliability?
- **RQ6.4:** Can abstention prevent invalid or semantically doubtful parses from becoming confident logical answers?
- **H6.1:** Validation-guided correction will increase complete AST acceptance above Phase 5's 13.33% rate.
- **H6.2:** The corrected pipeline will improve end-to-end coverage relative to Phase 5.
- **H6.3:** Selective abstention will have equal or higher answered-only accuracy than answering every deterministically valid corrected output.
