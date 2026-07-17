# Dataset card: ProofWriter in VeriLogic-NS

## Identity

- **Name:** ProofWriter updated RuleTaker datasets
- **Observed version:** V2020.12.3
- **Primary use:** evaluation of natural-language deductive reasoning with proofs
- **Primary VeriLogic-NS subset:** Open-World-Assumption main train/development/test JSONL files
- **Paper:** Tafjord, Dalvi, and Clark (2021), Findings of ACL-IJCNLP, DOI `10.18653/v1/2021.findings-acl.317`

## Provenance and integrity

The public S3 archive URL and observed HTTP/archive metadata are recorded in `provenance.observed.json`. The recorded SHA-256 was computed locally from the successfully validated archive. It is not a publisher-signed or publisher-listed checksum. The multipart S3 ETag is retained as HTTP metadata only and is not treated as a content checksum.

## Licence status

**Unverified for the dataset archive.** The inspected archive contains a README but no licence or citation file, and its README does not declare a dataset licence. The ACL Anthology licence for the paper must not be projected onto the dataset. Users are responsible for confirming permitted use with the dataset owner or their institution before redistribution or publication.

## Composition

The archive contains OWA and CWA directories. Observed variants include synthetic depth-bounded theories, extended mixtures, crowdsourced natural-language paraphrases, and a birds/electricity test set. Main JSONL records group a theory with structured triples, structured rules, multiple questions, labels, proof-depth metadata, and proof metadata.

The loader emits one `BenchmarkExample` per question while retaining the original theory ID, question ID, structured facts/rules, raw label, proof payload, source-relative path, record hash, and context-query hash.

## Label policy

For OWA records:

- Boolean `true` maps to `ENTAILED`.
- Boolean `false` maps to `CONTRADICTED` only when proof strategy and intermediate-proof metadata establish the opposite.
- String `Unknown` maps to `UNKNOWN`.

CWA false labels are rejected as ambiguous for the three-way contract rather than silently treated as contradiction.

## Splits and leakage

Official train, development, and test files are preserved exactly; VeriLogic-NS never resplits them. Test selection requires an explicit flag. Inspection reports duplicate stable IDs, qualified question IDs, exact context-query hashes, and theory IDs across splits. Reported overlap is not automatically removed or repaired.

## Ethical and research limitations

ProofWriter is a synthetic and semi-synthetic reasoning benchmark. Performance may not generalize to factual, open-domain, or naturally occurring reasoning. Benchmark correctness is relative to supplied premises. Smoke runs are engineering checks, not research performance claims. Raw data is excluded from Git because its licence is uncertain, redistribution is unnecessary, and the archive is large.
