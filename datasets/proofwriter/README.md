# ProofWriter dataset integration

VeriLogic-NS uses the ProofWriter Open-World-Assumption data as its primary benchmark. Phase 2 adds safe acquisition, streaming normalization, deterministic development sampling, leakage reporting, and model-independent evaluation. Raw and normalized examples are never committed.

## Source and citation

- Archive: `https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip`
- Observed archive version: `V2020.12.3`
- Paper: Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. 2021. “ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language.” Findings of ACL-IJCNLP 2021. DOI `10.18653/v1/2021.findings-acl.317`.

The archive README identifies both `OWA/` and `CWA/`. It does not state a dataset licence and the archive contains no separate licence file. The ACL paper’s publication licence is not assumed to license the dataset archive. Dataset reuse therefore remains subject to licence uncertainty and institutional review.

## Observed layout

The verified ZIP contains a single top-level `proofwriter-dataset-V2020.12.3/` directory. Both OWA and CWA contain `depth-0`, `depth-1`, `depth-2`, `depth-3`, `depth-5`, `depth-3ext`, `NatLang`, `depth-3ext-NatLang`, and `birds-electricity` variants. Main files are named `meta-train.jsonl`, `meta-dev.jsonl`, and `meta-test.jsonl`; `birds-electricity` has only a main test split. Stage and abductive files are present but are not normalized by the Phase 2 main-example loader.

OWA unproved questions use `Unknown`. CWA false values can reflect closed-world failure and are never automatically mapped to `CONTRADICTED` by VeriLogic-NS.

## Commands

Run from the repository root with the backend virtual environment activated:

```text
python -m verilogic_ns_api.datasets download proofwriter
python -m verilogic_ns_api.datasets inspect proofwriter --variant depth-1
python -m verilogic_ns_api.datasets prepare proofwriter --variant depth-1
```

The download streams to `raw/archives/*.zip.part`, enforces timeouts and a size limit, validates ZIP integrity, computes SHA-256, and atomically renames only after success. An existing valid archive is reused; replacement requires `--force`. `--expected-sha256` upgrades an observed checksum to an explicit expected-and-matched check. The published source does not provide a checksum in the inspected archive, so the repository’s checksum is explicitly marked observed-only.

ZIP extraction is optional (`--extract`) because the archive expands to roughly 3.41 GB and the loader can stream directly from ZIP. Extraction rejects traversal paths, drive-like paths, backslashes, symbolic links, excess entries, and excess expanded size. Extracted directories are content-addressed under `raw/extracted/<sha256>/`.

## Local layout

```text
datasets/proofwriter/
|-- README.md
|-- DATASET_CARD.md
|-- manifest.example.json
|-- provenance.observed.json
|-- raw/                 ignored archives, extraction, local provenance
`-- processed/           ignored normalized JSONL and preparation manifests
```

Raw archives, extracted data, normalized examples, temporary files, generated samples, and evaluation outputs are ignored because they are large, locally reproducible, potentially copyrighted, and not source code. Only acquisition/normalization code, synthetic fixtures, schemas, documentation, and non-example aggregate provenance metadata are tracked.

## Limitations

- The Phase 2 loader targets main OWA question files only; staged and abductive tasks are deferred.
- Proof metadata is preserved as supplied but is not semantically normalized or verified yet.
- Inspection reports overlap without changing official splits.
- Logical benchmark labels describe consequence under dataset semantics, not factual truth.
