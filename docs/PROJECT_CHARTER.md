# Project Charter

## Identity and purpose

**Official topic:** Neuro-Symbolic Large Language Model
**Implementation name:** VeriLogic-NS

VeriLogic-NS is an explainable research framework for testing whether a natural-language conclusion follows from supplied natural-language premises. An existing LLM will serve as a semantic parser, while a deterministic symbolic engine will own logical inference. Every accepted conclusion must be accompanied by a proof linked to the source statements used.

The framework evaluates logical consequence relative to supplied premises. It does not certify that the premises describe the real world accurately.

## Research problem

Direct LLM answers can be unsupported or logically inconsistent. VeriLogic-NS separates probabilistic language interpretation from deterministic formal reasoning:

1. A provider-independent LLM adapter produces a restricted typed AST.
2. Structural and semantic validators check the formalization.
3. Limited, auditable correction may repair only approved structural defects.
4. Confidence gating abstains when formalization remains unsafe or uncertain.
5. A forward-chaining engine derives consequences and source-linked proofs.
6. The public decision is `ENTAILED`, `CONTRADICTED`, or `UNKNOWN`; `INVALID` and `INCONSISTENT` remain internal safety states.

## Approved MVP

The MVP includes ProofWriter as the primary benchmark; direct and few-shot LLM baselines; natural-language-to-AST parsing; a versioned JSON AST; unary and binary predicates; positive and explicit-negative facts; conjunctive Horn-style rules; multi-step deductions; open-world reasoning; validation, correction, abstention, and proof tracing; automated experiments and metrics; a FastAPI backend; a Next.js interface; tests; and reproducible local, Docker, and deployment-ready execution.

The system will measure accuracy, proof behavior, robustness, latency, and cost. No improvement is assumed before experiments are run.

## Explicit exclusions

Unless a later prompt explicitly approves them, do not implement FOLIO support, Z3 verification, fine-tuning, RAG, document retrieval, knowledge graphs, Neo4j, provider-comparison studies, multi-agent architecture, authentication, payments, a mobile application, or training a new LLM.

## Principles

- Formal reasoning is deterministic and independent of the parser provider.
- Untrusted model output crosses a strict typed-AST boundary.
- Validation and confidence controls fail closed.
- Explicit negation does not imply closed-world negation.
- Proofs must be reproducible and traceable to source IDs.
- Baselines, prompts, datasets, configurations, seeds, outputs, and software versions must be recorded.
- Research reports distinguish measured evidence, hypotheses, and limitations.

## Success criteria

The completed capstone must execute the approved benchmark conditions reproducibly, compare LLM-only and neuro-symbolic systems fairly, report the defined metrics without fabricated claims, produce auditable proofs for accepted symbolic decisions, and document failure and abstention modes. Each phase must meet its gate in `PHASE_PLAN.md` before the next begins.
