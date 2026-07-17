# Vibe Coding Workflow

The project is implemented through Codex-driven phases; users are not expected to edit or repair code manually. Automation does not reduce engineering or research rigor.

## Phase cycle

1. Read the current prompt, `AGENTS.md`, relevant decisions, and affected contracts.
2. Inspect the repository and Git state before editing; preserve all user work.
3. Restate the exact phase boundary and identify any conflict before proceeding.
4. Make the smallest coherent implementation within the approved architecture.
5. Add tests with each feature, including negative and failure paths.
6. Run focused checks while iterating, then the full applicable phase suite.
7. Fix failures caused by the change; never ask the user to edit a file manually.
8. Update commands, examples, architecture, security notes, and decisions when behavior changes.
9. Inspect the final diff, Git status, secrets, generated files, and scope.
10. Report exact commands, pass/fail results, limitations, Git state, and the next approved phase; then stop.

## Safe autonomy

Codex may create code, configuration, tests, docs, local virtual environments, dependency installs, builds, and local commits authorized by the prompt. It must not create remotes, push, expose secrets, fabricate results, install system software silently, perform destructive Git/filesystem operations, or enter the next phase without a new prompt.

If a runtime is missing, complete every independent check that remains possible and report the exact prerequisite. If existing files conflict with a phase, adapt without overwriting; stop only when continuing would materially redesign or destroy user work.

## Change hygiene

- Keep each phase reviewable and avoid opportunistic refactors.
- Use provider mocks for routine tests; paid/network calls require the approved phase and explicit configuration.
- Preserve public APIs and versioned schemas unless the prompt authorizes a change.
- Do not weaken validation to make a fixture pass.
- Prefer explicit states and recorded errors over silent fallback.
- Do not commit `.env`, dependencies, build outputs, caches, datasets without provenance, or generated results.

## Handoff template

Every phase handoff states: phase status; created/changed artifacts; architecture decisions; every verification command with result; missing prerequisites/blockers; Git status and commit hash if any; current limitations; and the exact next recommended phase.
