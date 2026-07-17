import { BackendHealth } from "@/components/backend-health";

const principles = [
  "Typed AST boundary",
  "Deterministic inference",
  "Source-linked proofs",
];

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col px-6 py-10 sm:px-10 sm:py-16">
      <nav aria-label="Project status" className="flex items-center justify-between gap-4">
        <span className="font-mono text-sm font-semibold tracking-tight text-slate-950">
          VeriLogic-NS
        </span>
        <span className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 font-mono text-xs font-medium text-indigo-800">
          Phase 1: Foundation
        </span>
      </nav>

      <section aria-labelledby="project-title" className="flex flex-1 flex-col justify-center py-20">
        <p className="mb-5 font-mono text-xs font-semibold uppercase tracking-[0.2em] text-indigo-700">
          Neuro-Symbolic Large Language Model
        </p>
        <h1
          id="project-title"
          className="max-w-4xl text-balance text-5xl font-semibold tracking-[-0.045em] text-slate-950 sm:text-7xl"
        >
          Language understanding, with logic you can inspect.
        </h1>
        <p className="mt-7 max-w-2xl text-pretty text-lg leading-8 text-slate-600">
          An explainable neuro-symbolic framework that translates natural-language theories into
          a restricted typed AST and will verify conclusions with deterministic, source-linked
          reasoning.
        </p>

        <ul aria-label="Architectural principles" className="mt-8 flex flex-wrap gap-2">
          {principles.map((principle) => (
            <li
              key={principle}
              className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm"
            >
              {principle}
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="system-status" className="border-t border-slate-200 py-8">
        <div className="grid gap-5 sm:grid-cols-[1fr_auto] sm:items-center">
          <div>
            <h2 id="system-status" className="text-sm font-semibold text-slate-950">
              Foundation status
            </h2>
            <p className="mt-1 max-w-xl text-sm leading-6 text-slate-600">
              The API shell and contract are active. No LLM or symbolic reasoning result is
              produced in this phase.
            </p>
          </div>
          <BackendHealth />
        </div>
      </section>
    </main>
  );
}
