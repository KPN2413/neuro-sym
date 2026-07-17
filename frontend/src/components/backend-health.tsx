"use client";

import { useEffect, useState } from "react";

type HealthResponse = {
  service: string;
  status: "ok";
  version: string;
};

type HealthState =
  | { kind: "loading" }
  | { kind: "success"; health: HealthResponse }
  | { kind: "failure" };

const apiBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

function isHealthResponse(value: unknown): value is HealthResponse {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const candidate = value as Partial<HealthResponse>;
  return (
    typeof candidate.service === "string" &&
    candidate.status === "ok" &&
    typeof candidate.version === "string"
  );
}

export function BackendHealth() {
  const [state, setState] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    async function checkHealth() {
      try {
        const response = await fetch(`${apiBaseUrl}/health`, {
          cache: "no-store",
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("Health endpoint returned an unsuccessful status.");
        }

        const payload: unknown = await response.json();
        if (!isHealthResponse(payload)) {
          throw new Error("Health endpoint returned an unexpected payload.");
        }

        setState({ kind: "success", health: payload });
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setState({ kind: "failure" });
      }
    }

    void checkHealth();
    return () => controller.abort();
  }, []);

  if (state.kind === "loading") {
    return (
      <div
        role="status"
        className="flex min-w-56 items-center gap-3 rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm"
      >
        <span className="size-2 animate-pulse rounded-full bg-amber-500" aria-hidden="true" />
        Checking backend…
      </div>
    );
  }

  if (state.kind === "failure") {
    return (
      <div
        role="alert"
        className="flex min-w-56 items-center gap-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800"
      >
        <span className="size-2 rounded-full bg-rose-600" aria-hidden="true" />
        Backend unavailable
      </div>
    );
  }

  return (
    <div
      role="status"
      className="flex min-w-56 items-center gap-3 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900"
    >
      <span className="size-2 rounded-full bg-emerald-600" aria-hidden="true" />
      <span>
        Backend healthy <span className="font-mono text-xs">v{state.health.version}</span>
      </span>
    </div>
  );
}
