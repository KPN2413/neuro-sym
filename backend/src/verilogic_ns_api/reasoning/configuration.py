from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningLimits:
    max_derived_literals: int = 100_000
    max_rule_firings: int = 1_000_000
    max_rounds: int = 1_000
    max_proof_nodes: int = 100_000
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_derived_literals",
            "max_rule_firings",
            "max_rounds",
            "max_proof_nodes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "max_derived_literals": self.max_derived_literals,
            "max_rule_firings": self.max_rule_firings,
            "max_rounds": self.max_rounds,
            "max_proof_nodes": self.max_proof_nodes,
            "timeout_seconds": self.timeout_seconds,
        }


class ReasoningError(Exception):
    """Base class for safe symbolic-reasoning failures."""


class ResourceLimitError(ReasoningError):
    def __init__(self, limit_name: str, limit: int | float, observed: int | float) -> None:
        self.limit_name = limit_name
        self.limit = limit
        self.observed = observed
        super().__init__(
            f"Reasoning resource limit exceeded: {limit_name}={limit}, observed={observed}"
        )


class ProofVerificationError(ReasoningError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class FormalRepresentationError(ReasoningError):
    pass
