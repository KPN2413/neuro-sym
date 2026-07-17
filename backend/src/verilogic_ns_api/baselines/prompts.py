from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from verilogic_ns_api.baselines.models import LLMRequest, canonical_json, sha256_json, sha256_text
from verilogic_ns_api.research.models import GoldLabel, PredictionInput, Split

EXAMPLE_OPEN = "<untrusted_proofwriter_example>"
EXAMPLE_CLOSE = "</untrusted_proofwriter_example>"
DEMOS_OPEN = "<approved_training_demonstrations>"
DEMOS_CLOSE = "</approved_training_demonstrations>"


@dataclass(frozen=True)
class PromptTemplate:
    path: Path
    version: str
    text: str
    sha256: str


@dataclass(frozen=True)
class Demonstration:
    example: PredictionInput
    label: GoldLabel

    def __post_init__(self) -> None:
        if self.example.split is not Split.TRAIN:
            raise ValueError("Few-shot demonstrations must come from the training split")


def load_prompt_template(
    path: Path, *, version: str, expected_sha256: str | None = None
) -> PromptTemplate:
    text = path.read_text(encoding="utf-8")
    observed = sha256_text(text)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            f"Prompt hash mismatch for {path}: expected {expected_sha256}, observed {observed}"
        )
    return PromptTemplate(path=path, version=version, text=text, sha256=observed)


def _render_example(example: PredictionInput) -> str:
    payload = canonical_json({"context": example.context, "query": example.query})
    return f"{EXAMPLE_OPEN}\n{payload}\n{EXAMPLE_CLOSE}"


def render_prompt(
    template: PromptTemplate,
    example: PredictionInput,
    *,
    demonstrations: list[Demonstration] | None = None,
) -> tuple[str, str, str]:
    demonstrations = demonstrations or []
    instructions = template.text.rstrip()
    if demonstrations:
        if len(demonstrations) != 6:
            raise ValueError("Few-shot rendering requires exactly six demonstrations")
        rendered = [DEMOS_OPEN]
        for index, demonstration in enumerate(demonstrations, start=1):
            rendered.extend(
                [
                    f'<demonstration index="{index}">',
                    _render_example(demonstration.example),
                    canonical_json({"label": demonstration.label.value}),
                    "</demonstration>",
                ]
            )
        rendered.append(DEMOS_CLOSE)
        instructions = f"{instructions}\n\n" + "\n".join(rendered)
    input_text = _render_example(example)
    rendered_hash = sha256_json({"instructions": instructions, "input_text": input_text})
    return instructions, input_text, rendered_hash


def build_request(
    *,
    template: PromptTemplate,
    example: PredictionInput,
    configured_model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    output_schema_hash: str,
    demonstration_manifest_hash: str | None = None,
    selection_manifest_hash: str | None = None,
    demonstrations: list[Demonstration] | None = None,
    provider: str = "openai",
    api_family: Literal["responses", "native_chat"] = "responses",
    endpoint_identity: str | None = None,
    provider_version: str | None = None,
    model_digest: str | None = None,
    model_options: dict[str, JsonValue] | None = None,
) -> LLMRequest:
    instructions, input_text, rendered_hash = render_prompt(
        template, example, demonstrations=demonstrations
    )
    return LLMRequest(
        provider=provider,
        api_family=api_family,
        configured_model=configured_model,
        endpoint_identity=endpoint_identity,
        provider_version=provider_version,
        model_digest=model_digest,
        model_options=model_options or {},
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        instructions=instructions,
        input_text=input_text,
        prompt_version=template.version,
        prompt_hash=template.sha256,
        output_schema_hash=output_schema_hash,
        demonstration_manifest_hash=demonstration_manifest_hash,
        selection_manifest_hash=selection_manifest_hash,
        example_id=example.example_id,
        rendered_request_hash=rendered_hash,
    )
