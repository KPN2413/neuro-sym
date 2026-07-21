from __future__ import annotations

from dataclasses import dataclass

from verilogic_ns_api.reasoning.models import sha256_payload
from verilogic_ns_api.research.models import BenchmarkExample, Split
from verilogic_ns_api.semantic_parsing.models import (
    NeutralStatement,
    QueryParseInput,
    TheoryParseInput,
)


class ParserInputError(ValueError):
    pass


@dataclass(frozen=True)
class SourceBinding:
    neutral_id: str
    original_id: str
    text: str
    expected_kind: str


@dataclass(frozen=True)
class PreparedTheoryView:
    public: TheoryParseInput
    bindings: tuple[SourceBinding, ...]


@dataclass(frozen=True)
class PreparedQueryView:
    public: QueryParseInput
    original_source_id: str
    text: str


def prepare_theory_view(example: BenchmarkExample) -> PreparedTheoryView:
    _reject_test(example)
    statements: list[NeutralStatement] = []
    bindings: list[SourceBinding] = []
    for index, source in enumerate(example.source_statements, start=1):
        neutral = f"sent{index}"
        statements.append(NeutralStatement(source_id=neutral, text=source.text))
        bindings.append(
            SourceBinding(
                neutral_id=neutral,
                original_id=source.source_id,
                text=source.text,
                expected_kind=source.kind,
            )
        )
    payload = [{"source_id": item.source_id, "text": item.text} for item in statements]
    return PreparedTheoryView(
        public=TheoryParseInput(input_hash=sha256_payload(payload), statements=tuple(statements)),
        bindings=tuple(bindings),
    )


def prepare_query_view(example: BenchmarkExample) -> PreparedQueryView:
    _reject_test(example)
    payload = {"text": example.query}
    return PreparedQueryView(
        public=QueryParseInput(input_hash=sha256_payload(payload), text=example.query),
        original_source_id=example.question_id or "query",
        text=example.query,
    )


def assert_same_theory(left: PreparedTheoryView, right: PreparedTheoryView) -> None:
    if left.public.input_hash != right.public.input_hash or left.bindings != right.bindings:
        raise ParserInputError("records sharing a theory ID contain different theory text")


def _reject_test(example: BenchmarkExample) -> None:
    if example.split == Split.TEST:
        raise ParserInputError("test-split parsing requires a future explicit protocol")
