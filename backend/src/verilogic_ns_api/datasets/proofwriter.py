from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from verilogic_ns_api.datasets.acquisition import PROOFWRITER_VERSION
from verilogic_ns_api.datasets.errors import (
    AmbiguousWorldAssumptionError,
    DatasetRecordError,
    DuplicateExampleError,
)
from verilogic_ns_api.research.models import (
    BenchmarkExample,
    ExampleProvenance,
    GoldLabel,
    SourceStatement,
    Split,
    StructuredStatement,
    WorldAssumption,
)

LOADER_VERSION = "1.0"
MAIN_FILE_PATTERN = re.compile(r"(?:^|/)(OWA|CWA)/([^/]+)/meta-(train|dev|test)\.jsonl$")


@dataclass
class LoaderStats:
    records_seen: int = 0
    examples_yielded: int = 0
    invalid_records: int = 0
    invalid_questions: int = 0
    duplicate_ids: int = 0
    missing_fields: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_seen": self.records_seen,
            "examples_yielded": self.examples_yielded,
            "invalid_records": self.invalid_records,
            "invalid_questions": self.invalid_questions,
            "duplicate_ids": self.duplicate_ids,
            "missing_fields": dict(sorted(self.missing_fields.items())),
        }


@dataclass(frozen=True)
class DatasetLayout:
    files: dict[WorldAssumption, dict[str, dict[Split, str]]]

    def serializable(self) -> dict[str, dict[str, list[str]]]:
        return {
            world.value: {
                variant: sorted(split.value for split in splits)
                for variant, splits in sorted(variants.items())
            }
            for world, variants in sorted(self.files.items(), key=lambda item: item[0].value)
        }


class ProofWriterSource:
    """Read ProofWriter JSONL lazily from a ZIP or extracted directory."""

    def __init__(self, data_source: Path) -> None:
        self.data_source = data_source.resolve()
        if not self.data_source.exists():
            raise DatasetRecordError(f"ProofWriter data source does not exist: {data_source}")
        if not self.data_source.is_dir() and not zipfile.is_zipfile(self.data_source):
            raise DatasetRecordError(
                f"ProofWriter data source must be an extracted directory or ZIP: {data_source}"
            )
        self._directory_root = self._resolve_directory_root() if self.data_source.is_dir() else None
        self._layout = self._discover_layout()

    @property
    def layout(self) -> DatasetLayout:
        return self._layout

    def _resolve_directory_root(self) -> Path:
        if (self.data_source / "OWA").is_dir() or (self.data_source / "CWA").is_dir():
            return self.data_source
        candidates = [
            path
            for path in self.data_source.iterdir()
            if path.is_dir()
            and path.name.startswith("proofwriter-dataset-")
            and ((path / "OWA").is_dir() or (path / "CWA").is_dir())
        ]
        if len(candidates) != 1:
            raise DatasetRecordError(
                f"Could not identify exactly one ProofWriter dataset root inside {self.data_source}"
            )
        return candidates[0]

    def _member_names(self) -> Iterator[str]:
        if self._directory_root is not None:
            for path in self._directory_root.rglob("meta-*.jsonl"):
                if path.is_file():
                    yield path.relative_to(self._directory_root).as_posix()
            return
        with zipfile.ZipFile(self.data_source) as archive:
            yield from (info.filename for info in archive.infolist() if not info.is_dir())

    def _discover_layout(self) -> DatasetLayout:
        files: dict[WorldAssumption, dict[str, dict[Split, str]]] = {}
        for member in self._member_names():
            match = MAIN_FILE_PATTERN.search(member)
            if match is None:
                continue
            world = WorldAssumption(match.group(1))
            variant = match.group(2)
            split = Split(match.group(3))
            files.setdefault(world, {}).setdefault(variant, {})[split] = member
        if not files:
            raise DatasetRecordError(f"No ProofWriter main split files found in {self.data_source}")
        return DatasetLayout(files=files)

    def member_for(self, world: WorldAssumption, variant: str, split: Split) -> str:
        try:
            return self._layout.files[world][variant][split]
        except KeyError as error:
            raise DatasetRecordError(
                f"No {world.value} {variant!r} {split.value!r} split in {self.data_source}"
            ) from error

    @contextmanager
    def open_text(self, member: str) -> Generator[TextIO]:
        if self._directory_root is not None:
            path = self._directory_root / member
            with path.open(encoding="utf-8") as stream:
                yield stream
            return
        with (
            zipfile.ZipFile(self.data_source) as archive,
            archive.open(member) as binary_stream,
            io.TextIOWrapper(binary_stream, encoding="utf-8") as stream,
        ):
            yield stream


def stable_content_hash(context: str, query: str) -> str:
    payload = json.dumps(
        {"context": context, "query": query},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_example_id(
    *,
    dataset_version: str,
    world_assumption: WorldAssumption,
    variant: str,
    theory_id: str,
    question_id: str,
) -> str:
    return "/".join(
        [
            "proofwriter",
            dataset_version,
            world_assumption.value,
            variant,
            theory_id,
            question_id,
        ]
    )


def map_gold_label(
    raw_label: Any,
    *,
    world_assumption: WorldAssumption,
    question: Mapping[str, Any],
) -> GoldLabel:
    if world_assumption is WorldAssumption.CLOSED:
        if raw_label is True:
            return GoldLabel.ENTAILED
        raise AmbiguousWorldAssumptionError(
            "CWA false/unknown labels cannot be mapped to CONTRADICTED without explicit proof semantics"
        )

    if raw_label is True:
        return GoldLabel.ENTAILED
    if raw_label is False:
        strategy = question.get("strategy")
        proof_metadata = question.get("proofsWithIntermediates")
        if strategy not in {"proof", "inv-proof"} or not proof_metadata:
            raise AmbiguousWorldAssumptionError(
                "OWA false label lacks the proof metadata required to establish the opposite"
            )
        return GoldLabel.CONTRADICTED
    if isinstance(raw_label, str) and raw_label.casefold() == "unknown":
        return GoldLabel.UNKNOWN
    raise DatasetRecordError(f"Unsupported ProofWriter label: {raw_label!r}")


def _require_mapping(
    record: Mapping[str, Any], field_name: str, stats: LoaderStats
) -> Mapping[str, Any]:
    value = record.get(field_name)
    if not isinstance(value, Mapping):
        stats.missing_fields[field_name] += 1
        raise DatasetRecordError(f"Missing or invalid mapping field {field_name!r}")
    return value


def _structured_statements(
    entries: Mapping[str, Any], *, field_name: str
) -> dict[str, StructuredStatement]:
    normalized: dict[str, StructuredStatement] = {}
    for identifier, raw in entries.items():
        if not isinstance(identifier, str) or not isinstance(raw, Mapping):
            raise DatasetRecordError(f"Invalid {field_name} entry {identifier!r}")
        text = raw.get("text")
        representation = raw.get("representation")
        if not isinstance(text, str) or not text.strip():
            raise DatasetRecordError(f"{field_name}.{identifier} is missing non-empty text")
        if representation is not None and not isinstance(representation, str):
            raise DatasetRecordError(f"{field_name}.{identifier}.representation must be a string")
        normalized[identifier] = StructuredStatement(text=text, representation=representation)
    return normalized


def _source_statements(
    record: Mapping[str, Any],
    facts: Mapping[str, StructuredStatement],
    rules: Mapping[str, StructuredStatement],
) -> list[SourceStatement]:
    sentences = record.get("sentences")
    if sentences is not None:
        if not isinstance(sentences, Mapping) or not sentences:
            raise DatasetRecordError("sentences must be a non-empty mapping when present")
        result = []
        for identifier, text in sentences.items():
            if not isinstance(identifier, str) or not identifier:
                raise DatasetRecordError("Sentence identifiers must be non-empty strings")
            if not isinstance(text, str) or not text.strip():
                raise DatasetRecordError(f"Sentence {identifier!r} must contain non-empty text")
            result.append(SourceStatement(source_id=identifier, text=text, kind="sentence"))
        return result
    result = [
        SourceStatement(
            source_id=identifier,
            text=statement.text,
            kind="fact",
            representation=statement.representation,
        )
        for identifier, statement in facts.items()
    ]
    result.extend(
        SourceStatement(
            source_id=identifier,
            text=statement.text,
            kind="rule",
            representation=statement.representation,
        )
        for identifier, statement in rules.items()
    )
    return result


def _reasoning_depth(question: Mapping[str, Any]) -> int | None:
    raw_depth = question.get("QDep")
    if raw_depth is None or raw_depth == "":
        return None
    if isinstance(raw_depth, bool):
        raise DatasetRecordError("QDep cannot be Boolean")
    if isinstance(raw_depth, int) and raw_depth >= 0:
        return raw_depth
    if isinstance(raw_depth, str) and raw_depth.isdigit():
        return int(raw_depth)
    raise DatasetRecordError(f"Invalid QDep value: {raw_depth!r}")


class ProofWriterLoader:
    def __init__(
        self,
        data_source: Path,
        *,
        dataset_version: str = PROOFWRITER_VERSION,
        dataset_manifest_reference: str | None = None,
    ) -> None:
        self.source = ProofWriterSource(data_source)
        self.dataset_version = dataset_version
        self.dataset_manifest_reference = dataset_manifest_reference

    @property
    def layout(self) -> DatasetLayout:
        return self.source.layout

    def iter_examples(
        self,
        *,
        variant: str,
        split: Split,
        world_assumption: WorldAssumption = WorldAssumption.OPEN,
        strict: bool = True,
        stats: LoaderStats | None = None,
        max_examples: int | None = None,
    ) -> Iterator[BenchmarkExample]:
        stats = stats or LoaderStats()
        member = self.source.member_for(world_assumption, variant, split)
        seen_ids: set[str] = set()

        with self.source.open_text(member) as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                stats.records_seen += 1
                try:
                    record = json.loads(raw_line)
                    if not isinstance(record, Mapping):
                        raise DatasetRecordError("Record must be a JSON object")
                    theory_id = record.get("id")
                    context = record.get("theory")
                    if not isinstance(theory_id, str) or not theory_id:
                        stats.missing_fields["id"] += 1
                        raise DatasetRecordError("Missing non-empty theory id")
                    if not isinstance(context, str) or not context.strip():
                        stats.missing_fields["theory"] += 1
                        raise DatasetRecordError("Missing non-empty theory text")
                    facts = _structured_statements(
                        _require_mapping(record, "triples", stats), field_name="triples"
                    )
                    rules = _structured_statements(
                        _require_mapping(record, "rules", stats), field_name="rules"
                    )
                    questions = _require_mapping(record, "questions", stats)
                    statements = _source_statements(record, facts, rules)
                    if not statements:
                        stats.missing_fields["source_statements"] += 1
                        raise DatasetRecordError("No usable source statements")
                    record_sha256 = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
                except (json.JSONDecodeError, DatasetRecordError, ValidationError) as error:
                    stats.invalid_records += 1
                    contextual = DatasetRecordError(f"{member}:{line_number}: {error}")
                    if strict:
                        raise contextual from error
                    continue

                for question_id, raw_question in questions.items():
                    try:
                        if not isinstance(question_id, str) or not isinstance(
                            raw_question, Mapping
                        ):
                            raise DatasetRecordError(f"Invalid question entry {question_id!r}")
                        query = raw_question.get("question")
                        if not isinstance(query, str) or not query.strip():
                            stats.missing_fields["questions.question"] += 1
                            raise DatasetRecordError(f"{question_id} is missing question text")
                        if "answer" not in raw_question:
                            stats.missing_fields["questions.answer"] += 1
                            raise DatasetRecordError(f"{question_id} is missing answer")
                        raw_label = raw_question["answer"]
                        label = map_gold_label(
                            raw_label,
                            world_assumption=world_assumption,
                            question=raw_question,
                        )
                        example_id = stable_example_id(
                            dataset_version=self.dataset_version,
                            world_assumption=world_assumption,
                            variant=variant,
                            theory_id=theory_id,
                            question_id=question_id,
                        )
                        if example_id in seen_ids:
                            stats.duplicate_ids += 1
                            raise DuplicateExampleError(f"Duplicate example id {example_id}")
                        seen_ids.add(example_id)
                        proof_payload = {
                            key: raw_question[key]
                            for key in ("proofs", "proofsWithIntermediates", "strategy", "QLen")
                            if key in raw_question
                        }
                        content_sha256 = stable_content_hash(context, query)
                        example = BenchmarkExample(
                            example_id=example_id,
                            dataset_version=self.dataset_version,
                            variant=variant,
                            split=split,
                            theory_id=theory_id,
                            question_id=question_id,
                            reasoning_depth=_reasoning_depth(raw_question),
                            source_statements=statements,
                            context=context,
                            query=query,
                            gold_label=label,
                            original_raw_label=raw_label,
                            world_assumption=world_assumption,
                            structured_facts=facts,
                            structured_rules=rules,
                            gold_proofs=proof_payload or None,
                            source_relative_path=member,
                            provenance=ExampleProvenance(
                                loader_version=LOADER_VERSION,
                                record_line=line_number,
                                record_sha256=record_sha256,
                                content_sha256=content_sha256,
                                dataset_manifest_reference=self.dataset_manifest_reference,
                            ),
                        )
                    except (DatasetRecordError, ValidationError) as error:
                        stats.invalid_questions += 1
                        contextual = DatasetRecordError(
                            f"{member}:{line_number}:{question_id}: {error}"
                        )
                        if strict:
                            raise contextual from error
                        continue

                    stats.examples_yielded += 1
                    yield example
                    if max_examples is not None and stats.examples_yielded >= max_examples:
                        return
