import json
from pathlib import Path

import pytest

from verilogic_ns_api.reasoning.cli import main
from verilogic_ns_api.reasoning.configuration import FormalRepresentationError
from verilogic_ns_api.reasoning.engine import ForwardChainingEngine
from verilogic_ns_api.reasoning.models import ReasoningOutput, ReasoningStatus
from verilogic_ns_api.reasoning.proofwriter import ProofWriterFormalParser, run_conformance
from verilogic_ns_api.reasoning.verifier import ProofVerifier
from verilogic_ns_api.research.models import Split

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
THEORY_FIXTURES = REPOSITORY_ROOT / "examples" / "theories"


def test_reason_and_verify_cli_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    theory_path = THEORY_FIXTURES / "unary-multistep.json"

    assert main(["reason", "--input", str(theory_path), "--output", "reason.json"]) == 0
    assert main(["verify-proof", "--theory", str(theory_path), "--proof", "reason.json"]) == 0

    payload = json.loads((tmp_path / "reason.json").read_text(encoding="utf-8"))
    assert ReasoningOutput.model_validate(payload).result.status is ReasoningStatus.ENTAILED


def test_cli_refuses_overwrite_and_reports_resource_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    theory_path = THEORY_FIXTURES / "unary-multistep.json"
    assert main(["reason", "--input", str(theory_path), "--output", "result.json"]) == 0

    assert main(["reason", "--input", str(theory_path), "--output", "result.json"]) == 2
    assert "Output already exists" in capsys.readouterr().err

    assert (
        main(
            [
                "reason",
                "--input",
                str(theory_path),
                "--max-derived-literals",
                "1",
            ]
        )
        == 3
    )
    assert "RESOURCE_LIMIT" in capsys.readouterr().err


def test_cli_refuses_output_outside_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)

    code = main(
        [
            "reason",
            "--input",
            str(THEORY_FIXTURES / "entailed.json"),
            "--output",
            str(tmp_path / "outside.json"),
        ]
    )

    assert code == 2
    assert "must remain beneath" in capsys.readouterr().err


def test_formal_parser_supports_fact_and_rule_negation_markers() -> None:
    parser = ProofWriterFormalParser()

    assert parser.parse_literal('("Alice" "is" "kind" "-")') == (
        "Alice",
        "is",
        "kind",
        True,
    )
    body, head = parser.parse_rule(
        '((("someone" "is" "green" "+") '
        '("someone" "is" "kind" "~")) -> '
        '("someone" "is" "nice" "+"))'
    )
    assert body[1][3] is True
    assert head[3] is False


@pytest.mark.parametrize(
    "representation",
    [
        "not-parenthesized",
        '("Alice" "is" "kind")',
        '("Alice" "is" "kind" "?")',
        '("unterminated)',
        '("Alice" "is" "kind" "+")) trailing',
    ],
)
def test_formal_parser_fails_closed_on_unsupported_input(representation: str) -> None:
    with pytest.raises(FormalRepresentationError):
        ProofWriterFormalParser().parse_literal(representation)


def test_formal_record_conversion_uses_only_oracle_structure() -> None:
    record = {
        "id": "Synthetic-1",
        "triples": {
            "triple1": {
                "text": "Nora is warm.",
                "representation": '("Nora" "is" "warm" "+")',
            }
        },
        "rules": {
            "rule1": {
                "text": "If something is warm and not blue then it is quiet.",
                "representation": '((("something" "is" "warm" "+") '
                '("something" "is" "blue" "~")) -> '
                '("something" "is" "quiet" "+"))',
            }
        },
        "questions": {
            "Q1": {
                "question": "Nora is quiet.",
                "representation": '("Nora" "is" "quiet" "+")',
                "answer": "Unknown",
                "QDep": 0,
                "strategy": "random",
            }
        },
    }

    example = ProofWriterFormalParser().convert_record_question(
        record,
        "Q1",
        variant="depth-1",
        split=Split.DEVELOPMENT,
    )
    outcome = ForwardChainingEngine().reason(example.theory)

    assert outcome.result.status is ReasoningStatus.UNKNOWN
    assert example.theory.parser_metadata is not None
    assert example.theory.parser_metadata.parser_name == "proofwriter.formal"
    assert ProofVerifier().verify_result(example.theory, outcome.result).valid is True


def test_conformance_report_counts_predictions_and_verified_proofs() -> None:
    parser = ProofWriterFormalParser()
    examples = []
    for question_id, representation, answer, strategy, expected_status in (
        ("Q1", '("Nora" "is" "warm" "+")', True, "proof", "ENTAILED"),
        ("Q2", '("Nora" "is" "warm" "-")', False, "inv-proof", "CONTRADICTED"),
        ("Q3", '("Nora" "is" "blue" "+")', "Unknown", "random", "UNKNOWN"),
    ):
        record = {
            "id": f"Synthetic-{question_id}",
            "triples": {
                "triple1": {
                    "text": "Nora is warm.",
                    "representation": '("Nora" "is" "warm" "+")',
                }
            },
            "rules": {},
            "questions": {
                question_id: {
                    "question": "Synthetic query.",
                    "representation": representation,
                    "answer": answer,
                    "QDep": 0,
                    "strategy": strategy,
                    "proofsWithIntermediates": [{"expected": expected_status}],
                }
            },
        }
        examples.append(
            parser.convert_record_question(
                record,
                question_id,
                variant="depth-1",
                split=Split.DEVELOPMENT,
            )
        )

    report = run_conformance(tuple(examples))

    assert report["example_count"] == 3
    assert report["accuracy"] == 1.0
    assert report["proof_verification_rate"] == 1.0
    assert report["mismatch_count"] == 0
    assert report["invalid_prediction_count"] == 0
    assert report["per_label"]["ENTAILED"]["accuracy"] == 1.0
    assert report["per_depth"]["0"]["accuracy"] == 1.0
    assert report["total_closure_size"] >= 3
    assert report["natural_language_parsed"] is False
