from __future__ import annotations

from copy import deepcopy


def controlled_corruptions(
    theory: dict[str, object], query: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Create deterministic train-only corruption fixtures without consulting an answer label."""
    outputs: dict[str, dict[str, object]] = {}
    facts = theory.get("facts")
    rules = theory.get("rules")
    if isinstance(facts, list) and facts:
        omitted = deepcopy(theory)
        omitted["facts"] = deepcopy(facts[:-1])
        outputs["omitted_source"] = omitted

        duplicate = deepcopy(theory)
        duplicate["facts"].append(deepcopy(duplicate["facts"][0]))  # type: ignore[union-attr]
        outputs["duplicate_source"] = duplicate

        invented = deepcopy(theory)
        invented["facts"][0]["source_id"] = "sent999"  # type: ignore[index]
        outputs["invented_source"] = invented

        polarity = deepcopy(theory)
        literal = polarity["facts"][0]["fact"]  # type: ignore[index]
        literal["negated"] = not literal["negated"]
        outputs["flipped_polarity"] = polarity

        predicate = deepcopy(theory)
        predicate["facts"][0]["fact"]["predicate"] = "wrong_predicate"  # type: ignore[index]
        outputs["wrong_predicate"] = predicate

        constant = deepcopy(theory)
        constant["facts"][0]["fact"]["arguments"][0]["id"] = "wrong_constant"  # type: ignore[index]
        outputs["wrong_constant"] = constant

        arity = deepcopy(theory)
        arity["facts"][0]["fact"]["arity"] = 2  # type: ignore[index]
        outputs["wrong_arity"] = arity

        confused = deepcopy(theory)
        original = confused["facts"].pop(0)  # type: ignore[union-attr]
        fact = original["fact"]
        confused["rules"].append(  # type: ignore[union-attr]
            {
                "source_id": original["source_id"],
                "kind": "rule",
                "rule": {"variables": [], "body": [deepcopy(fact)], "head": deepcopy(fact)},
            }
        )
        outputs["fact_rule_confusion"] = confused
    if isinstance(rules, list) and rules:
        body = rules[0].get("rule", {}).get("body", [])
        if isinstance(body, list) and body:
            missing = deepcopy(theory)
            missing["rules"][0]["rule"]["body"] = deepcopy(body[1:])  # type: ignore[index]
            outputs["missing_premise"] = missing

            invented = deepcopy(theory)
            extra = deepcopy(body[0])
            extra["predicate"] = "invented_premise"
            invented["rules"][0]["rule"]["body"].append(extra)  # type: ignore[index]
            outputs["invented_premise"] = invented

        conclusion = deepcopy(theory)
        conclusion["rules"][0]["rule"]["head"]["predicate"] = "wrong_conclusion"  # type: ignore[index]
        outputs["wrong_conclusion"] = conclusion

        unsafe = deepcopy(theory)
        unsafe["rules"][0]["rule"]["head"]["arguments"] = [  # type: ignore[index]
            {"kind": "variable", "name": "Unbound"}
        ]
        outputs["unsafe_variable"] = unsafe

    query_polarity = deepcopy(query)
    query_polarity["query"]["negated"] = not query_polarity["query"]["negated"]  # type: ignore[index]
    outputs["query_polarity_error"] = query_polarity
    query_predicate = deepcopy(query)
    query_predicate["query"]["predicate"] = "wrong_query_predicate"  # type: ignore[index]
    outputs["query_predicate_error"] = query_predicate
    return outputs
