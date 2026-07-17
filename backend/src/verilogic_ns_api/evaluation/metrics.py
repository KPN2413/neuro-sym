from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import median

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from verilogic_ns_api.research.models import (
    BenchmarkExample,
    GoldLabel,
    MetricReport,
    PerDepthMetrics,
    PerLabelMetrics,
    PredictionLabel,
    PredictionRecord,
)

ANSWERED_LABELS = {
    PredictionLabel.ENTAILED,
    PredictionLabel.CONTRADICTED,
    PredictionLabel.UNKNOWN,
}


def compute_metrics(
    examples: Sequence[BenchmarkExample], predictions: Sequence[PredictionRecord]
) -> MetricReport:
    if len(examples) != len(predictions):
        raise ValueError("Examples and predictions must have identical lengths")
    for example, prediction in zip(examples, predictions, strict=True):
        if example.example_id != prediction.example_id:
            raise ValueError(
                f"Prediction/example mismatch: {prediction.example_id} != {example.example_id}"
            )

    total = len(examples)
    answered = sum(prediction.predicted_label in ANSWERED_LABELS for prediction in predictions)
    abstained = sum(
        prediction.predicted_label is PredictionLabel.ABSTAIN for prediction in predictions
    )
    errored = sum(prediction.predicted_label is PredictionLabel.ERROR for prediction in predictions)
    correct = sum(
        prediction.predicted_label.value == example.gold_label.value
        for example, prediction in zip(examples, predictions, strict=True)
    )

    y_true = [example.gold_label.value for example in examples]
    y_pred = [prediction.predicted_label.value for prediction in predictions]
    gold_labels = [label.value for label in GoldLabel]
    prediction_labels = [label.value for label in PredictionLabel]

    if total:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=gold_labels,
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=prediction_labels)
    else:
        precision = recall = f1 = support = [0, 0, 0]
        matrix = [[0 for _ in prediction_labels] for _ in prediction_labels]

    predicted_counts = Counter(y_pred)
    per_label = {
        label: PerLabelMetrics(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
            predicted=predicted_counts[label],
        )
        for index, label in enumerate(gold_labels)
    }
    confusion = {
        gold: {
            predicted: int(matrix[row_index][column_index])
            for column_index, predicted in enumerate(prediction_labels)
        }
        for row_index, gold in enumerate(gold_labels)
    }

    depth_groups: dict[str, list[tuple[BenchmarkExample, PredictionRecord]]] = defaultdict(list)
    for example, prediction in zip(examples, predictions, strict=True):
        key = str(example.reasoning_depth) if example.reasoning_depth is not None else "missing"
        depth_groups[key].append((example, prediction))
    per_depth: dict[str, PerDepthMetrics] = {}
    for depth, items in sorted(depth_groups.items()):
        depth_total = len(items)
        depth_answered = sum(item[1].predicted_label in ANSWERED_LABELS for item in items)
        depth_correct = sum(
            prediction.predicted_label.value == example.gold_label.value
            for example, prediction in items
        )
        per_depth[depth] = PerDepthMetrics(
            total=depth_total,
            answered=depth_answered,
            correct=depth_correct,
            accuracy=depth_correct / depth_total if depth_total else 0,
            coverage=depth_answered / depth_total if depth_total else 0,
        )

    answered_accuracy = correct / answered if answered else None
    non_cache_latencies = [
        prediction.latency_ms for prediction in predictions if prediction.cache_hit is False
    ]
    provider_predictions = [
        prediction for prediction in predictions if prediction.cache_hit is False
    ]
    provider_generation_duration_ms = sum(
        prediction.provider_generation_duration_ms or 0 for prediction in provider_predictions
    )
    provider_generation_tokens = sum(
        prediction.completion_tokens or 0 for prediction in provider_predictions
    )
    return MetricReport(
        total_examples=total,
        answered_examples=answered,
        abstained_examples=abstained,
        errored_examples=errored,
        accuracy=correct / total if total else 0,
        answered_only_accuracy=answered_accuracy,
        coverage=answered / total if total else 0,
        selective_risk=1 - answered_accuracy if answered_accuracy is not None else None,
        macro_precision=float(sum(precision) / len(gold_labels)),
        macro_recall=float(sum(recall) / len(gold_labels)),
        macro_f1=float(sum(f1) / len(gold_labels)),
        confusion_matrix=confusion,
        per_label_metrics=per_label,
        per_depth_metrics=per_depth,
        invalid_prediction_count=sum(
            prediction.error_type == "InvalidPredictionError" for prediction in predictions
        ),
        refusal_count=sum(
            prediction.abstention_reason == "provider_refusal" for prediction in predictions
        ),
        cache_hit_count=sum(prediction.cache_hit is True for prediction in predictions),
        cache_miss_count=sum(prediction.cache_hit is False for prediction in predictions),
        input_tokens=sum(prediction.prompt_tokens or 0 for prediction in predictions),
        output_tokens=sum(prediction.completion_tokens or 0 for prediction in predictions),
        reasoning_tokens=sum(prediction.reasoning_tokens or 0 for prediction in predictions),
        cached_input_tokens=sum(prediction.cached_input_tokens or 0 for prediction in predictions),
        non_cache_total_latency_ms=sum(non_cache_latencies),
        non_cache_median_latency_ms=(
            float(median(non_cache_latencies)) if non_cache_latencies else None
        ),
        provider_total_duration_ms=sum(
            prediction.provider_total_duration_ms or 0 for prediction in provider_predictions
        ),
        provider_load_duration_ms=sum(
            prediction.provider_load_duration_ms or 0 for prediction in provider_predictions
        ),
        provider_prompt_eval_duration_ms=sum(
            prediction.provider_prompt_eval_duration_ms or 0 for prediction in provider_predictions
        ),
        provider_generation_duration_ms=provider_generation_duration_ms,
        generation_tokens_per_second=(
            provider_generation_tokens / (provider_generation_duration_ms / 1000)
            if provider_generation_duration_ms
            else None
        ),
        estimated_cost_usd=sum(prediction.estimated_cost_usd or 0 for prediction in predictions),
    )
