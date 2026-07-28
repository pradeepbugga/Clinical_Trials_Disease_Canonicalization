from models.models import EvaluationExample
from pathlib import Path
import json
import logging

from finetuning.prompts import SYSTEM_PROMPT
from openai import OpenAI

logger = logging.getLogger(__name__)


def load_examples(path: str | Path) -> list[EvaluationExample]:
    """
    Load the test examples from a JSONL file.

    Parameters
    ----------
    path
        Path to the JSONL file containing the test examples.

    Returns
    -------
    list[EvaluationExample]
        List of EvaluationExample instances.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            messages = record["messages"]

            user = messages[1]["content"]
            assistant = json.loads(messages[2]["content"])

            example = EvaluationExample(
                input=user,
                expected_common_name=assistant["common_name"],
                expected_technical_name=assistant["technical_name"],
                expected_abbreviations=assistant["abbreviations"],
            )
            examples.append(example)

    return examples


def predict_condition(
    client: OpenAI,
    model: str,
    condition: str,
) -> dict:
    """
    Predict the canonical representation of a condition using a fine-tuned model.
    """

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": condition,
            },
        ],
    )

    try:
        return json.loads(response.output_text)
    except json.JSONDecodeError as e:
        logging.exception("Failed to parse model output for condition: %s", condition)
        logging.debug("Raw model output: %r", response.output_text)
        raise e


def compare_example(example, prediction):

    pred_common = prediction.get("common_name")

    pred_technical = prediction.get("technical_name") or None
    truth_technical = example.expected_technical_name or None

    pred_abbrev = set(prediction.get("abbreviations", []))
    truth_abbrev = set(example.expected_abbreviations)

    return {
        "common_correct": pred_common == example.expected_common_name,
        "technical_correct": pred_technical == truth_technical,
        "abbreviation_tp": len(pred_abbrev & truth_abbrev),
        "abbreviation_fp": len(pred_abbrev - truth_abbrev),
        "abbreviation_fn": len(truth_abbrev - pred_abbrev),
    }


def compute_metrics(results):

    n = len(results)

    common_accuracy = sum(r["common_correct"] for r in results) / n

    technical_accuracy = sum(r["technical_correct"] for r in results) / n

    tp = sum(r["abbreviation_tp"] for r in results)
    fp = sum(r["abbreviation_fp"] for r in results)
    fn = sum(r["abbreviation_fn"] for r in results)

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "common_accuracy": common_accuracy,
        "technical_accuracy": technical_accuracy,
        "abbreviation_precision": precision,
        "abbreviation_recall": recall,
        "abbreviation_f1": f1,
    }


def write_predictions(
    rows,
    output_path,
):
    """
    Write the predictions and evaluation results to a CSV file.
    """

    import csv

    if not rows:
        logger.warning("No rows to write to CSV.")
        return

    output_path = Path(output_path)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    client,
    model: str,
    examples: list[EvaluationExample],
    output_path
):

    examples = load_examples("data/test.jsonl")

    rows = []
    results = []


    for example in examples:

        prediction = predict_condition(
            client=client,
            model=model,
            condition=example.input,
        )

        comparison = compare_example(example, prediction)

        row = {
            "input": example.input,
            "expected_common_name": example.expected_common_name,
            "predicted_common_name": prediction.get("common_name"),
            "expected_technical_name": example.expected_technical_name,
            "predicted_technical_name": prediction.get("technical_name"),
            "expected_abbreviations": ", ".join(example.expected_abbreviations),
            "predicted_abbreviations": ", ".join(prediction.get("abbreviations", [])),
            **comparison,
        }

        rows.append(row)
        results.append(comparison)

    metrics = compute_metrics(results)
    write_predictions(rows, "data/evaluation_results.csv")

    return metrics

    logger.info("Examples: %d", len(examples))
    logger.info("Common accuracy: %.2f%%", metrics["common_accuracy"] * 100)
    logger.info("Technical accuracy: %.2f%%", metrics["technical_accuracy"] * 100)
    logger.info(
        "Abbreviation precision: %.2f%%",
        metrics["abbreviation_precision"] * 100,
    )
    logger.info(
        "Abbreviation recall: %.2f%%",
        metrics["abbreviation_recall"] * 100,
    )
    logger.info(
        "Abbreviation F1: %.2f%%",
        metrics["abbreviation_f1"] * 100,
    )


def main():

    client = OpenAI()

    examples = load_examples("data/test.jsonl")

    base_model_metrics = evaluate_model(
        client=client,
        model="gpt-4.1-nano",
        examples=examples,
        output_path="data/base_model_predictions.csv",

    )

    finetuned_model_metrics = evaluate_model(
        client=client,
        model="ft:gpt-4o-mini:pb929:canonicalization-2024-06-05-20-30-00",
        examples=examples,
        output_path="data/finetuned_model_predictions.csv",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
