from pathlib import Path
import json
from openai import OpenAI
import logging
import argparse

from tqdm import tqdm

from models.models import EvaluationExample
from finetuning.prompts import SYSTEM_PROMPT
from canonicalization.schema import CANONICALIZATION_SCHEMA


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
                input_name=user,
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
    Predict the canonical representation of a condition.
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
        text={
            "format": {
                "type": "json_schema",
                "name": "canonicalization",
                "strict": True,
                "schema": CANONICALIZATION_SCHEMA,
            }
        },
    )

    try:
        prediction = json.loads(response.output_text)
    except json.JSONDecodeError:
        logger.exception(
            "Failed to parse model output for condition: %s",
            condition,
        )
        logger.error(
            "Raw model output:\n%s",
            response.output_text,
        )
        raise

    required_fields = {
        "common_name",
        "technical_name",
        "abbreviations",
    }

    missing = required_fields - prediction.keys()

    if missing:
        raise ValueError(
            f"Model output is missing fields {sorted(missing)} "
            f"for condition {condition!r}: {prediction}"
        )

    if not isinstance(prediction["common_name"], str):
        raise TypeError(f"common_name must be a string: {prediction}")

    if not isinstance(prediction["technical_name"], str):
        raise TypeError(f"technical_name must be a string: {prediction}")

    if not isinstance(prediction["abbreviations"], list):
        raise TypeError(f"abbreviations must be a list: {prediction}")

    if not all(isinstance(value, str) for value in prediction["abbreviations"]):
        raise TypeError(f"Every abbreviation must be a string: {prediction}")

    return prediction


def normalize_text(value) -> str:
    """
    Normalize a scalar text value for exact-match evaluation.

    Missing values are represented as an empty string, consistent with the
    fine-tuning data contract.
    """
    if value is None:
        return ""

    return str(value).strip()


def normalize_name(value) -> str:
    """
    Normalize a medical name for case-insensitive exact matching.
    """
    return normalize_text(value).casefold()


def normalize_abbreviations(values) -> set[str]:
    """
    Normalize abbreviation lists for order- and case-insensitive comparison.
    """
    if not values:
        return set()

    return {normalize_name(value) for value in values if normalize_text(value)}


def compare_example(
    example: EvaluationExample,
    prediction: dict,
) -> dict:
    """
    Compare the model's prediction against the expected values for a single example.
    Returns a dictionary containing the comparison results.
    """

    pred_common = normalize_name(prediction.get("common_name"))

    truth_common = normalize_name(example.expected_common_name)

    pred_technical = normalize_name(prediction.get("technical_name"))

    truth_technical = normalize_name(example.expected_technical_name)

    pred_abbrev = normalize_abbreviations(prediction.get("abbreviations", []))

    truth_abbrev = normalize_abbreviations(example.expected_abbreviations)

    return {
        "common_correct": pred_common == truth_common,
        "technical_correct": pred_technical == truth_technical,
        "abbreviation_tp": len(pred_abbrev & truth_abbrev),
        "abbreviation_fp": len(pred_abbrev - truth_abbrev),
        "abbreviation_fn": len(truth_abbrev - pred_abbrev),
    }


def compute_metrics(results):
    """
    Compute evaluation metrics from the comparison results of multiple examples.
    Returns a dictionary containing the computed metrics.
    """

    if not results:
        return {
            "common_accuracy": 0.0,
            "technical_accuracy": 0.0,
            "abbreviation_precision": 1.0,
            "abbreviation_recall": 1.0,
            "abbreviation_f1": 1.0,
        }

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


def evaluate_model(client, model: str, examples: list[EvaluationExample], output_path):
    """
    Evaluate a model on a set of examples and write the predictions to a CSV file.

    Parameters
    ----------
    client
        OpenAI client instance.
    model
        Model ID to evaluate.
    examples
        List of EvaluationExample instances to evaluate.
    output_path
        Path to the output CSV file for predictions.

    Returns
    -------
    dict
        Computed evaluation metrics.
    """

    rows = []
    results = []

    for example in tqdm(examples, desc=f"Evaluating {model}"):

        prediction = predict_condition(
            client=client,
            model=model,
            condition=example.input_name,
        )

        comparison = compare_example(example, prediction)

        row = {
            "input": example.input_name,
            "expected_common_name": example.expected_common_name,
            "predicted_common_name": prediction.get("common_name"),
            "normalized_expected_common_name": normalize_name(
                example.expected_common_name
            ),
            "normalized_predicted_common_name": normalize_name(
                prediction.get("common_name")
            ),
            "expected_technical_name": normalize_text(example.expected_technical_name),
            "predicted_technical_name": normalize_text(
                prediction.get("technical_name")
            ),
            "normalized_expected_technical_name": normalize_name(
                example.expected_technical_name
            ),
            "normalized_predicted_technical_name": normalize_name(
                prediction.get("technical_name")
            ),
            "expected_abbreviations": ", ".join(example.expected_abbreviations or []),
            "predicted_abbreviations": ", ".join(prediction.get("abbreviations", [])),
            **comparison,
        }

        rows.append(row)
        results.append(comparison)

    metrics = compute_metrics(results)
    write_predictions(rows, output_path)

    return metrics


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate condition canonicalization models."
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("./finetuning/data/test.jsonl"),
        help="Path to the test dataset.",
    )

    parser.add_argument(
        "--base-model",
        default="ft:gpt-4.1-nano-2025-04-14:personal:canonicalize:BzdChpxY",
        help="Base model to evaluate.",
    )

    parser.add_argument(
        "--finetuned-model",
        default="ft:gpt-4.1-nano-2025-04-14:personal:canonicalize:E6moEi9A",
        help="Fine-tuned model ID.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for prediction CSVs.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    client = OpenAI()

    examples = load_examples(args.test_file)

    base_model_metrics = evaluate_model(
        client=client,
        model=args.base_model,
        examples=examples,
        output_path=args.output_dir / "base_model_predictions.csv",
    )

    finetuned_model_metrics = evaluate_model(
        client=client,
        model=args.finetuned_model,
        examples=examples,
        output_path=args.output_dir / "finetuned_model_predictions.csv",
    )

    print(base_model_metrics)
    print(finetuned_model_metrics)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()