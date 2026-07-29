from pathlib import Path
import json
import random
import logging

from finetuning.prompts import SYSTEM_PROMPT
from models.models import FineTuningExample

logger = logging.getLogger(__name__)


def iter_training_examples(input_path: str | Path):
    """
    Yield training examples for fine-tuning from the exported canonical mappings.
    """

    input_path = Path(input_path)

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    for record in records:
        technical_name = record["technical_name"]

        if (
            technical_name
            and technical_name.strip().lower() == record["common_name"].strip().lower()
        ):
            technical_name = None

        yield FineTuningExample(
            input_name=record["input_name"],
            common_name=record["common_name"],
            technical_name=technical_name,
            abbreviations=record.get("abbreviations") or [],
        )


def build_messages(
    example: FineTuningExample,
) -> dict:
    """
    Convert a training example into OpenAI fine-tuning format.
    """

    assistant_response = {
        "common_name": example.common_name,
        "technical_name": example.technical_name,
        "abbreviations": example.abbreviations,
    }

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": example.input_name,
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    assistant_response,
                    ensure_ascii=False,
                ),
            },
        ]
    }


def write_jsonl(
    records,
    output_path: str | Path,
):
    """
    Write training examples to an OpenAI JSONL file.
    """

    output_path = Path(output_path)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            messages = build_messages(record)

            f.write(
                json.dumps(
                    messages,
                    ensure_ascii=False,
                )
                + "\n"
            )


def prepare(
    input_path: str | Path,
    train_output_path: str | Path,
    test_output_path: str | Path,
    test_fraction: float = 0.1,
    seed: int = 42,
):
    """
    Prepare training and test datasets for fine-tuning.

    Parameters
    ----------
    input_path
        Path to the exported canonical mappings JSON file.
    train_output_path
        Path to the output training JSONL file.
    test_output_path
        Path to the output test JSONL file.
    test_fraction
        Fraction of examples to use for the test set (between 0 and 1).
    seed
        Random seed for shuffling the examples.
    """

    if not (0 <= test_fraction <= 1):
        raise ValueError("test_fraction must be between 0 and 1.")

    examples = list(iter_training_examples(input_path))

    random.Random(seed).shuffle(examples)

    split_index = int(len(examples) * (1 - test_fraction))

    train_examples = examples[:split_index]
    test_examples = examples[split_index:]

    write_jsonl(train_examples, train_output_path)
    write_jsonl(test_examples, test_output_path)

    logger.info(
        "Prepared %d examples, with %d for training and %d for testing.",
        len(examples),
        len(train_examples),
        len(test_examples),
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(__file__).parent / "data" / "canonical_mappings.json"

    train_path = Path(__file__).parent / "data" / "train.jsonl"
    test_path = Path(__file__).parent / "data" / "test.jsonl"

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    prepare(
        input_path=input_path,
        train_output_path=train_path,
        test_output_path=test_path,
    )
