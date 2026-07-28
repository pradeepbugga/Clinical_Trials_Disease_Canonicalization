from pathlib import Path
import json

from core.db.connection import get_db_connection

from fine_tuning.prompts import SYSTEM_PROMPT
from models.models import FineTuningExample


def iter_training_examples(cur):
    """
    Yield training examples for fine-tuning from the canonical condition database.
    """

    cur.execute(
        """
        SELECT
            ec.name,
            cc.common_name,
            cc.technical_name,
            cc.canonical_condition_id
        FROM ExtractedConditions ec
        JOIN ExtractedCanonicalConditions ecc
            ON ec.extracted_condition_id = ecc.extracted_condition_id
        JOIN CanonicalConditions cc
            ON ecc.canonical_condition_id = cc.canonical_condition_id
        ORDER BY ec.extracted_condition_id
        """
    )

    for (
        input_name,
        common_name,
        technical_name,
        canonical_condition_id,
    ) in cur.fetchall():

        cur.execute(
            """
            SELECT abbreviation
            FROM CanonicalAbbreviations
            WHERE canonical_condition_id = %s
            ORDER BY abbreviation
            """,
            (canonical_condition_id,),
        )

        abbreviations = [
            row[0]
            for row in cur.fetchall()
        ]

        if (
            technical_name
            and common_name.lower().strip()
            == technical_name.lower().strip()
        ):
            technical_name = ""

        yield FineTuningExample(
            input_name=input_name,
            common_name=common_name,
            technical_name=technical_name,
            abbreviations=abbreviations,
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
    output_path: str | Path,
):
    """
    Export the canonical condition database as an OpenAI fine-tuning dataset.
    """

    conn = get_db_connection()
    cur = conn.cursor()

    try:

        records = iter_training_examples(cur)

        write_jsonl(
            records,
            output_path,
        )

    finally:

        cur.close()
        conn.close()


if __name__ == "__main__":

    output_path = (
        Path(__file__).parent / "data" / "train.jsonl"
    )

    prepare(output_path)