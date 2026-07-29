from pathlib import Path
from collections.abc import Iterable
from canonicalization.parser import parse_canonicalization_results
from models.models import CanonicalizationOutput


def get_or_create_canonical_condition(
    cur,
    common_name: str,
    technical_name: str,
) -> int:
    """
    Insert a condition if it does not already exist and return its ID.
    """

    cur.execute(
        """
        INSERT INTO CanonicalConditions (
        common_name,
        technical_name)
        VALUES (%s, %s)
        ON CONFLICT (common_name) DO NOTHING
        """,
        (common_name, technical_name),
    )

    cur.execute(
        """
        SELECT canonical_condition_id
        FROM CanonicalConditions
        WHERE common_name = %s
        """,
        (common_name,),
    )

    return cur.fetchone()[0]


def populate_abbreviations(
    cur,
    canonical_condition_id,
    abbreviations,
):
    """
    Insert abbreviations for a canonical condition.
    """

    for abbreviation in abbreviations:

        cur.execute(
            """
            INSERT INTO CanonicalAbbreviations (
                canonical_condition_id,
                abbreviation
            )
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (canonical_condition_id, abbreviation),
        )


def populate_records(cur, records: Iterable[CanonicalizationOutput]) -> dict:
    """
    Populate the database with canonicalization results.
    """

    processed = 0

    for record in records:

        canonical_condition_id = get_or_create_canonical_condition(
            cur,
            record.common_name,
            record.technical_name,
        )

        populate_abbreviations(
            cur,
            canonical_condition_id,
            record.abbreviations,
        )

        cur.execute(
            """
            INSERT INTO ExtractedCanonicalConditions (
                extracted_condition_id,
                canonical_condition_id
            )
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (record.extracted_condition_id, canonical_condition_id),
        )

        processed += 1

    return {
        "conditions_processed": processed,
    }


def populate_jsonl(
    cur,
    results_path: str | Path,
) -> dict:
    """
    Populate the database from an OpenAI Batch results JSONL file.
    """

    records = parse_canonicalization_results(results_path)

    return populate_records(cur, records)
