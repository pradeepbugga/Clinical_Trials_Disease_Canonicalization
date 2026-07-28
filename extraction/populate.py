from pathlib import Path
from collections.abc import Iterable
from extraction.parser import parse_condition_results


def get_or_create_condition(cur, name: str) -> int:
    """
    Insert a condition if it does not already exist and return its ID.
    """

    cur.execute(
        """
        INSERT INTO Conditions (name)
        VALUES (%s)
        ON CONFLICT (name) DO NOTHING
        """,
        (name,),
    )

    cur.execute(
        """
        SELECT condition_id
        FROM Conditions
        WHERE name = %s
        """,
        (name,),
    )

    return cur.fetchone()[0]


def populate_records(
    cur,
    records: Iterable[dict],
) -> dict:
    """
    Populate Conditions and TrialConditions from parsed records.
    """

    trials_processed = 0
    condition_links = 0

    for record in records:

        nct_id = record["nct_id"]
        conditions = record["conditions"]

        trials_processed += 1

        seen = set()

        for condition in conditions:

            condition = condition.strip()

            if not condition or condition in seen:
                continue

            condition_id = get_or_create_condition(cur, condition)

            cur.execute(
                """
                INSERT INTO TrialConditions (
                    nct_id,
                    condition_id
                )
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    nct_id,
                    condition_id,
                ),
            )

            seen.add(condition)

            condition_links += 1

    return {
        "trials_processed": trials_processed,
        "condition_links": condition_links,
    }


def populate_jsonl(
    cur,
    results_path: str | Path,
) -> dict:
    """
    Populate the database from an OpenAI Batch results JSONL file.
    """

    with Path(results_path).open(
        "r",
        encoding="utf-8",
    ) as f:
        batch_results = f.read()

    records = parse_condition_results(batch_results)

    return populate_records(cur, records)
