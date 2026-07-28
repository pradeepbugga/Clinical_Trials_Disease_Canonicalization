from pathlib import Path

from core.db import get_db_connection

from core.llm.batch import combine_jsonl
from core.llm.retrieve_batch import retrieve_batches

from extraction.parser import parse_results
from extraction.populate import populate_database


def retrieve():

    conn = get_db_connection()

    try:

        batch_dir = Path("batch_files/extraction")

        raw_files = retrieve_batches(batch_dir)

        combined_raw = combine_jsonl(
            input_files=raw_files,
            output_path=batch_dir / "raw_results.jsonl",
        )

        parsed_results = parse_results(combined_raw)

        populate_database(
            conn,
            parsed_results,
        )

        conn.commit()

    finally:

        conn.close()


if __name__ == "__main__":
    retrieve()