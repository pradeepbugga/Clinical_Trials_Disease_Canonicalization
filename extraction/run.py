from pathlib import Path

from core.db.connection import get_db_connection

from core.llm.batch_utils import (
    generate_batch_jsonl,
    split_jsonl,
    combine_jsonl,
)
from core.llm.submit_batch import submit_batches
from core.llm.retrieve_batch import retrieve_batches

from extraction.extract import iter_trials, build_request
from extraction.parser import parse_condition_results
from extraction.populate import populate_database


def run():

    conn = get_db_connection()
    cur = conn.cursor()

    try:

        batch_dir = Path("batch_files/extraction")

        input_jsonl = batch_dir / "batch_input.jsonl"

        generate_batch_jsonl(
            records=iter_trials(cur),
            output_path=input_jsonl,
            build_request=build_request,
        )

        split_jsonl(
            input_jsonl,
            batch_dir,
        )

        submit_batches(batch_dir)

        raw_files = retrieve_batches(batch_dir)

        combined_raw = combine_jsonl(
            raw_files,
            batch_dir / "raw_results.jsonl",
        )

        parsed_results = parse_results(combined_raw)

        populate_database(
            cur,
            parsed_results,
        )

        conn.commit()

    finally:

        cur.close()
        conn.close()


if __name__ == "__main__":
    run()
