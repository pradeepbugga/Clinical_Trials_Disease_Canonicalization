from pathlib import Path

from core.db import get_db_connection

from core.llm.batch import (
    generate_batch_jsonl,
    split_jsonl,
)
from core.llm.submit_batch import submit_batches

from extraction.extract import (
    iter_trials,
    build_request,
)


def submit():

    conn = get_db_connection()
    cur = conn.cursor()

    try:

        batch_dir = Path("batch_files/extraction")
        batch_dir.mkdir(parents=True, exist_ok=True)

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

    finally:

        cur.close()
        conn.close()


if __name__ == "__main__":
    submit()