from pathlib import Path

from core.db.connection import get_db_connection
from core.schema import create_extraction_tables


from core.llm.batch_utils import (
    generate_batch_jsonl,
    split_jsonl,
)
from core.llm.submit_batch import submit_batches

from extraction.extract import (
    iter_trials,
    build_request,
)

import logging

logger = logging.getLogger(__name__)


def submit():

    """
    Submit clinical trial data for disease extraction using OpenAI Batch API.
    This function connects to the database, retrieves clinical trial records that require
    disease extraction, generates JSONL files for batch processing, and submits them to the
    OpenAI Batch API.
    """

    conn = get_db_connection()
    cur = conn.cursor()

    try:

        create_extraction_tables(cur)
        conn.commit()

        batch_dir = Path("batch_files/extraction")
        batch_dir.mkdir(parents=True, exist_ok=True)

        input_jsonl = batch_dir / "batch_input.jsonl"

        count = sum(1 for _ in iter_trials(cur))
        logger.info("Total trials to process: %d", count)

        generate_batch_jsonl(
            records=iter_trials(cur),
            output_path=input_jsonl,
            build_request=build_request,
        )

        batch_files = split_jsonl(
            input_jsonl,
            batch_dir,
        )

        submit_batches(batch_files)

    finally:

        cur.close()
        conn.close()


if __name__ == "__main__":
    submit()
