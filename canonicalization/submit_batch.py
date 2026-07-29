from pathlib import Path

from core.db.connection import get_db_connection
from core.schema import create_canonicalization_tables


from core.llm.batch_utils import (
    generate_batch_jsonl,
    split_jsonl,
)
from core.llm.submit_batch import submit_batches

from canonicalization.extract import (
    iter_conditions,
    build_request,
)

import logging

logger = logging.getLogger(__name__)


def submit():

    """
    Submit condition data for canonicalization using OpenAI Batch API.
    This function connects to the database, retrieves condition records that require
    canonicalization, generates JSONL files for batch processing, and submits them to the
    OpenAI Batch API.
    """
    

    conn = get_db_connection()
    cur = conn.cursor()

    try:

        create_canonicalization_tables(cur)
        conn.commit()

        batch_dir = Path("batch_files/canonicalization")
        batch_dir.mkdir(parents=True, exist_ok=True)

        input_jsonl = batch_dir / "batch_input.jsonl"

        conditions = list(iter_conditions(cur))

        logger.info("Total conditions to process: %d", len(conditions))

        generate_batch_jsonl(
            records=conditions,
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
