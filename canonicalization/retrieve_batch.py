from pathlib import Path

from core.db.connection import get_db_connection

from core.llm.batch_utils import combine_jsonl
from core.llm.retrieve_batch import retrieve_batches

from canonicalization.populate import populate_jsonl

import logging

logger = logging.getLogger(__name__)


def retrieve():

    conn = get_db_connection()
    cur = conn.cursor()

    try:

        batch_dir = Path("batch_files/canonicalization")

        raw_files = retrieve_batches(batch_dir)

        combined_raw = combine_jsonl(
            input_paths=raw_files,
            output_path=batch_dir / "raw_results.jsonl",
        )

        populate_jsonl(
            cur,
            combined_raw,
        )

        conn.commit()

    finally:

        cur.close()
        conn.close()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    retrieve()
