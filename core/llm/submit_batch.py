from pathlib import Path
import json
import logging

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


def submit_batches(jsonl_files: list[Path])->Path:
    """
    Upload JSONL files and create OpenAI Batch jobs.

    Parameters
    ----------
    jsonl_files
        List of JSONL files to submit.

    Returns
    -------
    Path
        Path to the generated batch_manifest.json.
    """

    load_dotenv()

    client = OpenAI()

    if not jsonl_files:
        raise ValueError("No JSONL files provided for submission.")

    manifest = []

    for jsonl_path in jsonl_files:

        logger.info("Submitting %s", jsonl_path.name)

        with jsonl_path.open("rb") as f:
            uploaded = client.files.create(
                file=f,
                purpose="batch",
            )

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "filename": jsonl_path.name,
            },
        )

        manifest.append(
            {
                "filename": jsonl_path.name,
                "file_id": uploaded.id,
                "batch_id": batch.id,
            }
        )

    batch_dir = Path(jsonl_files[0]).parent
    manifest_path = batch_dir / "batch_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return manifest_path
