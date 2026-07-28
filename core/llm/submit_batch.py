from pathlib import Path
import json
import logging

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


def submit_batches(batch_dir: str | Path) -> Path:
    """
    Upload JSONL files and create OpenAI Batch jobs.

    Parameters
    ----------
    batch_dir
        Directory containing input JSONL batch files.

    Returns
    -------
    Path
        Path to the generated batch_manifest.json.
    """

    load_dotenv()

    client = OpenAI()

    batch_dir = Path(batch_dir)

    jsonl_files = sorted(batch_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {batch_dir}")

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

    manifest_path = batch_dir / "batch_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return manifest_path
