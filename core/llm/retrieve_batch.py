from pathlib import Path
import json
import logging

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


def retrieve_batches(batch_dir: str | Path) -> list[Path]:
    """
    Download completed OpenAI Batch outputs.

    Parameters
    ----------
    batch_dir
        Directory containing batch_manifest.json.

    Returns
    -------
    list[Path]
        Paths to the downloaded raw JSONL output files.
    """

    load_dotenv()

    client = OpenAI()

    batch_dir = Path(batch_dir)

    manifest_path = batch_dir / "batch_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    raw_files = []

    for entry in manifest:

        batch = client.batches.retrieve(entry["batch_id"])

        logger.info(
            "%s: %s",
            entry["filename"],
            batch.status,
        )

        if batch.status != "completed":
            logger.info(
                "Skipping %s (status=%s)",
                entry["filename"],
                batch.status,
            )
            continue

        output = client.files.content(batch.output_file_id)

        raw_path = batch_dir / f"{Path(entry['filename']).stem}_raw.jsonl"

        with raw_path.open("wb") as f:
            f.write(output.content)

        raw_files.append(raw_path)

        logger.info(
            "Downloaded %s",
            raw_path.name,
        )

    return raw_files
