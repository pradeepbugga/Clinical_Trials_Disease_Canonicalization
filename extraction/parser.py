import json
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def parse_condition_results(batch_results_path: str | Path) -> list[dict]:
    """
    Parse successful OpenAI Batch API condition extraction results.

    Parameters
    ----------
    batch_results
        Raw contents of the Batch API output JSONL file.

    Returns
    -------
    list[dict]
        Parsed condition extraction records.
    """

    parsed_records = []

    batch_results = Path(batch_results_path).read_text(encoding="utf-8")

    results = (json.loads(line) for line in batch_results.splitlines() if line.strip())

    for result in results:

        nct_id = result["custom_id"]
        body = result["response"]["body"]

        # Skip incomplete responses
        if body["status"] != "completed":
            logger.warning(
                "Skipping %s: %s",
                nct_id,
                body["incomplete_details"]["reason"],
            )
            continue

        output = body["output"]

        # Get the last non-empty output_text
        text = None
        for item in reversed(output):
            if item["type"] != "message":
                continue

            for content in item["content"]:
                if content["type"] == "output_text" and content["text"].strip():
                    text = content["text"]
                    break

            if text is not None:
                break

        if text is None:
            logger.warning("No extraction results for %s", nct_id)
            continue

        try:
            extraction = json.loads(text)
        except json.JSONDecodeError:
            logger.exception("Failed to parse %s", nct_id)
            logger.debug("Raw text: %r", text)
            continue

        parsed_records.append(
            {
                "nct_id": nct_id,
                "conditions": extraction["conditions"],
            }
        )

    return parsed_records


def write_jsonl(path: str | Path, records: list[dict]):
    """
    Write a list of dictionaries to a JSONL file.
    """

    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
