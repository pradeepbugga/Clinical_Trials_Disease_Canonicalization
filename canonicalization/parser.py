import json
from pathlib import Path

from models.models import CanonicalizationOutput

import logging

logger = logging.getLogger(__name__)


def parse_canonicalization_results(batch_results_path: str | Path) -> list[dict]:
    """
    Parse successful OpenAI Batch API condition canonicalization results from a JSONL file.

    Parameters
    ----------
    batch_results
        Raw contents of the Batch API output JSONL file.

    Returns
    -------
    list[dict]
        Parsed condition canonicalization results, each containing:
        - extracted_condition_id: int
        - common_name: str
    """

    parsed_records = []

    batch_results = Path(batch_results_path).read_text(encoding="utf-8")

    results = (json.loads(line) for line in batch_results.splitlines() if line.strip())

    for result in results:

        extracted_condition_id = int(result["custom_id"])
        body = result["response"]["body"]

        # Skip incomplete responses
        if body["status"] != "completed":
            logger.warning(
                "Skipping %s: %s",
                extracted_condition_id,
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
            logger.warning("No extraction results for %s", extracted_condition_id)
            continue

        try:
            canonicalization = json.loads(text)
        except json.JSONDecodeError:
            logger.exception("Failed to parse %s", extracted_condition_id)
            logger.debug("Raw text: %r", text)
            continue

        parsed_records.append(
            CanonicalizationOutput(
                extracted_condition_id=extracted_condition_id,
                common_name=canonicalization["common_name"],
                technical_name=canonicalization["technical_name"],
                abbreviations=canonicalization["abbreviations"],
            )
        )

    return parsed_records
