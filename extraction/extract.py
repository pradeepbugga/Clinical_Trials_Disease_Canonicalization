from typing import Any, Iterator
from models.models import ExtractionInput
from extraction.prompts import EXTRACTION_PROMPT
from extraction.schema import EXTRACTION_SCHEMA

DEFAULT_MODEL = "gpt-4.1-nano"


def iter_trials(cur) -> Iterator[ExtractionInput]:
    """
    Yield Trial objects requiring condition extraction.

    Parameters
    ----------
    cur
        Database cursor.

    Yields
    ------
    ExtractionInput
        An object containing trial information for extraction.

    """

    cur.execute(
        """
        SELECT
            nct_id,
            title,
            summary,
            detailed_description
        FROM ClinicalTrials
        WHERE nct_id NOT IN (
            SELECT DISTINCT nct_id
            FROM TrialExtractedConditions
        )
        """
    )

    while True:

        rows = cur.fetchmany(1000)

        if not rows:
            break

        for nct_id, title, summary, detailed_description in rows:

            yield ExtractionInput(
                nct_id=nct_id,
                title=title,
                summary=summary,
                detailed_description=detailed_description,
            )


def _format_trial(extraction_input: ExtractionInput) -> str:
    """Convert a clinical trial into text for disease extraction."""

    parts = []

    if extraction_input.title:
        parts.append(f"Title: {extraction_input.title.strip()}")

    if extraction_input.summary:
        parts.append(f"Summary: {extraction_input.summary.strip()}")

    if extraction_input.detailed_description:
        parts.append(
            f"Detailed Description: {extraction_input.detailed_description.strip()}"
        )

    return "\n\n".join(parts)


def build_request(
    extraction_input: ExtractionInput,
    prompt: str = EXTRACTION_PROMPT,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Build an OpenAI Batch API request for disease extraction.

    Parameters
    ----------
    extraction_input : ExtractionInput
        An object containing trial information for extraction.
    prompt : str, optional
        Prompt for the disease extraction task.
    model : str, optional
        OpenAI model used for extraction.

    Returns
    -------
    dict
        JSON payload compatible with the OpenAI Batch API.
    """
    return {
        "custom_id": extraction_input.nct_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": _format_trial(extraction_input)},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "condition_extraction",
                    "strict": True,
                    "schema": EXTRACTION_SCHEMA,
                }
            },
        },
    }
