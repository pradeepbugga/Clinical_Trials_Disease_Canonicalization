from typing import Any
from models import Trial
from extraction.prompts import EXTRACTION_PROMPT
from extraction.schema import EXTRACTION_SCHEMA

DEFAULT_MODEL = "gpt-4.1-nano"


def iter_trials(cur) -> Iterator[Trial]:
    """
    Yield Trial objects requiring condition extraction.

    Parameters
    ----------
    cur
        Database cursor.

    Yields
    ------
    Trial
        Trial objects to process.
    """

    cur.execute(
        """
        SELECT
            nct_id,
            title,
            summary
        FROM ClinicalTrials
        WHERE nct_id NOT IN (
            SELECT DISTINCT nct_id
            FROM TrialConditions
        )
        """
    )

    while True:

        rows = cur.fetchmany(1000)

        if not rows:
            break

        for nct_id, title, summary in rows:

            yield Trial(
                nct_id=nct_id,
                title=title,
                summary=summary,
            )


def _format_trial(trial: Trial) -> str:
    """Convert a clinical trial into text for disease extraction."""

    parts = []

    if trial.summary:
        parts.append(trial.summary.strip())

    if trial.detailed_description:
        parts.append(trial.detailed_description.strip())

    return "\n\n".join(parts)


def build_request(
    trial: Trial, prompt: str = EXTRACTION_PROMPT, model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    """
    Build an OpenAI Batch API request for disease extraction.

    Parameters
    ----------
    trial_text : str
        Trial text to analyze.
    model : str, optional
        OpenAI model used for extraction.

    Returns
    -------
    dict
        JSON payload compatible with the OpenAI Batch API.
    """
    return {
        "custom_id": trial.nct_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": _format_trial(trial)},
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
