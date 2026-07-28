from collections.abc import Iterator

from canonicalization.prompts import CANONICALIZATION_PROMPT
from canonicalization.schema import CANONICALIZATION_SCHEMA
from models.models import CanonicalizationInput


DEFAULT_MODEL = "gpt-4.1-nano"


def iter_conditions(cur) -> Iterator[CanonicalizationInput]:
    """
    Iterate over extracted conditions that have not yet been
    canonicalized.
    """

    cur.execute(
        """
        SELECT
            ec.extracted_condition_id,
            ec.name
        FROM ExtractedConditions ec
        LEFT JOIN ExtractedCanonicalConditions ecc
            ON ec.extracted_condition_id = ecc.extracted_condition_id
        WHERE ecc.extracted_condition_id IS NULL
        ORDER BY ec.extracted_condition_id
        """
    )

    for extracted_condition_id, name in cur:

        yield CanonicalizationInput(
            extracted_condition_id=extracted_condition_id,
            name=name,
        )


def build_request(
    canonicalization_input: CanonicalizationInput,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Build an OpenAI Batch API request for disease canonicalization.

    Parameters
    ----------
    canonicalization_input : CanonicalizationInput
        An object containing extracted condition information for canonicalization.
    model : str, optional
        OpenAI model used for canonicalization.

    Returns
    -------
    dict
        JSON payload compatible with the OpenAI Batch API.
    """

    return {
        "custom_id": str(canonicalization_input.extracted_condition_id),
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": CANONICALIZATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": canonicalization_input.name,
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "canonicalization",
                    "strict": True,
                    "schema": CANONICALIZATION_SCHEMA,
                }
            },
        },
    }
