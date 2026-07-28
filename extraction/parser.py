import json
from pathlib import Path


def parse_condition_results(batch_results: str) -> list[dict]:
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

    results = (json.loads(line) for line in batch_results.splitlines() if line.strip())

    for result in results:

        nct_id = result["custom_id"]

        output = result["response"]["body"]["output"]

        message = next(item for item in output if item["type"] == "message")

        extraction = json.loads(message["content"][0]["text"])

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
