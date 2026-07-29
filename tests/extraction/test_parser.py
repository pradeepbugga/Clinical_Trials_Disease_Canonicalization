import json
import logging

from extraction.parser import parse_condition_results


def make_batch_result(
    nct_id: str,
    conditions: list[str] | None = None,
    *,
    status: str = "completed",
    text: str | None = None,
    incomplete_reason: str = "max_output_tokens",
) -> dict:
    """
    Create a minimal OpenAI Batch API response matching the parser's
    expected structure.
    """
    if status != "completed":
        return {
            "custom_id": nct_id,
            "response": {
                "body": {
                    "status": status,
                    "incomplete_details": {
                        "reason": incomplete_reason,
                    },
                }
            },
        }

    if text is None and conditions is not None:
        text = json.dumps({"conditions": conditions})

    content = []

    if text is not None:
        content.append(
            {
                "type": "output_text",
                "text": text,
            }
        )

    return {
        "custom_id": nct_id,
        "response": {
            "body": {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": content,
                    }
                ],
            }
        },
    }


def write_jsonl(tmp_path, records: list[dict]):
    path = tmp_path / "batch_results.jsonl"

    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    return path


def test_parse_single_condition(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                "NCT00380471",
                ["Vitiligo"],
            )
        ],
    )

    records = parse_condition_results(path)

    assert records == [
        {
            "nct_id": "NCT00380471",
            "conditions": ["Vitiligo"],
        }
    ]


def test_parse_multiple_conditions(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                "NCT03161977",
                [
                    "Mannitol Adverse Reaction",
                    "Hyperkalemia",
                ],
            )
        ],
    )

    records = parse_condition_results(path)

    assert records == [
        {
            "nct_id": "NCT03161977",
            "conditions": [
                "Mannitol Adverse Reaction",
                "Hyperkalemia",
            ],
        }
    ]


def test_parse_multiple_batch_results(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                "NCT00380471",
                ["Vitiligo"],
            ),
            make_batch_result(
                "NCT03161977",
                [
                    "Mannitol Adverse Reaction",
                    "Hyperkalemia",
                ],
            ),
        ],
    )

    records = parse_condition_results(path)

    assert records == [
        {
            "nct_id": "NCT00380471",
            "conditions": ["Vitiligo"],
        },
        {
            "nct_id": "NCT03161977",
            "conditions": [
                "Mannitol Adverse Reaction",
                "Hyperkalemia",
            ],
        },
    ]


def test_blank_jsonl_lines_are_ignored(tmp_path):
    result = make_batch_result(
        "NCT00380471",
        ["Vitiligo"],
    )

    path = tmp_path / "batch_results.jsonl"

    path.write_text(
        f"\n\n{json.dumps(result)}\n\n",
        encoding="utf-8",
    )

    records = parse_condition_results(path)

    assert records == [
        {
            "nct_id": "NCT00380471",
            "conditions": ["Vitiligo"],
        }
    ]


def test_incomplete_response_is_skipped(tmp_path, caplog):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                "NCT00380471",
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        ],
    )

    with caplog.at_level(logging.WARNING):
        records = parse_condition_results(path)

    assert records == []
    assert "Skipping NCT00380471: max_output_tokens" in caplog.text


def test_response_without_output_text_is_skipped(
    tmp_path,
    caplog,
):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                "NCT00380471",
                conditions=None,
                text=None,
            )
        ],
    )

    with caplog.at_level(logging.WARNING):
        records = parse_condition_results(path)

    assert records == []
    assert "No extraction results for NCT00380471" in caplog.text


def test_blank_output_text_is_skipped(tmp_path, caplog):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                "NCT00380471",
                text="   ",
            )
        ],
    )

    with caplog.at_level(logging.WARNING):
        records = parse_condition_results(path)

    assert records == []
    assert "No extraction results for NCT00380471" in caplog.text


def test_malformed_extraction_json_is_skipped(
    tmp_path,
    caplog,
):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                "NCT00380471",
                text="{not valid json",
            )
        ],
    )

    with caplog.at_level(logging.ERROR):
        records = parse_condition_results(path)

    assert records == []
    assert "Failed to parse NCT00380471" in caplog.text


def test_last_nonempty_output_text_is_used(tmp_path):
    result = make_batch_result(
        "NCT00380471",
        ["Incorrect earlier result"],
    )

    result["response"]["body"]["output"].append(
        {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": json.dumps(
                        {
                            "conditions": ["Vitiligo"],
                        }
                    ),
                }
            ],
        }
    )

    path = write_jsonl(tmp_path, [result])

    records = parse_condition_results(path)

    assert records == [
        {
            "nct_id": "NCT00380471",
            "conditions": ["Vitiligo"],
        }
    ]


def test_non_message_output_items_are_ignored(tmp_path):
    result = make_batch_result(
        "NCT00380471",
        ["Vitiligo"],
    )

    result["response"]["body"]["output"].append(
        {
            "type": "reasoning",
            "content": [],
        }
    )

    path = write_jsonl(tmp_path, [result])

    records = parse_condition_results(path)

    assert records == [
        {
            "nct_id": "NCT00380471",
            "conditions": ["Vitiligo"],
        }
    ]