import json
import logging

import pytest

from models.models import CanonicalizationOutput
from canonicalization.parser import parse_canonicalization_results


def make_batch_result(
    extracted_condition_id: int | str,
    *,
    common_name: str | None = None,
    technical_name: str | None = None,
    abbreviations: list[str] | None = None,
    status: str = "completed",
    text: str | None = None,
    incomplete_reason: str = "max_output_tokens",
) -> dict:
    """
    Create a minimal Batch API response matching the parser's
    expected response structure.
    """
    if status != "completed":
        return {
            "custom_id": str(extracted_condition_id),
            "response": {
                "body": {
                    "status": status,
                    "incomplete_details": {
                        "reason": incomplete_reason,
                    },
                }
            },
        }

    if text is None and common_name is not None:
        text = json.dumps(
            {
                "common_name": common_name,
                "technical_name": technical_name,
                "abbreviations": abbreviations or [],
            }
        )

    content = []

    if text is not None:
        content.append(
            {
                "type": "output_text",
                "text": text,
            }
        )

    return {
        "custom_id": str(extracted_condition_id),
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
    path = tmp_path / "canonicalization_results.jsonl"

    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    return path


def test_parse_single_canonicalization_result(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                101,
                common_name="Vitiligo",
                technical_name="Vitiligo",
                abbreviations=[],
            )
        ],
    )

    records = parse_canonicalization_results(path)

    assert records == [
        CanonicalizationOutput(
            extracted_condition_id=101,
            common_name="Vitiligo",
            technical_name="Vitiligo",
            abbreviations=[],
        )
    ]


def test_parse_condition_with_abbreviations(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                102,
                common_name="Chronic obstructive pulmonary disease",
                technical_name="Chronic obstructive pulmonary disease",
                abbreviations=["COPD"],
            )
        ],
    )

    records = parse_canonicalization_results(path)

    assert records == [
        CanonicalizationOutput(
            extracted_condition_id=102,
            common_name="Chronic obstructive pulmonary disease",
            technical_name="Chronic obstructive pulmonary disease",
            abbreviations=["COPD"],
        )
    ]


def test_parse_multiple_canonicalization_results(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                101,
                common_name="Vitiligo",
                technical_name="Vitiligo",
                abbreviations=[],
            ),
            make_batch_result(
                102,
                common_name="High blood potassium",
                technical_name="Hyperkalemia",
                abbreviations=[],
            ),
        ],
    )

    records = parse_canonicalization_results(path)

    assert records == [
        CanonicalizationOutput(
            extracted_condition_id=101,
            common_name="Vitiligo",
            technical_name="Vitiligo",
            abbreviations=[],
        ),
        CanonicalizationOutput(
            extracted_condition_id=102,
            common_name="High blood potassium",
            technical_name="Hyperkalemia",
            abbreviations=[],
        ),
    ]


def test_custom_id_is_converted_to_integer(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                "123",
                common_name="Multiple sclerosis",
                technical_name="Multiple sclerosis",
                abbreviations=["MS"],
            )
        ],
    )

    records = parse_canonicalization_results(path)

    assert records[0].extracted_condition_id == 123
    assert isinstance(records[0].extracted_condition_id, int)


def test_blank_jsonl_lines_are_ignored(tmp_path):
    result = make_batch_result(
        101,
        common_name="Vitiligo",
        technical_name="Vitiligo",
        abbreviations=[],
    )

    path = tmp_path / "canonicalization_results.jsonl"

    path.write_text(
        f"\n\n{json.dumps(result)}\n\n",
        encoding="utf-8",
    )

    records = parse_canonicalization_results(path)

    assert len(records) == 1
    assert records[0].extracted_condition_id == 101


def test_incomplete_response_is_skipped(tmp_path, caplog):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                101,
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        ],
    )

    with caplog.at_level(logging.WARNING):
        records = parse_canonicalization_results(path)

    assert records == []
    assert "Skipping 101: max_output_tokens" in caplog.text


def test_response_without_output_text_is_skipped(
    tmp_path,
    caplog,
):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                101,
                common_name=None,
                text=None,
            )
        ],
    )

    with caplog.at_level(logging.WARNING):
        records = parse_canonicalization_results(path)

    assert records == []
    assert "No extraction results for 101" in caplog.text


def test_blank_output_text_is_skipped(tmp_path, caplog):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                101,
                text="   ",
            )
        ],
    )

    with caplog.at_level(logging.WARNING):
        records = parse_canonicalization_results(path)

    assert records == []
    assert "No extraction results for 101" in caplog.text


def test_malformed_canonicalization_json_is_skipped(
    tmp_path,
    caplog,
):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                101,
                text="{not valid json",
            )
        ],
    )

    with caplog.at_level(logging.ERROR):
        records = parse_canonicalization_results(path)

    assert records == []
    assert "Failed to parse 101" in caplog.text


def test_last_nonempty_output_text_is_used(tmp_path):
    result = make_batch_result(
        101,
        common_name="Incorrect earlier result",
        technical_name="Incorrect result",
        abbreviations=[],
    )

    result["response"]["body"]["output"].append(
        {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": json.dumps(
                        {
                            "common_name": "Vitiligo",
                            "technical_name": "Vitiligo",
                            "abbreviations": [],
                        }
                    ),
                }
            ],
        }
    )

    path = write_jsonl(tmp_path, [result])

    records = parse_canonicalization_results(path)

    assert records == [
        CanonicalizationOutput(
            extracted_condition_id=101,
            common_name="Vitiligo",
            technical_name="Vitiligo",
            abbreviations=[],
        )
    ]


def test_non_message_output_items_are_ignored(tmp_path):
    result = make_batch_result(
        101,
        common_name="Vitiligo",
        technical_name="Vitiligo",
        abbreviations=[],
    )

    result["response"]["body"]["output"].append(
        {
            "type": "reasoning",
            "content": [],
        }
    )

    path = write_jsonl(tmp_path, [result])

    records = parse_canonicalization_results(path)

    assert len(records) == 1
    assert records[0].common_name == "Vitiligo"


def test_missing_required_canonicalization_field_raises_key_error(
    tmp_path,
):
    path = write_jsonl(
        tmp_path,
        [
            make_batch_result(
                101,
                text=json.dumps(
                    {
                        "common_name": "Vitiligo",
                        "abbreviations": [],
                    }
                ),
            )
        ],
    )

    with pytest.raises(KeyError, match="technical_name"):
        parse_canonicalization_results(path)