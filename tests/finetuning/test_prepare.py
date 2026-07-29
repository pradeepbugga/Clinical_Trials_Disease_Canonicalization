import json

import pytest

from finetuning.prepare import (
    build_messages,
    iter_training_examples,
    prepare,
    write_jsonl,
)
from finetuning.prompts import SYSTEM_PROMPT
from models.models import FineTuningExample


def write_input_json(tmp_path, records: list[dict]):
    path = tmp_path / "canonical_mappings.json"
    path.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_iter_training_examples_preserves_distinct_technical_name(
    tmp_path,
):
    input_path = write_input_json(
        tmp_path,
        [
            {
                "input_name": "High blood potassium",
                "common_name": "High blood potassium",
                "technical_name": "Hyperkalemia",
                "abbreviations": [],
            }
        ],
    )

    examples = list(iter_training_examples(input_path))

    assert examples == [
        FineTuningExample(
            input_name="High blood potassium",
            common_name="High blood potassium",
            technical_name="Hyperkalemia",
            abbreviations=[],
        )
    ]


def test_iter_training_examples_sets_duplicate_technical_name_to_none(
    tmp_path,
):
    input_path = write_input_json(
        tmp_path,
        [
            {
                "input_name": "Vitiligo",
                "common_name": "Vitiligo",
                "technical_name": "  VITILIGO  ",
                "abbreviations": [],
            }
        ],
    )

    examples = list(iter_training_examples(input_path))

    assert examples[0].technical_name is None


def test_iter_training_examples_preserves_none_technical_name(
    tmp_path,
):
    input_path = write_input_json(
        tmp_path,
        [
            {
                "input_name": "Vitiligo",
                "common_name": "Vitiligo",
                "technical_name": None,
                "abbreviations": [],
            }
        ],
    )

    examples = list(iter_training_examples(input_path))

    assert examples[0].technical_name is None


def test_iter_training_examples_preserves_abbreviations(tmp_path):
    input_path = write_input_json(
        tmp_path,
        [
            {
                "input_name": "Chronic obstructive lung disease",
                "common_name": (
                    "Chronic obstructive pulmonary disease"
                ),
                "technical_name": None,
                "abbreviations": ["COPD"],
            }
        ],
    )

    examples = list(iter_training_examples(input_path))

    assert examples[0].abbreviations == ["COPD"]


def test_iter_training_examples_normalizes_missing_abbreviations(
    tmp_path,
):
    input_path = write_input_json(
        tmp_path,
        [
            {
                "input_name": "Vitiligo",
                "common_name": "Vitiligo",
                "technical_name": None,
                "abbreviations": None,
            }
        ],
    )

    examples = list(iter_training_examples(input_path))

    assert examples[0].abbreviations == []

def test_build_messages_creates_expected_training_record():
    example = FineTuningExample(
        input_name="High blood potassium",
        common_name="High blood potassium",
        technical_name="Hyperkalemia",
        abbreviations=[],
    )

    result = build_messages(example)

    assert result["messages"][0] == {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }

    assert result["messages"][1] == {
        "role": "user",
        "content": "High blood potassium",
    }

    assert result["messages"][2]["role"] == "assistant"

    assistant_content = json.loads(
        result["messages"][2]["content"]
    )

    assert assistant_content == {
        "common_name": "High blood potassium",
        "technical_name": "Hyperkalemia",
        "abbreviations": [],
    }


def test_build_messages_serializes_none_as_json_null():
    example = FineTuningExample(
        input_name="Vitiligo",
        common_name="Vitiligo",
        technical_name=None,
        abbreviations=[],
    )

    result = build_messages(example)

    assistant_content = json.loads(
        result["messages"][2]["content"]
    )

    assert assistant_content["technical_name"] is None


def test_build_messages_preserves_unicode():
    example = FineTuningExample(
        input_name="Maladie de Ménière",
        common_name="Ménière disease",
        technical_name=None,
        abbreviations=[],
    )

    result = build_messages(example)

    assistant_text = result["messages"][2]["content"]

    assert "Ménière" in assistant_text
    assert "\\u00e9" not in assistant_text


def test_write_jsonl_writes_one_record_per_line(tmp_path):
    output_path = tmp_path / "train.jsonl"

    records = [
        FineTuningExample(
            input_name="Vitiligo",
            common_name="Vitiligo",
            technical_name=None,
            abbreviations=[],
        ),
        FineTuningExample(
            input_name="High blood potassium",
            common_name="High blood potassium",
            technical_name="Hyperkalemia",
            abbreviations=[],
        ),
    ]

    write_jsonl(records, output_path)

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    parsed_records = [json.loads(line) for line in lines]

    assert parsed_records[0] == build_messages(records[0])
    assert parsed_records[1] == build_messages(records[1])


def test_write_jsonl_empty_input_creates_empty_file(tmp_path):
    output_path = tmp_path / "train.jsonl"

    write_jsonl([], output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == ""


def test_prepare_splits_training_and_test_data(tmp_path):
    input_records = [
        {
            "input_name": f"Condition {index}",
            "common_name": f"Condition {index}",
            "technical_name": None,
            "abbreviations": [],
        }
        for index in range(10)
    ]

    input_path = write_input_json(tmp_path, input_records)
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"

    prepare(
        input_path=input_path,
        train_output_path=train_path,
        test_output_path=test_path,
        test_fraction=0.2,
        seed=42,
    )

    train_records = read_jsonl(train_path)
    test_records = read_jsonl(test_path)

    assert len(train_records) == 8
    assert len(test_records) == 2


def test_prepare_is_reproducible_with_same_seed(tmp_path):
    input_records = [
        {
            "input_name": f"Condition {index}",
            "common_name": f"Condition {index}",
            "technical_name": None,
            "abbreviations": [],
        }
        for index in range(10)
    ]

    input_path = write_input_json(tmp_path, input_records)

    first_train = tmp_path / "first_train.jsonl"
    first_test = tmp_path / "first_test.jsonl"

    second_train = tmp_path / "second_train.jsonl"
    second_test = tmp_path / "second_test.jsonl"

    prepare(
        input_path,
        first_train,
        first_test,
        test_fraction=0.2,
        seed=42,
    )

    prepare(
        input_path,
        second_train,
        second_test,
        test_fraction=0.2,
        seed=42,
    )

    assert first_train.read_text(
        encoding="utf-8"
    ) == second_train.read_text(encoding="utf-8")

    assert first_test.read_text(
        encoding="utf-8"
    ) == second_test.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "test_fraction",
    [-0.01, 1.01, -1, 2],
)
def test_prepare_rejects_invalid_test_fraction(
    tmp_path,
    test_fraction,
):
    input_path = write_input_json(tmp_path, [])

    with pytest.raises(
        ValueError,
        match="test_fraction must be between 0 and 1",
    ):
        prepare(
            input_path=input_path,
            train_output_path=tmp_path / "train.jsonl",
            test_output_path=tmp_path / "test.jsonl",
            test_fraction=test_fraction,
        )


def test_prepare_with_zero_test_fraction_puts_all_records_in_train(
    tmp_path,
):
    input_path = write_input_json(
        tmp_path,
        [
            {
                "input_name": "Vitiligo",
                "common_name": "Vitiligo",
                "technical_name": None,
                "abbreviations": [],
            }
        ],
    )

    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"

    prepare(
        input_path,
        train_path,
        test_path,
        test_fraction=0,
    )

    assert len(read_jsonl(train_path)) == 1
    assert read_jsonl(test_path) == []


def test_prepare_with_full_test_fraction_puts_all_records_in_test(
    tmp_path,
):
    input_path = write_input_json(
        tmp_path,
        [
            {
                "input_name": "Vitiligo",
                "common_name": "Vitiligo",
                "technical_name": None,
                "abbreviations": [],
            }
        ],
    )

    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"

    prepare(
        input_path,
        train_path,
        test_path,
        test_fraction=1,
    )

    assert read_jsonl(train_path) == []
    assert len(read_jsonl(test_path)) == 1

