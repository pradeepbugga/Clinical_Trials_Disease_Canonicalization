import csv
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from finetuning.evaluate import (
    compare_example,
    compute_metrics,
    evaluate_model,
    load_examples,
    normalize_abbreviations,
    normalize_name,
    normalize_text,
    predict_condition,
    write_predictions,
)
from models.models import EvaluationExample


def make_evaluation_example(
    input_name="COPD",
    common_name="Chronic obstructive pulmonary disease",
    technical_name="Chronic obstructive pulmonary disease",
    abbreviations=None,
):
    return EvaluationExample(
        input_name=input_name,
        expected_common_name=common_name,
        expected_technical_name=technical_name,
        expected_abbreviations=abbreviations or [],
    )


def write_test_jsonl(tmp_path, records):
    path = tmp_path / "test.jsonl"

    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    return path


def make_training_record(
    input_name,
    common_name,
    technical_name,
    abbreviations,
):
    return {
        "messages": [
            {
                "role": "system",
                "content": "System prompt",
            },
            {
                "role": "user",
                "content": input_name,
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "common_name": common_name,
                        "technical_name": technical_name,
                        "abbreviations": abbreviations,
                    }
                ),
            },
        ]
    }


def test_load_examples_reads_jsonl(tmp_path):
    path = write_test_jsonl(
        tmp_path,
        [
            make_training_record(
                input_name="COPD",
                common_name="Chronic obstructive pulmonary disease",
                technical_name="Chronic obstructive pulmonary disease",
                abbreviations=["COPD"],
            ),
            make_training_record(
                input_name="High blood potassium",
                common_name="High blood potassium",
                technical_name="Hyperkalemia",
                abbreviations=[],
            ),
        ],
    )

    examples = load_examples(path)

    assert examples == [
        EvaluationExample(
            input_name="COPD",
            expected_common_name=(
                "Chronic obstructive pulmonary disease"
            ),
            expected_technical_name=(
                "Chronic obstructive pulmonary disease"
            ),
            expected_abbreviations=["COPD"],
        ),
        EvaluationExample(
            input_name="High blood potassium",
            expected_common_name="High blood potassium",
            expected_technical_name="Hyperkalemia",
            expected_abbreviations=[],
        ),
    ]


def test_load_examples_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.jsonl"

    with pytest.raises(
        FileNotFoundError,
        match="File not found",
    ):
        load_examples(missing_path)


def test_normalize_text():
    assert normalize_text("  Vitiligo  ") == "Vitiligo"
    assert normalize_text(None) == ""
    assert normalize_text(123) == "123"


def test_normalize_name_is_case_insensitive():
    assert normalize_name("  Multiple Sclerosis ") == (
        "multiple sclerosis"
    )

    assert normalize_name("MÉNIÈRE DISEASE") == (
        normalize_name("ménière disease")
    )


def test_normalize_abbreviations_is_order_and_case_insensitive():
    result = normalize_abbreviations(
        ["COPD", " copd ", "MS", "", None]
    )

    assert result == {"copd", "ms"}


def test_normalize_abbreviations_returns_empty_set_for_none():
    assert normalize_abbreviations(None) == set()
    assert normalize_abbreviations([]) == set()


def test_compare_example_exact_match():
    example = make_evaluation_example(
        abbreviations=["COPD"],
    )

    prediction = {
        "common_name": "Chronic obstructive pulmonary disease",
        "technical_name": "Chronic obstructive pulmonary disease",
        "abbreviations": ["COPD"],
    }

    result = compare_example(example, prediction)

    assert result == {
        "common_correct": True,
        "technical_correct": True,
        "abbreviation_tp": 1,
        "abbreviation_fp": 0,
        "abbreviation_fn": 0,
    }


def test_compare_example_ignores_case_and_whitespace():
    example = make_evaluation_example(
        common_name="Multiple sclerosis",
        technical_name="Multiple sclerosis",
        abbreviations=["MS"],
    )

    prediction = {
        "common_name": "  MULTIPLE SCLEROSIS ",
        "technical_name": "multiple sclerosis",
        "abbreviations": ["ms"],
    }

    result = compare_example(example, prediction)

    assert result["common_correct"] is True
    assert result["technical_correct"] is True
    assert result["abbreviation_tp"] == 1
    assert result["abbreviation_fp"] == 0
    assert result["abbreviation_fn"] == 0


def test_compare_example_counts_abbreviation_errors():
    example = make_evaluation_example(
        abbreviations=["COPD", "COLD"],
    )

    prediction = {
        "common_name": "Wrong common name",
        "technical_name": (
            "Chronic obstructive pulmonary disease"
        ),
        "abbreviations": ["COPD", "ILD"],
    }

    result = compare_example(example, prediction)

    assert result == {
        "common_correct": False,
        "technical_correct": True,
        "abbreviation_tp": 1,
        "abbreviation_fp": 1,
        "abbreviation_fn": 1,
    }


def test_compare_example_treats_none_and_empty_string_as_equal():
    example = make_evaluation_example(
        technical_name=None,
    )

    prediction = {
        "common_name": (
            "Chronic obstructive pulmonary disease"
        ),
        "technical_name": "",
        "abbreviations": [],
    }

    result = compare_example(example, prediction)

    assert result["technical_correct"] is True

def test_compute_metrics_handles_empty_results():
    assert compute_metrics([]) == {
        "common_accuracy": 0.0,
        "technical_accuracy": 0.0,
        "abbreviation_precision": 1.0,
        "abbreviation_recall": 1.0,
        "abbreviation_f1": 1.0,
    }
    
def test_compute_metrics_perfect_results():
    results = [
        {
            "common_correct": True,
            "technical_correct": True,
            "abbreviation_tp": 2,
            "abbreviation_fp": 0,
            "abbreviation_fn": 0,
        },
        {
            "common_correct": True,
            "technical_correct": True,
            "abbreviation_tp": 1,
            "abbreviation_fp": 0,
            "abbreviation_fn": 0,
        },
    ]

    metrics = compute_metrics(results)

    assert metrics == {
        "common_accuracy": 1.0,
        "technical_accuracy": 1.0,
        "abbreviation_precision": 1.0,
        "abbreviation_recall": 1.0,
        "abbreviation_f1": 1.0,
    }


def test_compute_metrics_partial_results():
    results = [
        {
            "common_correct": True,
            "technical_correct": False,
            "abbreviation_tp": 1,
            "abbreviation_fp": 1,
            "abbreviation_fn": 0,
        },
        {
            "common_correct": False,
            "technical_correct": True,
            "abbreviation_tp": 1,
            "abbreviation_fp": 0,
            "abbreviation_fn": 1,
        },
    ]

    metrics = compute_metrics(results)

    assert metrics["common_accuracy"] == 0.5
    assert metrics["technical_accuracy"] == 0.5
    assert metrics["abbreviation_precision"] == pytest.approx(
        2 / 3
    )
    assert metrics["abbreviation_recall"] == pytest.approx(
        2 / 3
    )
    assert metrics["abbreviation_f1"] == pytest.approx(
        2 / 3
    )


def test_compute_metrics_without_any_abbreviations():
    results = [
        {
            "common_correct": True,
            "technical_correct": True,
            "abbreviation_tp": 0,
            "abbreviation_fp": 0,
            "abbreviation_fn": 0,
        }
    ]

    metrics = compute_metrics(results)

    assert metrics["abbreviation_precision"] == 1.0
    assert metrics["abbreviation_recall"] == 1.0
    assert metrics["abbreviation_f1"] == 1.0


def test_predict_condition_returns_valid_prediction():
    response = SimpleNamespace(
        output_text=json.dumps(
            {
                "common_name": "High blood potassium",
                "technical_name": "Hyperkalemia",
                "abbreviations": [],
            }
        )
    )

    client = Mock()
    client.responses.create.return_value = response

    prediction = predict_condition(
        client=client,
        model="test-model",
        condition="High blood potassium",
    )

    assert prediction == {
        "common_name": "High blood potassium",
        "technical_name": "Hyperkalemia",
        "abbreviations": [],
    }

    client.responses.create.assert_called_once()

    kwargs = client.responses.create.call_args.kwargs

    assert kwargs["model"] == "test-model"
    assert kwargs["input"][1] == {
        "role": "user",
        "content": "High blood potassium",
    }


def test_predict_condition_raises_for_invalid_json():
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="{not valid json"
    )

    with pytest.raises(json.JSONDecodeError):
        predict_condition(
            client=client,
            model="test-model",
            condition="Vitiligo",
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "common_name",
        "technical_name",
        "abbreviations",
    ],
)
def test_predict_condition_raises_for_missing_fields(
    missing_field,
):
    prediction = {
        "common_name": "Vitiligo",
        "technical_name": "Vitiligo",
        "abbreviations": [],
    }

    del prediction[missing_field]

    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(prediction)
    )

    with pytest.raises(
        ValueError,
        match="missing fields",
    ):
        predict_condition(
            client=client,
            model="test-model",
            condition="Vitiligo",
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("common_name", None),
        ("common_name", ["Vitiligo"]),
        ("technical_name", None),
        ("technical_name", 123),
        ("abbreviations", "MS"),
        ("abbreviations", None),
    ],
)
def test_predict_condition_validates_field_types(
    field,
    invalid_value,
):
    prediction = {
        "common_name": "Multiple sclerosis",
        "technical_name": "Multiple sclerosis",
        "abbreviations": ["MS"],
    }

    prediction[field] = invalid_value

    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(prediction)
    )

    with pytest.raises(TypeError):
        predict_condition(
            client=client,
            model="test-model",
            condition="MS",
        )


def test_predict_condition_requires_string_abbreviations():
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "common_name": "Multiple sclerosis",
                "technical_name": "Multiple sclerosis",
                "abbreviations": ["MS", 123],
            }
        )
    )

    with pytest.raises(
        TypeError,
        match="Every abbreviation must be a string",
    ):
        predict_condition(
            client=client,
            model="test-model",
            condition="MS",
        )


def test_write_predictions_creates_csv(tmp_path):
    output_path = tmp_path / "predictions.csv"

    rows = [
        {
            "input": "MS",
            "common_correct": True,
        },
        {
            "input": "COPD",
            "common_correct": False,
        },
    ]

    write_predictions(rows, output_path)

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        saved_rows = list(csv.DictReader(file))

    assert saved_rows == [
        {
            "input": "MS",
            "common_correct": "True",
        },
        {
            "input": "COPD",
            "common_correct": "False",
        },
    ]


def test_write_predictions_does_not_create_file_for_empty_rows(
    tmp_path,
):
    output_path = tmp_path / "predictions.csv"

    write_predictions([], output_path)

    assert not output_path.exists()


def test_evaluate_model_returns_metrics_and_writes_predictions(
    tmp_path,
    monkeypatch,
):
    examples = [
        EvaluationExample(
            input_name="MS",
            expected_common_name="Multiple sclerosis",
            expected_technical_name="Multiple sclerosis",
            expected_abbreviations=["MS"],
        ),
        EvaluationExample(
            input_name="High blood potassium",
            expected_common_name="High blood potassium",
            expected_technical_name="Hyperkalemia",
            expected_abbreviations=[],
        ),
    ]

    predictions = {
        "MS": {
            "common_name": "Multiple sclerosis",
            "technical_name": "Multiple sclerosis",
            "abbreviations": ["MS"],
        },
        "High blood potassium": {
            "common_name": "High blood potassium",
            "technical_name": "Hyperkalemia",
            "abbreviations": [],
        },
    }

    def fake_predict_condition(client, model, condition):
        return predictions[condition]

    monkeypatch.setattr(
        "finetuning.evaluate.predict_condition",
        fake_predict_condition,
    )

    output_path = tmp_path / "predictions.csv"

    metrics = evaluate_model(
        client=Mock(),
        model="test-model",
        examples=examples,
        output_path=output_path,
    )

    assert metrics == {
        "common_accuracy": 1.0,
        "technical_accuracy": 1.0,
        "abbreviation_precision": 1.0,
        "abbreviation_recall": 1.0,
        "abbreviation_f1": 1.0,
    }

    assert output_path.exists()

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["input"] == "MS"
    assert rows[0]["common_correct"] == "True"