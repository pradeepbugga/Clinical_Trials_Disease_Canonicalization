
import pytest

from ingest.persistence import (
    insert_conditions,
    insert_interventions,
    insert_trial,
)


def normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def test_insert_trial_executes_expected_query(vitiligo_trial):
    cursor = Mock()

    insert_trial(cursor, vitiligo_trial)

    cursor.execute.assert_called_once()

    sql, params = cursor.execute.call_args.args

    normalized_sql = normalize_sql(sql)

    assert "INSERT INTO ClinicalTrials" in normalized_sql
    assert "ON CONFLICT DO NOTHING" in normalized_sql

    assert params == (
        "NCT00380471",
        "Treatment and Complication of Bath PUVA in Vitiligo",
        "UNKNOWN",
        "PHASE2",
        (
            "The purpose of this study is to determine whether bath "
            "PUVA are effective in treatment of vitiligo and what is "
            "the complication of bath PUVA in vitiligo."
        ),
        "2006-01-01T00:00:00",
        "2006-08-01T00:00:00",
        None,
        "Shahid Beheshti University of Medical Sciences",
        "https://clinicaltrials.gov/study/NCT00380471",
    )


def test_insert_conditions_inserts_and_links_single_condition(
    vitiligo_trial,
):
    cursor = Mock()
    cursor.fetchone.return_value = (101,)

    insert_conditions(cursor, vitiligo_trial)

    assert cursor.execute.call_count == 3

    insert_call = cursor.execute.call_args_list[0]
    select_call = cursor.execute.call_args_list[1]
    link_call = cursor.execute.call_args_list[2]

    assert "INSERT INTO Conditions" in normalize_sql(
        insert_call.args[0]
    )
    assert insert_call.args[1] == ("Vitiligo",)

    assert "SELECT condition_id" in normalize_sql(
        select_call.args[0]
    )
    assert select_call.args[1] == ("Vitiligo",)

    assert "INSERT INTO TrialConditions" in normalize_sql(
        link_call.args[0]
    )
    assert link_call.args[1] == ("NCT00380471", 101)


def test_insert_conditions_inserts_multiple_conditions(
    mannitol_trial,
):
    cursor = Mock()
    cursor.fetchone.side_effect = [
        (201,),
        (202,),
    ]

    insert_conditions(cursor, mannitol_trial)

    assert cursor.execute.call_count == 6

    calls = cursor.execute.call_args_list

    assert calls[0].args[1] == ("Mannitol Adverse Reaction",)
    assert calls[1].args[1] == ("Mannitol Adverse Reaction",)
    assert calls[2].args[1] == ("NCT03161977", 201)

    assert calls[3].args[1] == ("Hyperkalemia",)
    assert calls[4].args[1] == ("Hyperkalemia",)
    assert calls[5].args[1] == ("NCT03161977", 202)


def test_insert_conditions_raises_when_condition_not_found(
    vitiligo_trial,
):
    cursor = Mock()
    cursor.fetchone.return_value = None

    with pytest.raises(
        RuntimeError,
        match="Condition 'Vitiligo' not found",
    ):
        insert_conditions(cursor, vitiligo_trial)


def test_insert_intervention_inserts_and_links_intervention(
    vitiligo_trial,
):
    cursor = Mock()
    cursor.fetchone.return_value = (301,)

    insert_interventions(cursor, vitiligo_trial)

    assert cursor.execute.call_count == 3

    insert_call = cursor.execute.call_args_list[0]
    select_call = cursor.execute.call_args_list[1]
    link_call = cursor.execute.call_args_list[2]

    assert "INSERT INTO Interventions" in normalize_sql(
        insert_call.args[0]
    )
    assert insert_call.args[1] == ("Bath PUVA", "DEVICE")

    assert "SELECT intervention_id" in normalize_sql(
        select_call.args[0]
    )
    assert select_call.args[1] == ("Bath PUVA",)

    assert "INSERT INTO TrialInterventions" in normalize_sql(
        link_call.args[0]
    )
    assert link_call.args[1] == ("NCT00380471", 301)


def test_insert_intervention_from_observational_trial(
    mannitol_trial,
):
    cursor = Mock()
    cursor.fetchone.return_value = (401,)

    insert_interventions(cursor, mannitol_trial)

    assert cursor.execute.call_count == 3

    calls = cursor.execute.call_args_list

    assert calls[0].args[1] == ("Mannitol", "OTHER")
    assert calls[1].args[1] == ("Mannitol",)
    assert calls[2].args[1] == ("NCT03161977", 401)


def test_insert_interventions_does_nothing_when_none_exist(
    copd_trial,
):
    cursor = Mock()

    insert_interventions(cursor, copd_trial)

    cursor.execute.assert_not_called()


def test_insert_interventions_raises_when_intervention_not_found(
    vitiligo_trial,
):
    cursor = Mock()
    cursor.fetchone.return_value = None

    with pytest.raises(
        RuntimeError,
        match="Intervention 'Bath PUVA' not found",
    ):
        insert_interventions(cursor, vitiligo_trial)