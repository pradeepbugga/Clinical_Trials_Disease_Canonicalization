from collections import deque
from unittest.mock import Mock

import pytest

from ingest.parser import parse_trial
from ingest.persistence import (
    insert_conditions,
    insert_interventions,
    insert_trial,
)


@pytest.fixture
def trial():
    return parse_trial(
        {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Example Trial",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2024-01-15"},
                "completionDateStruct": {"date": "2025-02"},
            },
            "descriptionModule": {
                "briefSummary": "Example summary",
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Example Sponsor"},
            },
            "designModule": {
                "phases": ["PHASE2"],
            },
            "conditionsModule": {
                "conditions": ["Asthma", "Diabetes"],
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"name": "Drug A", "type": "DRUG"},
                    {"name": "Device B", "type": "DEVICE"},
                ],
            },
        }
    )


def normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def test_insert_trial_executes_expected_insert(trial):
    cur = Mock()

    insert_trial(cur, trial)

    cur.execute.assert_called_once()
    sql, params = cur.execute.call_args.args

    assert "INSERT INTO ClinicalTrials" in normalize_sql(sql)
    assert "ON CONFLICT DO NOTHING" in normalize_sql(sql)
    assert params == (
        "NCT12345678",
        "Example Trial",
        "COMPLETED",
        "PHASE2",
        "Example summary",
        "2024-01-15T00:00:00",
        "2025-02-01T00:00:00",
        None,
        "Example Sponsor",
        "https://clinicaltrials.gov/study/NCT12345678",
    )


def test_insert_trial_writes_none_for_missing_dates():
    trial = parse_trial(
        {
            "identificationModule": {
                "nctId": "NCT00000001",
            }
        }
    )
    cur = Mock()

    insert_trial(cur, trial)

    _, params = cur.execute.call_args.args
    assert params[5] is None
    assert params[6] is None


def test_insert_conditions_inserts_and_links_each_condition(trial):
    cur = Mock()
    cur.fetchone.side_effect = [(101,), (202,)]

    insert_conditions(cur, trial)

    assert cur.execute.call_count == 6

    calls = cur.execute.call_args_list

    assert "INSERT INTO Conditions" in normalize_sql(calls[0].args[0])
    assert calls[0].args[1] == ("Asthma",)

    assert "SELECT condition_id" in normalize_sql(calls[1].args[0])
    assert calls[1].args[1] == ("Asthma",)

    assert "INSERT INTO TrialConditions" in normalize_sql(calls[2].args[0])
    assert calls[2].args[1] == ("NCT12345678", 101)

    assert "INSERT INTO Conditions" in normalize_sql(calls[3].args[0])
    assert calls[3].args[1] == ("Diabetes",)

    assert "SELECT condition_id" in normalize_sql(calls[4].args[0])
    assert calls[4].args[1] == ("Diabetes",)

    assert "INSERT INTO TrialConditions" in normalize_sql(calls[5].args[0])
    assert calls[5].args[1] == ("NCT12345678", 202)


def test_insert_conditions_raises_when_inserted_condition_cannot_be_found(trial):
    cur = Mock()
    cur.fetchone.return_value = None

    with pytest.raises(
        RuntimeError,
        match="Condition 'Asthma' not found",
    ):
        insert_conditions(cur, trial)


def test_insert_conditions_does_nothing_for_empty_condition_list():
    trial = parse_trial(
        {
            "identificationModule": {
                "nctId": "NCT00000002",
            }
        }
    )
    cur = Mock()

    insert_conditions(cur, trial)

    cur.execute.assert_not_called()


def test_insert_interventions_inserts_and_links_each_intervention(trial):
    cur = Mock()
    cur.fetchone.side_effect = [(301,), (302,)]

    insert_interventions(cur, trial)

    assert cur.execute.call_count == 6

    calls = cur.execute.call_args_list

    assert "INSERT INTO Interventions" in normalize_sql(calls[0].args[0])
    assert calls[0].args[1] == ("Drug A", "DRUG")

    assert "SELECT intervention_id" in normalize_sql(calls[1].args[0])
    assert calls[1].args[1] == ("Drug A",)

    assert "INSERT INTO TrialInterventions" in normalize_sql(calls[2].args[0])
    assert calls[2].args[1] == ("NCT12345678", 301)

    assert "INSERT INTO Interventions" in normalize_sql(calls[3].args[0])
    assert calls[3].args[1] == ("Device B", "DEVICE")

    assert "SELECT intervention_id" in normalize_sql(calls[4].args[0])
    assert calls[4].args[1] == ("Device B",)

    assert "INSERT INTO TrialInterventions" in normalize_sql(calls[5].args[0])
    assert calls[5].args[1] == ("NCT12345678", 302)


def test_insert_interventions_raises_when_inserted_intervention_cannot_be_found(
    trial,
):
    cur = Mock()
    cur.fetchone.return_value = None

    with pytest.raises(
        RuntimeError,
        match="Intervention 'Drug A' not found",
    ):
        insert_interventions(cur, trial)


def test_insert_interventions_does_nothing_for_empty_intervention_list():
    trial = parse_trial(
        {
            "identificationModule": {
                "nctId": "NCT00000003",
            }
        }
    )
    cur = Mock()

    insert_interventions(cur, trial)

    cur.execute.assert_not_called()