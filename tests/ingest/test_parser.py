from datetime import datetime

import pytest

from ingest.parser import parse_date, parse_trial


def test_parse_date_full_date():
    assert parse_date("2026-07-28") == datetime(2026, 7, 28)


def test_parse_date_year_month_defaults_to_first_day():
    assert parse_date("2026-07") == datetime(2026, 7, 1)


@pytest.mark.parametrize("value", [None, "", "not-a-date", "2026-13", "07/28/2026"])
def test_parse_date_returns_none_for_missing_or_invalid_values(value):
    assert parse_date(value) is None


def test_parse_trial_parses_complete_study():
    study = {
        "identificationModule": {
            "nctId": "NCT12345678",
            "briefTitle": "  Trial of Drug A  ",
        },
        "statusModule": {
            "overallStatus": "  RECRUITING  ",
            "startDateStruct": {"date": "2025-03"},
            "completionDateStruct": {"date": "2026-11-15"},
            "studyFirstSubmitDate": "2025-01-10",
        },
        "descriptionModule": {
            "briefSummary": "  A trial summary.  ",
        },
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"name": "Example University"},
        },
        "designModule": {
            "phases": ["PHASE2", "PHASE3"],
        },
        "armsInterventionsModule": {
            "interventions": [
                {"name": "  Drug A  ", "type": "  DRUG  "},
                {"name": "Behavioral Program", "type": "BEHAVIORAL"},
            ],
        },
        "conditionsModule": {
            "conditions": [
                "  Multiple Sclerosis  ",
                "Fatigue",
            ],
        },
    }

    trial = parse_trial(study)

    assert trial.nct_id == "NCT12345678"
    assert trial.title == "Trial of Drug A"
    assert trial.status == "RECRUITING"
    assert trial.phase == "PHASE2"
    assert trial.summary == "A trial summary."
    assert trial.start_date == datetime(2025, 3, 1)
    assert trial.end_date == datetime(2026, 11, 15)
    assert trial.sponsor == "Example University"
    assert trial.url == "https://clinicaltrials.gov/study/NCT12345678"
    assert trial.conditions == ["Multiple Sclerosis", "Fatigue"]

    assert len(trial.interventions) == 2
    assert trial.interventions[0].name == "Drug A"
    assert trial.interventions[0].type == "DRUG"
    assert trial.interventions[1].name == "Behavioral Program"
    assert trial.interventions[1].type == "BEHAVIORAL"


def test_parse_trial_uses_submit_date_when_start_date_is_missing():
    study = {
        "identificationModule": {
            "nctId": "NCT00000001",
            "briefTitle": "Fallback date test",
        },
        "statusModule": {
            "studyFirstSubmitDate": "2024-09-20",
        },
    }

    trial = parse_trial(study)

    assert trial.start_date == datetime(2024, 9, 20)


def test_parse_trial_defaults_missing_optional_fields():
    study = {
        "identificationModule": {
            "nctId": "NCT00000002",
        },
    }

    trial = parse_trial(study)

    assert trial.title == ""
    assert trial.status == ""
    assert trial.phase == ""
    assert trial.summary == ""
    assert trial.start_date is None
    assert trial.end_date is None
    assert trial.sponsor == ""
    assert trial.conditions == []
    assert trial.interventions == []
    assert trial.url == "https://clinicaltrials.gov/study/NCT00000002"


def test_parse_trial_filters_blank_conditions_and_unnamed_interventions():
    study = {
        "identificationModule": {
            "nctId": "NCT00000003",
        },
        "conditionsModule": {
            "conditions": ["Asthma", "  ", ""],
        },
        "armsInterventionsModule": {
            "interventions": [
                {"name": "Albuterol", "type": "DRUG"},
                {"name": "", "type": "DRUG"},
                {"type": "DEVICE"},
            ],
        },
    }

    trial = parse_trial(study)

    assert trial.conditions == ["Asthma"]
    assert len(trial.interventions) == 1
    assert trial.interventions[0].name == "Albuterol"


def test_parse_trial_requires_nct_id():
    with pytest.raises(KeyError):
        parse_trial({"identificationModule": {}})