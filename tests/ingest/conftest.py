import json
from pathlib import Path

import pytest

from ingest.parser import parse_trial


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"


def load_protocol_section(filename: str) -> dict:
    path = FIXTURE_DIR / filename

    with path.open(encoding="utf-8") as file:
        study = json.load(file)

    return study["protocolSection"]


@pytest.fixture
def vitiligo_trial():
    study = load_protocol_section("NCT00380471.json")
    return parse_trial(study)


@pytest.fixture
def copd_trial():
    study = load_protocol_section("NCT03161587.json")
    return parse_trial(study)


@pytest.fixture
def mannitol_trial():
    study = load_protocol_section("NCT03161977.json")
    return parse_trial(study)