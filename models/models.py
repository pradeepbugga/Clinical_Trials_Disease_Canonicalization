from dataclasses import dataclass
from datetime import datetime

@dataclass
class Intervention:
    name: str
    type: str


@dataclass
class Trial:
    nct_id: str
    title: str
    status: str
    phase: str
    summary: str
    start_date: datetime
    end_date: datetime
    sponsor: str
    url: str
    conditions: list[str]
    interventions: list[Intervention]


@dataclass
class ExtractionInput
    nct_id: str
    title: str
    summary: str 