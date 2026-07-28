from dataclasses import dataclass, field
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
class ExtractionInput:
    nct_id: str
    title: str
    summary: str 

@dataclass
class CanonicalizationInput:
    extracted_condition_id: int
    name: str

@dataclass
class CanonicalizationOutput:
    extracted_condition_id: int
    common_name: str
    technical_name: str
    abbreviations: list[str] = field(default_factory=list)
