# Clinical Trial Disease Canonicalization

An end-to-end biomedical NLP pipeline for extracting disease names from ClinicalTrials.gov records and mapping them to standardized condition names.

The project includes:

- ClinicalTrials.gov JSON ingestion
- LLM-based disease extraction
- Disease-name canonicalization
- GPT-4.1-nano fine-tuning
- Batch API processing
- PostgreSQL persistence
- Model evaluation
- Automated tests and GitHub Actions

## Table of Contents
- [Motivation](#motivation)
- [Pipeline](#pipeline)
- [Repository Structure](repository-structure)
- [Core Data Models](#core-data-models)
- [Disease Extraction](disease-extraction)
- [Disease Canonicalization](disease-canonicalization)
- [Fine-Tuning](fine-tuning)
- [Evaluation](evaluation)
- [Installation](installation)
- [Configuration](configuration)
- [Running Tests](running-tests)
- [CI](ci)
- [Limitations](limitations)
- [Future Work](future-work)
- [License](license)


## Motivation

Clinical trial records describe diseases using inconsistent terminology, including:

- common names
- technical names
- abbreviations
- spelling variants
- disease subtypes
- broad symptom descriptions

For example, the same condition may appear as:

```text
COPD
Chronic obstructive pulmonary disease
Chronic obstructive lung disease
```

This project converts those variants into a consistent structured representation:
```
{
  "common_name": "Chronic obstructive pulmonary disease",
  "technical_name": null,
  "abbreviations": ["COPD"]
}
```
The resulting data can support clinical-trial search, aggregation, analytics, and disease-level reporting.

## Pipeline
```
ClinicalTrials.gov JSON
        |
        v
Trial ingestion and parsing
        |
        v
Disease extraction
        |
        v
Condition deduplication
        |
        v
Disease canonicalization
        |
        v
PostgreSQL persistence
        |
        v
Fine-tuning and evaluation
```

## Repository Structure
```
├── canonicalization/
│   ├── batch.py
│   ├── canonicalize.py
│   ├── parser.py
│   └── populate.py
│
├── extraction/
│   ├── batch.py
│   ├── extract.py
│   ├── parser.py
│   └── populate.py
│
├── finetuning/
│   ├── evaluate.py
│   ├── prepare.py
│   └── prompts.py
│
├── ingest/
│   ├── ingest.py
│   ├── parser.py
│   ├── persistence.py
│   └── schema.py
│
├── models/
│   └── models.py
│
├── tests/
│   ├── canonicalization/
│   ├── extraction/
│   ├── finetuning/
│   └── ingest/
│
├── data/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── requirements.txt
└── README.md
```

## Core Data Models
A parsed clinical trial contains fields such as:
```Trial(
    nct_id="NCT00380471",
    title="...",
    status="COMPLETED",
    phase="PHASE2",
    conditions=["Vitiligo"],
    interventions=[...],
)
```
A canonicalized disease record contains:
```CanonicalizationOutput(
    extracted_condition_id=123,
    common_name="High blood potassium",
    technical_name="Hyperkalemia",
    abbreviations=[],
)
```
## Disease Extraction
The extraction stage identifies diseases and medical conditions from clinical-trial text.

Example input:
```
A study evaluating a treatment for patients with relapsing multiple sclerosis.
```
Example output:
```{
  "conditions": [
    "Multiple sclerosis"
  ]
}
```
Extraction requests can be generated and submitted through the OpenAI Batch API. Batch responses are parsed from JSONL and associated with the original trial using the custom_id field.

## Disease Canonicalization
The canonicalization stage converts extracted disease strings into standardized names.

Example input:
```
High blood potassium
```
Example output:
```
{
  "common_name": "High blood potassium",
  "technical_name": "Hyperkalemia",
  "abbreviations": []
}
```
Another example:
```
COPD
{
  "common_name": "Chronic obstructive pulmonary disease",
  "technical_name": null,
  "abbreviations": ["COPD"]
}
```
The structured output separates:
- ```common_name```
- ```technical_name```
- ```abbreviations```
When the common and technical names are identical, the technical name may be stored as null to avoid redundant labels.

## Fine-Tuning
The repository includes utilities for preparing supervised fine-tuning data in OpenAI chat JSONL format.

Each training example contains:
```
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "High blood potassium"
    },
    {
      "role": "assistant",
      "content": "{\"common_name\":\"High blood potassium\",\"technical_name\":\"Hyperkalemia\",\"abbreviations\":[]}"
    }
  ]
}
```
The preparation pipeline:

1. Loads canonical disease mappings
2. Normalizes redundant technical names
3. Shuffles examples using a fixed random seed
4. Splits examples into training and test sets
5. Writes valid JSONL files

The deployed fine-tuned model was based on GPT-4.1-nano.

## Evaluation

The evaluation pipeline compares model predictions against held-out examples.

### Metrics
The system reports:

- Common-name exact-match accuracy
- Technical-name exact-match accuracy
- Abbreviation precision
- Abbreviation recall
- Abbreviation F1 score

Name matching is normalized for case and surrounding whitespace. Abbreviation comparison is set-based and order-independent.

### Results
| Model                     | Common-name accuracy | Technical-name accuracy | Abbreviation F1 |
|---------------------------|---------------------:|------------------------:|----------------:|
| GPT-4.1-nano baseline     | 54.5%                | 32.2%                   | 32.4%           |
| Fine-tuned GPT-4.1-nano   | 77.1%                | 77.4%                   | 73.8%           |

The fine-tuned model substantially improved performance across all three output fields.

Current work focuses on expanding the training corpus and improving generalization to less common disease names and terminology variants.

## Installation
Clone the repository:
```
git clone <repository-url>
cd Clinical_Trials_Disease_Canonicalization

