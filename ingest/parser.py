from datetime import datetime

from models import Trial, Intervention


def parse_date(date_str: str) -> datetime:
    """
    Parse a date string in the format 'YYYY-MM-DD' or 'YYYY-MM' and return a datetime object.
    If the date string is in 'YYYY-MM' format, the day will default to 1.

    Parameters:
    date_str (str): The date string to parse.

    Returns:
    datetime: A datetime object representing the parsed date, or None if parsing fails.

    """

    if not date_str:
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            dt = datetime.strptime(date_str, fmt)
            # If month-only format, default day=1
            if fmt == "%Y-%m":
                dt = dt.replace(day=1)
            return dt
        except ValueError:
            pass
    return None  # Could not parse


def parse_trial(study: dict) -> Trial:
    """
    Parse a clinical trial study dictionary and return a Trial object.

    Parameters:
    study (dict): A dictionary containing clinical trial study data.

    Returns:
    Trial: A Trial object representing the parsed clinical trial study.
    """

    identification = study.get("identificationModule", {})
    status = study.get("statusModule", {})
    description = study.get("descriptionModule", {})
    sponsor = study.get("sponsorCollaboratorsModule", {})
    design = study.get("designModule", {})
    interventions_module = study.get("armsInterventionsModule", {})
    conditions_module = study.get("conditionsModule", {})

    phases = design.get("phases", [])

    interventions = [
        Intervention(
            name=item.get("name", "").strip(),
            type=item.get("type", "").strip(),
        )
        for item in interventions_module.get("interventions", [])
        if item.get("name")
    ]

    conditions = [
        c.strip() for c in conditions_module.get("conditions", []) if c.strip()
    ]

    nct_id = identification["nctId"]

    return Trial(
        nct_id=nct_id,
        title=identification.get("briefTitle", "").strip(),
        status=status.get("overallStatus", "").strip(),
        phase=phases[0].strip() if phases else "",
        summary=description.get("briefSummary", "").strip(),
        start_date=parse_date(
            status.get("startDateStruct", {}).get("date")
            or status.get("studyFirstSubmitDate")
        ),
        end_date=parse_date(status.get("completionDateStruct", {}).get("date")),
        sponsor=sponsor.get("leadSponsor", {}).get("name", ""),
        url=f"https://clinicaltrials.gov/study/{nct_id}",
        conditions=conditions,
        interventions=interventions,
    )
