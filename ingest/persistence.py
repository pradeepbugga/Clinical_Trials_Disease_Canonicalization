from models import Trial

def insert_trial(cur, trial: Trial):
    """
    Insert a Trial object into the database.

    Parameters:
    cur : The database cursor.
    trial (Trial): The Trial object to be inserted.
    """

    cur.execute(
        """
        INSERT INTO ClinicalTrials (nct_id, title, status, phase, summary, start_date, end_date, location, sponsor, url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            trial.nct_id,
            trial.title,
            trial.status,
            trial.phase,
            trial.summary,
            trial.start_date.isoformat() if trial.start_date else None,
            trial.end_date.isoformat() if trial.end_date else None,
            None,  # Placeholder for location, as it's not provided in the Trial object
            trial.sponsor,
            trial.url,
        ),
    )


def insert_conditions(cur, trial: Trial):

    for condition in trial.conditions:

        cur.execute(
            """
        INSERT INTO Conditions (name)
        VALUES (%s)
        ON CONFLICT DO NOTHING        
        """,
            (condition,),
        )

        cur.execute(
            """
        SELECT condition_id
        FROM Conditions
        WHERE name = %s
        """,
            (condition,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Condition '{condition}' not found in the database after insertion.")
        condition_id = row[0]

        cur.execute(
            """
        INSERT INTO TrialConditions
        (nct_id, condition_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """,
            (trial.nct_id, condition_id),
        )


def insert_interventions(cur, trial: Trial):

    for intervention in trial.interventions:

        cur.execute(
            """
            INSERT INTO Interventions (name, type)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (intervention.name, intervention.type),
        )

        cur.execute(
            """
            SELECT intervention_id
            FROM Interventions
            WHERE name = %s
            """,
            (intervention.name,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Intervention '{intervention.name}' not found in the database after insertion.")
        intervention_id = row[0]

        cur.execute(
            """
            INSERT INTO TrialInterventions
            (nct_id, intervention_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (trial.nct_id, intervention_id),
        )
