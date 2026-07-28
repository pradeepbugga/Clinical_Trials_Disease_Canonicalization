def create_tables(cur) -> None:
    """
    Create the necessary tables for the clinical trials database if they do not already exist.

    Parameters:
    cur: Cursor object for the database connection.
    """

    cur.execute("""
    CREATE TABLE IF NOT EXISTS Conditions (
                condition_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name TEXT UNIQUE
                )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ClinicalTrials (nct_id TEXT PRIMARY KEY, 
                title TEXT NOT NULL, 
                status TEXT,
                phase TEXT,
                summary TEXT,
                start_date DATE,
                end_date DATE,
                location TEXT,
                sponsor TEXT,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP     
                )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS TrialConditions (nct_id TEXT NOT NULL,  
                condition_id BIGINT NOT NULL,
                PRIMARY KEY (nct_id, condition_id),
                FOREIGN KEY(nct_id) REFERENCES ClinicalTrials(nct_id),
                FOREIGN KEY(condition_id) REFERENCES Conditions(condition_id)
                
                )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Interventions (
                intervention_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                type TEXT
                )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS TrialInterventions (nct_id TEXT NOT NULL,  
                intervention_id BIGINT NOT NULL,
                PRIMARY KEY (nct_id, intervention_id),
                FOREIGN KEY(nct_id) REFERENCES ClinicalTrials(nct_id),
                FOREIGN KEY(intervention_id) REFERENCES Interventions(intervention_id)
                )""")


def create_extraction_tables(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ExtractedConditions (
            extracted_condition_id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS TrialExtractedConditions (
            nct_id TEXT NOT NULL,
            extracted_condition_id INTEGER NOT NULL,
            PRIMARY KEY (nct_id, extracted_condition_id),
            FOREIGN KEY (nct_id) REFERENCES ClinicalTrials(nct_id),
            FOREIGN KEY (extracted_condition_id)
                REFERENCES ExtractedConditions(extracted_condition_id)
        )
    """)