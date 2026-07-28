import os
import json
from tqdm import tqdm

from db.connection import get_db_connection
from ingest.schema import create_tables
from ingest.parser import parse_trial
from ingest.persistence import insert_trial, insert_conditions, insert_interventions

DATA_DIR = "data/ClinTrialFiles"


def main():

    conn = get_db_connection()
    cur = conn.cursor()

    try:

        create_tables(cur)

        for json_file in tqdm(os.listdir(DATA_DIR), desc="Processing JSON files"):

            if not json_file.endswith(".json"):
                continue
            
            file_path = os.path.join(DATA_DIR, json_file)
        
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            study = data.get("protocolSection", {})
            if not study:
                print(f"Skipping {json_file}: 'protocolSection' not found.")
                continue
            
            trial = parse_trial(study)

            insert_trial(cur, trial)
            insert_conditions(cur, trial)
            insert_interventions(cur, trial)

        conn.commit()

    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()

