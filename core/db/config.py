import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


POSTGRES_CONFIG = {
    "host": require_env("PGHOST"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": require_env("PGDATABASE"),
    "user": require_env("PGUSER"),
    "password": require_env("PGPASSWORD"),
}
