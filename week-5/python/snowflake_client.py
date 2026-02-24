# python/snowflake_client.py
import time
import getpass
import pandas as pd
import snowflake.connector
# from python.logging_utils import log_pipeline_event  # adjust if path differs
from python.logging_utils import log_pipeline_event  # adjust if path differs

# Cached connection to avoid re‑entering TOTP on every query
_sf_conn = None

def get_sf_connection():
    """
    Returns a cached Snowflake connection.
    Uses authenticator='snowflake' with MFA TOTP.
    Prompts for TOTP once per process.
    """
    global _sf_conn

    if _sf_conn is not None and not _sf_conn.is_closed():
        return _sf_conn

    # Hard-code or load from .env as you prefer
    ACCOUNT   = "SFEDU02-DCB73175"      # add region suffix if your URL has it
    USER      = "GIRAFFE"
    PASSWORD  = "Sn0wFl@ke@UMKC"        # or read from os.getenv(...)
    ROLE      = "TRAINING_ROLE"
    WAREHOUSE = "WEATHER_TWIN_WH"
    DATABASE  = "WEATHER_TWIN_DB"
    SCHEMA    = "PUBLIC"

    totp = getpass.getpass("Enter current Snowflake MFA TOTP code: ")

    _sf_conn = snowflake.connector.connect(
        account=ACCOUNT,
        user=USER,
        password=PASSWORD,
        passcode=totp,               # TOTP separate from password
        authenticator="snowflake",
        role=ROLE,
        warehouse=WAREHOUSE,
        database=DATABASE,
        schema=SCHEMA,
    )
    return _sf_conn

def run_query(sql: str, query_name: str = "unnamed_query"):
    """
    Run SQL, return (DataFrame, latency_sec), and log to pipeline_logs.csv
    """
    conn = get_sf_connection()
    start = time.perf_counter()
    df = pd.read_sql(sql, conn)
    latency = time.perf_counter() - start
    rows = len(df)
    log_pipeline_event(query_name, latency, rows)
    return df, latency