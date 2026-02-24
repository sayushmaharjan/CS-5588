import csv
import os
from datetime import datetime, timezone

LOG_FILE = "pipeline_logs.csv"

def log_pipeline_event(query_name: str, latency_sec: float, rows_returned: int):
    """
    Append a row to pipeline_logs.csv.

    Columns:
      timestamp (ISO8601, UTC)
      query      - logical query name
      latency_sec
      rows       - number of rows returned
    """
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query", "latency_sec", "rows"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            query_name,
            round(latency_sec, 4),
            rows_returned,
        ])