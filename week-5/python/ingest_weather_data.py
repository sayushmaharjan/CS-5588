# python/ingest_weather_data.py
import os
import time
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

load_dotenv()

def get_sf_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "CS5588_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "CS5588_DB"),
        schema="RAW",
    )

def build_weather_dataframe():
    base_path = "data"

    files_and_cities = [
        ("los_angeles.csv", "Los Angeles"),
        ("san_diego.csv", "San Diego"),
        ("san_francisco.csv", "San Francisco"),
    ]

    dfs = []
    for fname, city in files_and_cities:
        fpath = os.path.join(base_path, fname)
        df = pd.read_csv(fpath)
        df["city"] = city
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Derived columns as in your Week‑4 app
    df["temperature"] = df["TAVG"]
    df["wind_speed"] = df["AWND"]
    df["precipitation"] = df["PRCP"]

    df["condition"] = df.apply(
        lambda row: "Rainy" if row.get("PRCP", 0) > 0 else
                    "Snowy" if row.get("SNOW", 0) > 0 else
                    "Clear",
        axis=1
    )

    # Rename DATE -> OBS_DATE to avoid reserved word issues
    if "DATE" in df.columns:
        df.rename(columns={"DATE": "OBS_DATE"}, inplace=True)

    # Drop rows with missing temperature
    df = df.dropna(subset=["temperature"])

    # ~50% sample for assignment requirement
    df_sample = df.sample(frac=0.5, random_state=42)
    return df_sample

def ingest_weather():
    df = build_weather_dataframe()
    print(f"Sample size: {len(df)} rows")

    conn = get_sf_connection()
    try:
        conn.cursor().execute("USE SCHEMA RAW")

        # Write to RAW.WEATHER_HIST
        success, nchunks, nrows, _ = write_pandas(
            conn,
            df,
            "WEATHER_HIST",
            schema="RAW",
            auto_create_table=True,
            overwrite=True,
        )
        print(f"Success={success}, rows_written={nrows}, chunks={nchunks}")

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM RAW.WEATHER_HIST")
        count = cur.fetchone()[0]
        print(f"RAW.WEATHER_HIST now has {count} rows.")
    finally:
        conn.close()

if __name__ == "__main__":
    start = time.time()
    ingest_weather()
    print(f"Ingestion done in {time.time() - start:.2f}s")