# WeatherTwin Dashboard

An interactive Streamlit dashboard designed to visualize historical weather patterns directly from Snowflake. This project leverages **Snowpark** to perform in-database data processing, enabling high performance even with large datasets.

## Architecture

The application follows a **data-app decoupled architecture**:

* **Data Layer:** Snowflake stores raw and aggregated weather data. Snowpark handles transformations.
* **Application Layer:** Streamlit dashboards (local or cloud) query and visualize the data.
* **UI Layer:** Interactive charts, filters, and metrics delivered via Streamlit.
<img width="1536" height="1024" alt="Architecture" src="https://github.com/user-attachments/assets/6a2afffe-ed2f-4b79-882f-c5ffdea21e36" />

---

## Implemented Extensions

### Automated Feature-Engineering Pipeline (Data / Feature Extension)

Implemented entirely in Snowflake views:

* **WEATHER_ENRICHED**

  * Cleans `DATE` into `OBS_DATE` using robust parsing (`TRY_TO_DATE` with fallback).
  * Derives `CITY` from `NAME`.
  * Derives categorical `CONDITION` from `PRCP` and `SNOW`.

* **CITY_STATS**

  * Computes city-level KPIs: `AVG_TEMP`, `MIN_TEMP`, `MAX_TEMP`, `AVG_WIND`, `RAINY_DAYS`.

* **V_WEATHER_WITH_CITY_STATS**

  * Joins row-level records with city-level features for downstream analytics.

### Monitoring Dashboard for Pipeline Performance (System Extension)

* **Logging Utilities (`python/logging_utils.py` + `python/snowflake_client.py`)**

  * Logs every Snowflake query with `{timestamp, query_name, latency_sec, rows}` into `pipeline_logs.csv`.

* **Streamlit Panel**

  * Shows recent logs, total query count, and latency time-series plots for monitoring pipeline performance.

### Interactive Analytics Dashboard Component (System Extension)

* **Dataset Info Section (`app_bert.py`)**

  * Displays Snowflake-backed metrics: record count, number of cities, and load latency.
  * Shows city-level KPIs from `CITY_STATS`.
  * Allows user to select a city and view last 30 days of weather data from `RECENT_CITY_WEATHER`.

---

## 👥 Contributors & Workflows

This project supports **two development workflows**, allowing flexibility for local and cloud-based development.

**1. Harsha Sri Neeriganti - Cloud-Native Developer (Direct in Snowflake)**

* **Environment:** Streamlit app inside Snowsight.
* **Authentication:** Automatic using `get_active_session()`.
* **Advantage:** No setup; live connection to Snowflake.

**Responsibilities & Contributions:**

* Uploaded raw weather data directly to Snowflake via CSV ingestion.
* Developed the multi-city interactive dashboard for selection and comparison.
* Implemented real-time metrics calculations (Max/Min temperatures, Precipitation) using Snowpark in-database computations.
* Optimized queries for performance to handle large datasets efficiently.
* Integrated time-series visualizations and dynamic filters using Streamlit and Plotly.

**2. Sayush Maharjan - Local Developer (Remote Connection to Snowflake)**

* **Environment:** Local IDE (VS Code).
* **Authentication:** MFA-authenticated Snowflake connection.
* **Advantage:** Full control, faster iteration, and local debugging tools.

**Responsibilities & Contributions:**

* Connected to Snowflake to fetch and transform data for local testing.
* Developed local Streamlit dashboard for offline iteration.
* Implemented hybrid code support for both local and cloud usage.
* Added interactive charts and dynamic filters for quick exploration.
* Assisted with data validation and cleaning for consistent results across environments.

Together, both workflows enable seamless development: cloud offers zero-setup deployment, local allows flexible testing.

---

## Setup & Installation

### For Local Contributors

1. **Install dependencies:**

```bash
pip install snowflake-snowpark-python streamlit pandas plotly
```

2. **Configure credentials:**

```python
account="SFEDU02-DCB73175"
user="GIRAFFE"
authenticator="snowflake"
password=<from env or constant>
passcode=<MFA TOTP, prompted in terminal>
role="TRAINING_ROLE"
warehouse="WEATHER_TWIN_WH"
database="WEATHER_TWIN_DB"
schema="PUBLIC"
```

### For Cloud Contributors

1. Open **Snowsight** → **Streamlit**.
2. Create a new app.
3. Copy `app_snowflake.py` contents (or your main `streamlit_app.py`) into the editor.
4. Add required packages via the **Packages** dropdown.

---

## 🛠️ Hybrid Code Support

```python
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
try:
     session = get_active_session()
except Exception:
      st.error("No active Snowflake session found. Run inside Snowflake Projects.")
      st.stop()

session = get_session()
```

---

## 📂 Project Structure

```
weather-dashboard/
│
├─ load_weather_csv.py
├─ snowflake_app.py            # Streamlit app for cloud deployment in Snowflake
├─ app_bert.py                 # Streamlit app for local development
├─ snowflake_client.py         # Shared core logic (optional if using hybrid code)
│
├─ requirements.txt            # Local dependencies
├─ environment.yml             # Cloud environment dependencies

```

## 📂 Project Structure for local implementation

```
week-5/
  app/
    app_bert.py                # Streamlit application
  python/
    __init__.py
    snowflake_client.py        # Snowflake connector + run_query
    logging_utils.py           # pipeline_logs.csv writer
  sql/
    01_views.sql               # WEATHER_ENRICHED, CITY_STATS, RECENT_CITY_WEATHER, V_WEATHER_WITH_CITY_STATS
  diagrams/
    architecture_week5.png     # Architecture diagram
  pipeline_logs.csv            # Generated at runtime (committed with sample logs)
  README.md
  CONTRIBUTIONS.md
  .env.example                 # Document required env vars (no secrets)
```


**Notes:**

* `app_snowflake.py` → For **cloud deployment** inside Snowsight.
* `app_local.py` → For **local Streamlit testing** and debugging.
* `ingestion/` → Handles initial data ingestion from CSV or external sources into Snowflake.

---

## Features

* **Dynamic Filtering:** Searchable dropdown populated with `NAME` values from `WEATHER_FULL`.
* **Live Metrics:** Max/Min temperatures and Precipitation calculated in real time.
* **Historical Trends:** Time-series charts using Streamlit and Plotly.
* **Environment-Agnostic:** Single codebase supports both local and cloud workflows.

---

## Demo & Deployment

* **Dashboard demo video:** (https://drive.google.com/file/d/1VK4R-Iro2UsWvcMJRorbFTdT3lqSHYHk/view?usp=drive_link)
* **Cloud Snowflake dashboard (Snowsight):** [Open here](https://app.snowflake.com/sfedu02/dcb73175/#/streamlit-apps/WEATHER_TWIN_DB.PUBLIC.LMTE323F0FBAR_NK)


