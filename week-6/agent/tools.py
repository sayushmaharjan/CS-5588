"""
WeatherTwin Agent — Tools
Five callable tools the agent can invoke to gather weather information.
Each tool returns a dict with keys: status, data, source.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fetch_forecast_json(city: str, days: int = 2):
    """Shared helper — calls WeatherAPI forecast endpoint."""
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {"key": WEATHERAPI_KEY, "q": city, "days": days, "aqi": "yes"}
    res = requests.get(url, params=params, timeout=10)
    res.raise_for_status()
    return res.json()


# ---------------------------------------------------------------------------
# Historical data loader (standalone — no Streamlit dependency)
# ---------------------------------------------------------------------------

_weather_df_cache = None

def _load_historical_data() -> pd.DataFrame:
    """Load historical CSV data. Cached after first call."""
    global _weather_df_cache
    if _weather_df_cache is not None:
        return _weather_df_cache

    # Resolve paths relative to this file → project root / data /
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    frames = []
    for fname, city_name in [
        ("los_angeles.csv", "Los Angeles"),
        ("san_diego.csv", "San Diego"),
        ("san_francisco.csv", "San Francisco"),
    ]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["city"] = city_name
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["temperature"] = df.get("TAVG", pd.Series(dtype=float))
    df["wind_speed"] = df.get("AWND", pd.Series(dtype=float))
    df["precipitation"] = df.get("PRCP", pd.Series(dtype=float))
    df["condition"] = df.apply(
        lambda r: "Rainy" if r.get("PRCP", 0) > 0
        else ("Snowy" if r.get("SNOW", 0) > 0 else "Clear"),
        axis=1,
    )
    df = df.dropna(subset=["temperature"])
    _weather_df_cache = df
    return df


# ---------------------------------------------------------------------------
# BERT model loader (standalone — no Streamlit dependency)
# ---------------------------------------------------------------------------

_classifier_cache = None

def _load_bert_classifier():
    """Load the BERT text-classification pipeline. Cached after first call."""
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        _classifier_cache = hf_pipeline(
            "text-classification",
            device=0 if torch.cuda.is_available() else -1,
        )
        return _classifier_cache
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Tool 1 — get_current_weather
# ═══════════════════════════════════════════════════════════════════════════

def get_current_weather(city: str) -> dict:
    """Fetch real-time current weather for *city* via WeatherAPI."""
    try:
        data = _fetch_forecast_json(city, days=1)
        cur = data["current"]
        loc = data["location"]
        result = {
            "city": loc["name"],
            "region": loc["region"],
            "country": loc.get("country", ""),
            "localtime": loc["localtime"],
            "temp_c": cur["temp_c"],
            "temp_f": cur["temp_f"],
            "feels_like_c": cur["feelslike_c"],
            "condition": cur["condition"]["text"],
            "humidity": cur["humidity"],
            "wind_kph": cur["wind_kph"],
            "pressure_mb": cur["pressure_mb"],
            "vis_km": cur["vis_km"],
            "uv": cur["uv"],
            "aqi_us_epa": cur.get("air_quality", {}).get("us-epa-index", "N/A"),
        }
        return {"status": "success", "data": result, "source": "WeatherAPI (live)"}
    except requests.exceptions.HTTPError:
        return {"status": "error", "data": f"Could not find weather for '{city}'.", "source": "WeatherAPI"}
    except Exception as e:
        return {"status": "error", "data": str(e), "source": "WeatherAPI"}


# ═══════════════════════════════════════════════════════════════════════════
# Tool 2 — get_weather_forecast
# ═══════════════════════════════════════════════════════════════════════════

def get_weather_forecast(city: str, hours: int = 12) -> dict:
    """Return hourly forecast for the next *hours* hours."""
    try:
        hours = min(max(hours, 1), 48)
        data = _fetch_forecast_json(city, days=2)
        current_time = datetime.strptime(data["location"]["localtime"], "%Y-%m-%d %H:%M")
        end_time = current_time + timedelta(hours=hours)

        all_hours = []
        for day in data["forecast"]["forecastday"]:
            all_hours.extend(day["hour"])

        forecast = []
        for h in all_hours:
            h_time = datetime.strptime(h["time"], "%Y-%m-%d %H:%M")
            if current_time <= h_time <= end_time:
                forecast.append({
                    "time": h["time"],
                    "temp_c": h["temp_c"],
                    "condition": h["condition"]["text"],
                    "humidity": h["humidity"],
                    "wind_kph": h["wind_kph"],
                    "chance_of_rain": h.get("chance_of_rain", 0),
                })

        result = {
            "city": data["location"]["name"],
            "region": data["location"]["region"],
            "hours_requested": hours,
            "forecast": forecast,
        }
        return {"status": "success", "data": result, "source": "WeatherAPI (forecast)"}
    except Exception as e:
        return {"status": "error", "data": str(e), "source": "WeatherAPI"}


# ═══════════════════════════════════════════════════════════════════════════
# Tool 3 — get_historical_analysis
# ═══════════════════════════════════════════════════════════════════════════

def get_historical_analysis(city: str) -> dict:
    """Statistical summary of historical weather CSV data for *city*."""
    try:
        df = _load_historical_data()
        if df.empty:
            return {"status": "error", "data": "No historical data available.", "source": "CSV"}

        city_data = df[df["city"].str.lower() == city.lower()]
        if city_data.empty:
            available = ", ".join(df["city"].unique())
            return {
                "status": "error",
                "data": f"No data for '{city}'. Available cities: {available}",
                "source": "CSV",
            }

        recent = city_data.tail(60)
        stats = {
            "city": city.title(),
            "records_analysed": len(recent),
            "avg_temp_c": round(recent["temperature"].mean(), 1),
            "min_temp_c": round(recent["temperature"].min(), 1),
            "max_temp_c": round(recent["temperature"].max(), 1),
            "avg_wind_kph": round(recent["wind_speed"].mean(), 1),
            "most_common_condition": (
                recent["condition"].mode()[0]
                if not recent["condition"].mode().empty
                else "N/A"
            ),
            "rainy_days_pct": round(
                (recent["condition"] == "Rainy").mean() * 100, 1
            ),
        }
        return {"status": "success", "data": stats, "source": "Historical CSV"}
    except Exception as e:
        return {"status": "error", "data": str(e), "source": "CSV"}


# ═══════════════════════════════════════════════════════════════════════════
# Tool 4 — predict_weather_bert
# ═══════════════════════════════════════════════════════════════════════════

def predict_weather_bert(city: str) -> dict:
    """BERT-based weather condition prediction from historical patterns."""
    try:
        df = _load_historical_data()
        if df.empty:
            return {"status": "error", "data": "No historical data available.", "source": "BERT"}

        city_data = df[df["city"].str.lower() == city.lower()]
        if city_data.empty:
            return {"status": "error", "data": f"No data for '{city}'.", "source": "BERT"}

        classifier = _load_bert_classifier()
        if classifier is None:
            return {"status": "error", "data": "BERT model failed to load.", "source": "BERT"}

        recent = city_data.tail(20)
        context_parts = []
        for _, row in recent.iterrows():
            context_parts.append(
                f"{row.get('condition', 'Unknown')}, {row['temperature']:.1f}°C, "
                f"{row['wind_speed']:.1f} km/h wind"
            )
        context = "; ".join(context_parts)

        prediction_input = f"Weather forecast for {city}: Based on recent patterns showing {context}"
        result = classifier(prediction_input[:512])

        stats = {
            "city": city.title(),
            "predicted_condition": result[0]["label"],
            "confidence": round(result[0]["score"] * 100, 1),
            "avg_temp_c": round(recent["temperature"].mean(), 1),
            "temp_range": f"{recent['temperature'].min():.1f}–{recent['temperature'].max():.1f}°C",
            "avg_wind_kph": round(recent["wind_speed"].mean(), 1),
            "most_common_condition": (
                recent["condition"].mode()[0]
                if not recent["condition"].mode().empty
                else "N/A"
            ),
            "records_used": len(recent),
        }
        return {"status": "success", "data": stats, "source": "BERT + Historical CSV"}
    except Exception as e:
        return {"status": "error", "data": str(e), "source": "BERT"}


# ═══════════════════════════════════════════════════════════════════════════
# Tool 5 — compare_cities
# ═══════════════════════════════════════════════════════════════════════════

def compare_cities(city1: str, city2: str) -> dict:
    """Compare real-time weather between two cities."""
    w1 = get_current_weather(city1)
    w2 = get_current_weather(city2)

    if w1["status"] == "error":
        return {"status": "error", "data": f"Error for {city1}: {w1['data']}", "source": "WeatherAPI"}
    if w2["status"] == "error":
        return {"status": "error", "data": f"Error for {city2}: {w2['data']}", "source": "WeatherAPI"}

    d1, d2 = w1["data"], w2["data"]

    comparison = {
        "city1": d1,
        "city2": d2,
        "temp_diff_c": round(d1["temp_c"] - d2["temp_c"], 1),
        "humidity_diff": d1["humidity"] - d2["humidity"],
        "wind_diff_kph": round(d1["wind_kph"] - d2["wind_kph"], 1),
        "warmer_city": d1["city"] if d1["temp_c"] >= d2["temp_c"] else d2["city"],
        "calmer_city": d1["city"] if d1["wind_kph"] <= d2["wind_kph"] else d2["city"],
    }
    return {"status": "success", "data": comparison, "source": "WeatherAPI (comparison)"}


# ═══════════════════════════════════════════════════════════════════════════
# Tool dispatcher (used by agent_runner)
# ═══════════════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "get_weather_forecast": get_weather_forecast,
    "get_historical_analysis": get_historical_analysis,
    "predict_weather_bert": predict_weather_bert,
    "compare_cities": compare_cities,
}
