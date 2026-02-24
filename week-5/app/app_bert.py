# app.py

import os
import sys
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
import torch

from datetime import datetime, timedelta
from huggingface_hub import login

import streamlit as st
import snowflake.connector



# Ensure project root is on sys.path so "python" package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python.snowflake_client import run_query

# login(token=os.getenv("HF_TOKEN"))


# =========================
# LOAD ENV
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY") 

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"
LOG_FILE = "chat_log.txt"

# =========================
# BERT MODEL SETUP
# =========================
@st.cache_resource
def load_weather_model():
    """
    Load Fine-tuned BERT for weather classification
    Model: https://huggingface.co/oliverguhr/weather-classification
    """
    try:
        # Load model directly

        # classifier = pipeline("zero-shot-classification",
        # model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

        classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

        # tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
        # model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

        # model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
       
        token = os.getenv("HF_TOKEN")

        # model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create pipeline for easier use
        classifier = pipeline(
            "text-classification",
            # model=model,
            # tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        st.success("✅ Weather BERT model loaded successfully!")
        return classifier#, tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading BERT model: {e}")
        return None#, None, None

# =========================
# WEATHER DATASET LOADING
# =========================
@st.cache_data
def load_weather_dataset():
    sql = """
        SELECT
            CITY,
            OBS_DATE AS obs_date,
            TAVG     AS temperature,
            AWND     AS wind_speed,
            PRCP     AS precipitation,
            CONDITION
        FROM WEATHER_ENRICHED
    """
    df, latency = run_query(sql, query_name="load_weather_dataset")
    df.columns = [c.lower() for c in df.columns]
    return df, latency

# def load_weather_dataset():
#     """
#     Load historical weather dataset
#     Dataset: https://huggingface.co/datasets/mongodb/weather
#     Fallback to sample data if not available
#     """
#     try:
#         from datasets import load_dataset
#         # df1 = pd.read_csv("/Users/sayush/Documents/cs5588/CS-5588/week-5/data/los_angeles.csv")
#         # df1["city"] = "Los Angeles"

#         # df2 = pd.read_csv("/Users/sayush/Documents/cs5588/CS-5588/week-5/data/san_diego.csv")
#         # df2["city"] = "San Diego"

#         # df3 = pd.read_csv("/Users/sayush/Documents/cs5588/CS-5588/week-5/data/san_francisco.csv")
#         # df3["city"] = "San Francisco"

#         # df = pd.concat([df1, df2, df3], ignore_index=True)

#         # in app.py
        


#         # df = load_dataset("sayush-m/us-weather-data", token=True)
#         # ds = load_dataset("sayush-m/us-weather-data")

#         # df = pd.DataFrame(ds["train"])



#         # st.write("Columns in dataset:", df.columns)


#         # Normalize column names
#         # df.columns = df.columns.str.upper()

#         # Create expected columns
#         df["temperature"] = df["TAVG"]
#         df["wind_speed"] = df["AWND"]
#         df["precipitation"] = df["PRCP"]

#         # Since we don't have condition, create simple rule-based condition
#         df["condition"] = df.apply(
#             lambda row: "Rainy" if row["PRCP"] > 0 else
#                         "Snowy" if row["SNOW"] > 0 else
#                         "Clear",
#             axis=1
#         )

#         # Keep city lowercase-safe
#         # df["city"] = df["city"].str.strip()

#         # Drop rows with missing temperature
#         df = df.dropna(subset=["temperature"])


#         # Try loading from HuggingFace
#         # dataset = load_dataset("mongodb/weather", split="train[:1000]")  # Limit to 1000 rows
#         # df = pd.DataFrame(dataset)
#         # st.success(f"✅ Loaded {len(df)} weather records from HuggingFace")
#         return df
#     except Exception as e:
#         st.warning(f"⚠️ Could not load HuggingFace dataset: {e}")
        
#         # Fallback: Create comprehensive sample weather data
#         import numpy as np
        
#         cities = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Berlin', 
#                   'Rome', 'Madrid', 'Beijing', 'Moscow', 'Dubai', 'Singapore']
        
#         conditions = ['Sunny', 'Partly Cloudy', 'Cloudy', 'Rainy', 'Stormy', 
#                      'Snowy', 'Foggy', 'Windy']
        
#         # Generate realistic weather data
#         n_records = 1000
        
#         sample_data = {
#             'city': np.random.choice(cities, n_records),
#             'temperature': np.random.normal(20, 10, n_records),  # Mean 20°C, std 10
#             'humidity': np.random.uniform(30, 90, n_records),
#             'wind_speed': np.random.uniform(5, 40, n_records),
#             'pressure': np.random.normal(1013, 20, n_records),
#             'condition': np.random.choice(conditions, n_records),
#             'date': pd.date_range('2023-01-01', periods=n_records, freq='6H')
#         }
        
#         df = pd.DataFrame(sample_data)
#         st.info(f"ℹ️ Using sample dataset with {len(df)} records")
#         return df


@st.cache_data(show_spinner=False)
def load_city_stats():
    sql = "SELECT * FROM CITY_STATS"
    return run_query(sql, query_name="load_city_stats")


@st.cache_data(show_spinner=False)
def load_joined_sample(limit: int = 100):
    sql = f"""
        SELECT *
        FROM V_WEATHER_WITH_CITY_STATS
        LIMIT {limit}
    """
    return run_query(sql, query_name="load_joined_sample")


@st.cache_data(show_spinner=False)
def load_recent_city_weather(city: str, days: int = 30):
    city_escaped = city.replace("'", "''")
    sql = f"""
        SELECT *
        FROM RECENT_CITY_WEATHER
        WHERE CITY = '{city_escaped}'
        ORDER BY OBS_DATE DESC
    """
    return run_query(sql, query_name=f"recent_weather_{city_escaped}")



# =========================
# WEATHER KNOWLEDGE BASE
# =========================
def create_weather_context(df, city: str, limit: int = 20):
    """Create context from historical weather data for a specific city"""
    city_data = df[df['city'].str.lower() == city.lower()]
    
    if city_data.empty:
        return None, None
    
    # Get recent records
    recent_data = city_data.tail(limit)
    
    # Create context string
    context_parts = []
    for _, row in recent_data.iterrows():
        context_parts.append(
            f"{row.get('condition', 'Unknown')}, {row.get('temperature', 0):.1f}°C, "
            f"{row.get('humidity', 0):.0f}% humidity, {row.get('wind_speed', 0):.1f} km/h wind"
        )
    
    context = "; ".join(context_parts)
    
    # Calculate statistics
    stats = {
        'avg_temp': recent_data['temperature'].mean(),
        # 'avg_humidity': recent_data['humidity'].mean(),
        'avg_wind': recent_data['wind_speed'].mean(),
        'common_condition': recent_data['condition'].mode()[0] if not recent_data['condition'].mode().empty else 'N/A',
        'max_temp': recent_data['temperature'].max(),
        'min_temp': recent_data['temperature'].min(),
        'records_count': len(recent_data)
    }
    
    return context, stats

# =========================
# BERT-BASED WEATHER PREDICTION
# =========================
def predict_weather_with_bert(query: str, weather_df: pd.DataFrame, classifier):
    """Use BERT to classify and predict weather based on historical data"""
    try:
        # Extract city from query
        city = extract_city_from_query(query)
        
        if not city:
            return None, "Could not identify city in query. Please mention a city name."
        
        # Get historical context
        context, stats = create_weather_context(weather_df, city)
        
        if not context or not stats:
            return None, f"No historical data found for {city}. Try using Live Weather mode."
        
        # Prepare input for BERT classifier
        # The model expects weather description text
        prediction_input = f"Weather forecast for {city}: Based on recent patterns showing {context}"
        
        # Get classification
        result = classifier(prediction_input[:512])  # BERT max length
        
        predicted_class = result[0]['label']
        confidence = result[0]['score']
        
        # Generate comprehensive forecast
        forecast = f"""
🔮 **Weather Forecast for {city}** (AI-Powered Prediction)

🤖 **BERT Model Prediction:** 
- **Condition:** {predicted_class}
- **Confidence:** {confidence * 100:.1f}%

📊 **Statistical Analysis** (Based on {stats['records_count']} recent records):
- **Average Temperature:** {stats['avg_temp']:.1f}°C
- **Temperature Range:** {stats['min_temp']:.1f}°C to {stats['max_temp']:.1f}°C

- **Average Wind Speed:** {stats['avg_wind']:.1f} km/h
- **Most Common Condition:** {stats['common_condition']}

🎯 **Forecast Summary:**
Expect weather conditions similar to recent patterns. The AI model predicts **{predicted_class}** 
with {confidence * 100:.1f}% confidence based on historical data analysis.

⚠️ *This prediction is based on historical patterns and AI classification.*
"""
# - **Average Humidity:** {stats['avg_humidity']:.1f}%
    
        return forecast, None
        
    except Exception as e:
        return None, f"BERT prediction error: {str(e)}"

# =========================
# EXTRACT CITY FROM QUERY
# =========================
def extract_city_from_query(query: str) -> str:
    """Extract city name from user query using simple pattern matching"""
    # Common cities (expand this list)
    common_cities = [
        'new york', 'london', 'tokyo', 'paris', 'sydney', 'berlin',
        'rome', 'madrid', 'beijing', 'moscow', 'dubai', 'singapore',
        'los angeles', 'chicago', 'toronto', 'mumbai', 'delhi',
        'bangkok', 'hong kong', 'seoul', 'amsterdam', 'barcelona',
        'san diego', 'san francisco'
    ]
    
    query_lower = query.lower()
    
    # Check for direct city mentions
    for city in common_cities:
        if city in query_lower:
            return city.title()
    
    # Try to extract using common patterns
    # patterns = [" in ", " for ", " at "]
    # for pattern in patterns:
    #     if pattern in query_lower:
    #         parts = query_lower.split(pattern)
    #         if len(parts) > 1:
    #             # Get the word after the pattern
    #             potential_city = parts[1].split()[0].strip('?,.')
    #             return potential_city.title()

    match = re.search(r"(?:in|at|for)\s+([a-z\s]+)", query_lower)
    if match:
        city_candidate = match.group(1).strip()
        city_candidate = re.sub(r"\b(summer|winter|spring|fall|today|tomorrow)\b", "", city_candidate)
        return city_candidate.strip().title()
    
    return None

# =========================
# LOGGING FUNCTION
# =========================
def log_query(user_input: str, bot_response: str, weather_data: str = None, source: str = "API"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"Timestamp   : {timestamp}\n")
        f.write(f"Source      : {source}\n")
        f.write(f"User Query  : {user_input}\n")
        if weather_data:
            f.write(f"Weather Data: {weather_data}\n")
        f.write(f"Bot Response: {bot_response}\n")
        f.write("="*60 + "\n\n")

# =========================
# WEATHER FUNCTIONS
# =========================
def fetch_weather(city: str):
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {"key": WEATHERAPI_KEY, "q": city, "days": 2, "aqi": "yes"}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data
    except Exception as e:
        return {"error": str(e)}

def parse_current(data):
    cur = data["current"]
    loc = data["location"]
    return {
        "city": loc["name"],
        "region": loc["region"],
        "localtime": loc["localtime"],
        "temp_c": cur["temp_c"],
        "condition": cur["condition"]["text"],
        "humidity": cur["humidity"],
        "wind_kph": cur["wind_kph"],
        "pressure_mb": cur["pressure_mb"],
        "vis_km": cur["vis_km"],
        "uv": cur["uv"],
        "aqi": cur.get("air_quality", {}).get("us-epa-index", "N/A"),
        "sunrise": data["forecast"]["forecastday"][0]["astro"]["sunrise"],
        "sunset": data["forecast"]["forecastday"][0]["astro"]["sunset"]
    }

def get_24h_data(data):
    # hours = data["forecast"]["forecastday"][0]["hour"]
    # times = [h["time"].split(" ")[1] for h in hours]
    # temps = [h["temp_c"] for h in hours]
    # humidity = [h["humidity"] for h in hours]


    # Current local time of the city
    current_time_str = data["location"]["localtime"]  # e.g., '2026-02-16 14:00'
    current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M")

    # Forecast hours
    hours = data["forecast"]["forecastday"][0]["hour"]

    # Keep only next 12 hours
    times_12h = []
    temps_12h = []
    humidity_12h = []
    conditions_12h = []
    icons_12h = []

    for h in hours:
        h_time = datetime.strptime(h["time"], "%Y-%m-%d %H:%M")
        if current_time <= h_time <= current_time + timedelta(hours=12):
            times_12h.append(h_time.strftime("%H:%M"))
            temps_12h.append(h["temp_c"])
            # humidity_12h.append(h["humidity"])
            # conditions_12h.append(h["condition"]["text"])
            cond = h["condition"]["text"]
            conditions_12h.append(cond)
            icons_12h.append(weather_to_icon(cond))

    return times_12h, temps_12h, humidity_12h, conditions_12h

# =========================
# AI AGENT FUNCTIONS
# =========================
SYSTEM_PROMPT = """You are a helpful AI agent that can use tools to find weather information.

IMPORTANT: You must ALWAYS respond with a single line of valid JSON. No markdown, no extra text.

Available actions:
1. get_weather - Use this to fetch current weather for a city
2. user_answer - Use this to give the final answer to the user

Response format (strict JSON only):
{"thought": "your reasoning", "action": "get_weather", "action_input": "city name"}
or
{"thought": "your reasoning", "action": "user_answer", "action_input": "your final answer"}
"""

def get_weather(city: str) -> str:
    url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": WEATHERAPI_KEY,
        "q": city,
        "aqi": "no"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        description = data["current"]["condition"]["text"]
        temperature = data["current"]["temp_c"]
        humidity = data["current"]["humidity"]
        feels_like = data["current"]["feelslike_c"]
        wind_kph = data["current"]["wind_kph"]

        return (
            f"Current weather in {city}: {description}. "
            f"Temperature: {temperature}°C (feels like {feels_like}°C), "
            f"Humidity: {humidity}%, Wind: {wind_kph} km/h"
        )

    except requests.exceptions.HTTPError:
        return f"Error: Could not find weather for '{city}'."
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"


def run_agent(user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    max_steps = 5
    step = 0
    weather_info = None

    while step < max_steps:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0
            )
        except Exception as e:
            error_msg = f"❌ API Error: {str(e)}"
            log_query(user_input, error_msg)
            return error_msg

        content = response.choices[0].message.content.strip()

        try:
            content_json = parse_agent_response(content)
            action = content_json.get("action", "")
            action_input = content_json.get("action_input", "")

            if action == "user_answer":
                log_query(user_input, action_input, weather_info, source="API")
                return action_input

            if action == "get_weather":
                weather_info = get_weather(action_input)
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {weather_info}\n\nNow respond with a user_answer action."
                })
            else:
                error_msg = f"Unknown action: {action}"
                log_query(user_input, error_msg)
                return error_msg

        except json.JSONDecodeError:
            log_query(user_input, content, weather_info, source="API")
            return content

        step += 1

    error_msg = "⚠️ Max steps reached."
    log_query(user_input, error_msg)
    return error_msg

# =========================
# PARSE JSON (robust)
# =========================
def parse_agent_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()
    return json.loads(content)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Weather + AI Dashboard", layout="wide")
st.title("🌦 Weather + AI Dashboard with BERT Forecasting")

# Load BERT model and dataset
# with st.spinner("🔄 Loading AI model and weather dataset..."):
#     classifier= load_weather_model() #, tokenizer, model 
#     weather_df = load_weather_dataset()

with st.spinner("🔄 Loading AI model and weather dataset..."):
    classifier = load_weather_model()
    weather_df, dataset_latency = load_weather_dataset()

col1, col2 = st.columns([3, 2])





# -------------------------
# Left: Weather Dashboard
# -------------------------
with col1:
    city_input = st.text_input("🔍 Enter a city:", value="Kansas City")
    weather_data = fetch_weather(city_input)

    if "error" in weather_data:
        st.error(weather_data["error"])
    else:
        cur = parse_current(weather_data)
        
        # =============================
        # SECTION 1: WEATHER DATA
        # =============================
        with st.container(border=True):
            st.subheader(f"📍 {cur['city']}, {cur['region']}")
            st.caption(f"🕒 Local Time: {cur['localtime']}")
            
            # Main weather display
            main_col1, main_col2 = st.columns([1, 2])
            
            with main_col1:
                st.metric(
                    label="🌡️ Temperature",
                    value=f"{cur['temp_c']}°C",
                    delta=cur['condition']
                )
            
            with main_col2:
                st.info(f"**Condition:** {cur['condition']}")
            
            st.divider()
            
            # Metrics Row 1
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric(label="💧 Humidity", value=f"{cur['humidity']}%")
            
            with m2:
                st.metric(label="💨 Wind", value=f"{cur['wind_kph']} km/h")
            
            with m3:
                st.metric(label="🌡️ Pressure", value=f"{cur['pressure_mb']} mb")
            
            with m4:
                st.metric(label="👁️ Visibility", value=f"{cur['vis_km']} km")
            
            # Metrics Row 2
            m5, m6, m7, m8 = st.columns(4)
            
            with m5:
                st.metric(label="☀️ UV Index", value=cur['uv'])
            
            with m6:
                st.metric(label="🌫️ AQI", value=cur['aqi'])
            
            with m7:
                st.metric(label="🌅 Sunrise", value=cur['sunrise'])
            
            with m8:
                st.metric(label="🌇 Sunset", value=cur['sunset'])
        
        # # =============================
        # # SECTION 2: PLOTLY CHART
        # # =============================
        # with st.container(border=True):
        #     st.subheader("📊 24-Hour Forecast")
            
        #     times_12h, temps_12h, humidity_12h, conditions_12h = get_24h_data(weather_data)
            
        #     # Tab selection for different views
        #     tab1, tab2, tab3 = st.tabs(["📈 Combined", "🌡️ Temperature", "💧 Humidity"])
            
        #     with tab1:
        #         fig = go.Figure()
        #         fig.add_trace(go.Scatter(
        #             x=times_12h, 
        #             y=temps_12h, 
        #             name="Temp (°C)", 
        #             line=dict(color='#ff6b6b', width=3),
        #             fill='tozeroy',
        #             fillcolor='rgba(255, 107, 107, 0.1)',
        #             text=conditions_12h,  # this adds hover text
        #             hovertemplate='%{x}<br>%{y}°C<br>Condition: %{text}<extra></extra>'
        #         ))
        #         fig.add_trace(go.Scatter(
        #             x=times_12h, 
        #             y=humidity_12h, 
        #             name="Humidity %", 
        #             line=dict(color='#4ecdc4', width=3),
        #             yaxis="y2",
        #             text=conditions_12h,  # this adds hover text
        #             hovertemplate='%{x}<br>%{y}°C<br>Condition: %{text}<extra></extra>'
        #         ))
        #         fig.update_layout(
        #             yaxis=dict(title="Temperature (°C)"),
        #             yaxis2=dict(title="Humidity %", overlaying="y", side="right"),
        #             xaxis_tickangle=-45,
        #             height=400,
        #             hovermode='x unified',
        #             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        #         )
        #         st.plotly_chart(fig, use_container_width=True)
            
        #     with tab2:
        #         fig_temp = go.Figure()
        #         fig_temp.add_trace(go.Scatter(
        #             x=times_12h,
        #             y=temps_12h,
        #             name="Temperature",
        #             line=dict(color='#ff6b6b', width=3),
        #             fill='tozeroy',
        #             fillcolor='rgba(255, 107, 107, 0.2)',
        #             text=conditions_12h,  # this adds hover text
        #             hovertemplate='%{x}<br>%{y}°C<br>Condition: %{text}<extra></extra>'
                
        #         ))
        #         fig_temp.update_layout(
        #             yaxis=dict(title="Temperature (°C)"),
        #             xaxis_tickangle=-45,
        #             height=400,
        #             hovermode='x unified'
        #         )
        #         st.plotly_chart(fig_temp, use_container_width=True)
            
        #     with tab3:
        #         fig_hum = go.Figure()
        #         fig_hum.add_trace(go.Bar(
        #             x=times_12h,
        #             y=humidity_12h,
        #             name="Humidity",
        #             marker_color='#4ecdc4',
        #             text=conditions_12h,  # this adds hover text
        #             hovertemplate='%{x}<br>%{y}°C<br>Condition: %{text}<extra></extra>'
                
        #         ))
        #         fig_hum.update_layout(
        #             yaxis=dict(title="Humidity %"),
        #             xaxis_tickangle=-45,
        #             height=400,
        #             hovermode='x unified'
        #         )
        #         st.plotly_chart(fig_hum, use_container_width=True)
        # =============================
        # SECTION 2: PLOTLY ICON TIMELINE
        # =============================
        with st.container(border=True):
            st.subheader("📊 12-Hour Weather Forecast")

            # Get next 12 hours forecast + icons
            times_12h, temps_12h, humidity_12h, conditions_12h, icons_12h = [], [], [], [], []

            # Current local time of the city
            current_time_str = weather_data["location"]["localtime"]
            current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M")

            # Forecast hours
            hours = weather_data["forecast"]["forecastday"][0]["hour"]

            # Determine day/night based on forecast hour and sunrise/sunset
            sunrise_str = weather_data["forecast"]["forecastday"][0]["astro"]["sunrise"]
            sunset_str = weather_data["forecast"]["forecastday"][0]["astro"]["sunset"]
            sunrise = datetime.strptime(f"{current_time.date()} {sunrise_str}", "%Y-%m-%d %I:%M %p")
            sunset = datetime.strptime(f"{current_time.date()} {sunset_str}", "%Y-%m-%d %I:%M %p")

            
            # Map conditions to icons
            def weather_to_icon(condition: str, dt: datetime) -> str:
                condition = condition.lower()
                # print(f"Condition: {condition}, Time: {dt}")

                is_day = sunrise <= dt <= sunset
                if "sun" in condition or "clear" in condition:
                    return "☀️" if is_day else "🌙"
                elif "cloud" in condition or "overcast" in condition:
                    return "☁️" if is_day else "🌥️"
                elif "rain" in condition or "drizzle" in condition:
                    return "🌧️" if is_day else "🌧️"
                elif "snow" in condition or "sleet" in condition:
                    return "❄️"
                elif "storm" in condition or "thunder" in condition:
                    return "⛈️"
                elif "fog" in condition or "mist" in condition:
                    return "🌫️"
                elif "wind" in condition:
                    return "💨"
                else:
                    return "🌡️"
                
            # Filter next 12 hours
            for h in hours:
                h_time = datetime.strptime(h["time"], "%Y-%m-%d %H:%M")
                if current_time <= h_time <= current_time + timedelta(hours=12):
                    times_12h.append(h_time.strftime("%H:%M"))
                    temps_12h.append(h["temp_c"])
                    conditions_12h.append(h["condition"]["text"])
                    icons_12h.append(weather_to_icon(h["condition"]["text"], h_time))
                    humidity_12h.append(h["humidity"])

            # Prepend current weather
            cur = parse_current(weather_data)
            cur_time = datetime.strptime(cur["localtime"], "%Y-%m-%d %H:%M")
            cur_icon = weather_to_icon(cur["condition"], cur_time)

            times_12h = [cur["localtime"].split(" ")[1]] + times_12h
            temps_12h = [cur["temp_c"]] + temps_12h
            conditions_12h = [cur["condition"]] + conditions_12h
            icons_12h = [cur_icon] + icons_12h
            humidity_12h = [cur["humidity"]] + humidity_12h

             # Tab selection for different views
            tab1, tab2 = st.tabs(["📈 Combined", "💧 Humidity"])

            with tab1:
                # Plot icons on timeline
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=times_12h,
                    y=[0.5]*len(times_12h),  # single horizontal line
                    mode='text',
                    text=icons_12h,
                    textfont=dict(size=36),
                    hovertext=[f"{t} - {c} - {temp}°C" for t, c, temp in zip(times_12h, conditions_12h, temps_12h)],
                    hoverinfo='text'
                ))

                # Temp bars below icons
                fig.add_trace(go.Scatter(
                    x=times_12h,
                    y=[1]*len(times_12h),  # slightly below the icons
                    mode='text',
                    text=[f"{temp:.0f}°C" for temp in temps_12h],
                    textfont=dict(size=14, color="white"),
                    hoverinfo='skip',
                    showlegend=False
                ))

                # Layout adjustments
                fig.update_yaxes(visible=False, range=[0, 1.5])
                fig.update_xaxes(title_text="Time (next 12 hours)")
                fig.update_layout(
                    height=200,
                    margin=dict(t=20, b=40),
                    hovermode='x'
                )

                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig_hum = go.Figure()
                fig_hum.add_trace(go.Bar(
                    x=times_12h,
                    y=humidity_12h,
                    name="Humidity",
                    marker_color='#4ecdc4',
                    text=conditions_12h,  # this adds hover text
                    hovertemplate='%{x}<br>%{y}°C<br>Condition: %{text}<extra></extra>'
                
                ))
                fig_hum.update_layout(
                    yaxis=dict(title="Humidity %"),
                    xaxis_tickangle=-45,
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_hum, use_container_width=True)


# -------------------------
# Right: AI Assistant with BERT
# -------------------------
with col2:
    with st.container(border=True):
        st.subheader("🤖 AI Weather Assistant")
        st.caption("Powered by BERT + Historical Data")
        
        query_mode = st.radio(
            "Select Mode:",
            ["🔮 BERT Forecast", "🌐 Live Weather"],
            help="BERT uses historical data, Live uses real-time API"
        )
        
        ai_input = st.text_input(
            "Ask about weather:", 
            key="ai_input", 
            placeholder="e.g., Predict weather in London"
        )
        
        if st.button("🔍 Ask AI", key="ask_btn", use_container_width=True):
            if ai_input:
                with st.spinner("🔄 Processing..."):
                    if "BERT" in query_mode:
                        # Use BERT for prediction
                        if classifier:
                            prediction, error = predict_weather_with_bert(
                                ai_input, weather_df, classifier
                            )
                            
                            if prediction:
                                st.markdown(prediction)
                                log_query(ai_input, prediction, source="BERT")
                            else:
                                st.warning(f"⚠️ {error}")
                                # Fallback to API
                                st.info("🔄 Switching to Live Weather API...")
                                answer = run_agent(ai_input)
                                st.success(answer)
                        else:
                            st.error("❌ BERT model not loaded. Using API fallback.")
                            answer = run_agent(ai_input)
                            st.success(answer)
                    else:
                        # Use Live API
                        answer = run_agent(ai_input)
                        st.success(answer)
            else:
                st.warning("⚠️ Please enter a question.")
    
    with st.container(border=True):
        st.subheader("📊 Dataset Info")
        if weather_df is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📝 Records", len(weather_df))
            with col_b:
                st.metric("🏙️ Cities", weather_df['city'].nunique() if 'city' in weather_df.columns else 'N/A')
            
            with st.expander("🔍 View Sample Data"):
                st.dataframe(weather_df.head(10), use_container_width=True)
    
    with st.container(border=True):
        st.subheader("📋 Query Logs")
        
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = f.read()
            
            # Count logs
            log_count = logs.count("="*60) // 2
            st.caption(f"Total queries: {log_count}")
            
            with st.expander("📜 View Logs", expanded=False):
                st.code(logs[-2000:], language=None)
            
            if st.button("🗑️ Clear Logs", use_container_width=True):
                os.remove(LOG_FILE)
                st.rerun()
        else:
            st.info("ℹ️ No logs yet.")
    
    with st.container(border=True):
        st.subheader("📈 Backend Monitoring (Snowflake Queries)")
        import os

        if os.path.exists("pipeline_logs.csv"):
            logs_df = pd.read_csv("pipeline_logs.csv")
            st.metric("Total logged queries", len(logs_df))
            st.dataframe(logs_df.tail(20), use_container_width=True)

            try:
                logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
                logs_df = logs_df.sort_values("timestamp")
                logs_df.set_index("timestamp", inplace=True)
                st.line_chart(logs_df["latency_sec"], height=200)
            except Exception:
                pass
        else:
            st.info("No Snowflake pipeline logs yet. Trigger some queries first.")
        

# Footer
st.divider()
st.caption("🤖 Powered by BERT Weather Classification | 🌐 Live data from WeatherAPI")