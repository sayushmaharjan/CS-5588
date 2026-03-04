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
import torch

from datetime import datetime, timedelta
from app_auth import auth_ui

import folium
from streamlit_folium import st_folium
from folium import plugins
import streamlit.components.v1 as components
import time

# ── Agent module import ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.agent_runner import AgentRunner


DEV_MODE = True
# =========================
# SESSION STATE INIT
# =========================
if "user" not in st.session_state:
    st.session_state["user"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "sidebar_open" not in st.session_state:
    st.session_state["sidebar_open"] = True

# =========================
# AUTHENTICATION CHECK
# =========================
if "show_welcome" not in st.session_state:
    st.session_state["show_welcome"] = False

if DEV_MODE and st.session_state["user"] is None:
    # Auto login for development
    st.session_state["user"] = "dev_user"
    st.session_state["username"] = "Say"

if st.session_state["user"] is None:
    auth_ui()
    st.stop()
else:
    if not st.session_state["show_welcome"]:
        success_placeholder = st.empty()
        success_placeholder.success(f"👋 Welcome, {st.session_state['username']}")
        time.sleep(3)
        success_placeholder.empty()
        st.session_state["show_welcome"] = True
    

# =========================
# SIDEBAR TOGGLE & LOGOUT
# =========================
# if st.button("☰ Toggle Sidebar"):
#     st.session_state.sidebar_open = not st.session_state.sidebar_open

if st.session_state["user"] and st.session_state.sidebar_open:
    if st.sidebar.button("Logout"):
        st.session_state["user"] = None
        st.session_state["username"] = None
        st.session_state["show_welcome"] = False
        st.rerun()


# =========================
# LOAD ENV
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY") 
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # Add this to .env

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG_FILE = os.path.join(_PROJECT_ROOT, "chat_log.csv")

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
        # model_name = "oliverguhr/weather-classification"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        classifier = pipeline("zero-shot-classification",
                model="facebook/bart-large-mnli")

        token = os.getenv("HF_TOKEN")


        # Create pipeline for easier use
        classifier = pipeline(
            "text-classification",
            # model=model,
            # tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        

        return classifier#, tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading BERT model: {e}")
        return None#, None, None

# =========================
# WEATHER DATASET LOADING
# =========================
@st.cache_data
def load_weather_dataset():
    """
    Load historical weather dataset
    Dataset: https://huggingface.co/datasets/mongodb/weather
    Fallback to sample data if not available
    """
    try:
        _data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        df1 = pd.read_csv(os.path.join(_data_dir, "los_angeles.csv"))
        df1["city"] = "Los Angeles"

        df2 = pd.read_csv(os.path.join(_data_dir, "san_diego.csv"))
        df2["city"] = "San Diego"

        df3 = pd.read_csv(os.path.join(_data_dir, "san_francisco.csv"))
        df3["city"] = "San Francisco"

        df = pd.concat([df1, df2, df3], ignore_index=True)


        # from datasets import load_dataset
        # Create expected columns
        df["temperature"] = df["TAVG"]
        df["wind_speed"] = df["AWND"]
        df["precipitation"] = df["PRCP"]

        # Since we don't have condition, create simple rule-based condition
        df["condition"] = df.apply(
            lambda row: "Rainy" if row["PRCP"] > 0 else
                        "Snowy" if row["SNOW"] > 0 else
                        "Clear",
            axis=1
        )

        # Keep city lowercase-safe
        # df["city"] = df["city"].str.strip()

        # Drop rows with missing temperature
        df = df.dropna(subset=["temperature"])


        # Try loading from HuggingFace
        # dataset = load_dataset("mongodb/weather", split="train[:1000]")  # Limit to 1000 rows
        # df = pd.DataFrame(dataset)
        # st.success(f"✅ Loaded {len(df)} weather records from HuggingFace")
        return df
        
        # Try loading from HuggingFace
    #     dataset = load_dataset("mongodb/weather", split="train[:1000]")
    #     df = pd.DataFrame(dataset)
    #     st.success(f"✅ Loaded {len(df)} weather records from HuggingFace")
    #     return df
    except Exception as e:
        st.warning(f"⚠️ Could not load HuggingFace dataset: {e}")
        
    #     # Fallback: Create comprehensive sample weather data
        import numpy as np
        
        cities = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Berlin', 
                  'Rome', 'Madrid', 'Beijing', 'Moscow', 'Dubai', 'Singapore']
        
        conditions = ['Sunny', 'Partly Cloudy', 'Cloudy', 'Rainy', 'Stormy', 
                     'Snowy', 'Foggy', 'Windy']
        
        # Generate realistic weather data
        n_records = 1000
        
        sample_data = {
            'city': np.random.choice(cities, n_records),
            'temperature': np.random.normal(20, 10, n_records),
            'humidity': np.random.uniform(30, 90, n_records),
            'wind_speed': np.random.uniform(5, 40, n_records),
            'pressure': np.random.normal(1013, 20, n_records),
            'condition': np.random.choice(conditions, n_records),
            'date': pd.date_range('2023-01-01', periods=n_records, freq='6H')
        }
        
        df = pd.DataFrame(sample_data)
        st.info(f"ℹ️ Using sample dataset with {len(df)} records")
        return df

# =========================
# INTERACTIVE WEATHER MAP
# =========================
def create_weather_map(center_lat=39.8283, center_lon=-98.5795, zoom=4):
    """
    Create interactive weather map with real-time layers
    Uses OpenWeatherMap tiles for live weather visualization
    """
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add different tile layers with proper attributions
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.',
        name='Terrain',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB positron',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='Light Mode',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB dark_matter',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='Dark Mode',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add OpenWeatherMap layers (requires API key)
    if OPENWEATHER_API_KEY:
        # Temperature layer
        folium.TileLayer(
            tiles=f'https://tile.openweathermap.org/map/temp_new/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHER_API_KEY}',
            attr='&copy; <a href="https://openweathermap.org/">OpenWeatherMap</a>',
            name='🌡️ Temperature',
            overlay=True,
            control=True,
            opacity=0.6
        ).add_to(m)
        
        # Precipitation layer
        folium.TileLayer(
            tiles=f'https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHER_API_KEY}',
            attr='&copy; <a href="https://openweathermap.org/">OpenWeatherMap</a>',
            name='🌧️ Precipitation',
            overlay=True,
            control=True,
            opacity=0.6
        ).add_to(m)
        
        # Clouds layer
        folium.TileLayer(
            tiles=f'https://tile.openweathermap.org/map/clouds_new/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHER_API_KEY}',
            attr='&copy; <a href="https://openweathermap.org/">OpenWeatherMap</a>',
            name='☁️ Clouds',
            overlay=True,
            control=True,
            opacity=0.6
        ).add_to(m)
        
        # Wind layer
        folium.TileLayer(
            tiles=f'https://tile.openweathermap.org/map/wind_new/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHER_API_KEY}',
            attr='&copy; <a href="https://openweathermap.org/">OpenWeatherMap</a>',
            name='💨 Wind Speed',
            overlay=True,
            control=True,
            opacity=0.6
        ).add_to(m)
        
        # Pressure layer
        folium.TileLayer(
            tiles=f'https://tile.openweathermap.org/map/pressure_new/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHER_API_KEY}',
            attr='&copy; <a href="https://openweathermap.org/">OpenWeatherMap</a>',
            name='🌡️ Pressure',
            overlay=True,
            control=True,
            opacity=0.6
        ).add_to(m)
    
    # Add major cities with weather markers
    major_cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
        {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
        {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
        {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
    ]
    
    for city in major_cities:
        # Get current weather for the city
        weather_info = get_quick_weather(city["name"])
        
        # Determine marker color based on temperature
        temp = weather_info.get('temp', 20)
        if isinstance(temp, str):
            temp = 20  # Default if N/A
            
        if temp < 0:
            color = 'blue'
            icon = 'snowflake'
        elif temp < 10:
            color = 'lightblue'
            icon = 'cloud'
        elif temp < 20:
            color = 'green'
            icon = 'leaf'
        elif temp < 30:
            color = 'orange'
            icon = 'sun'
        else:
            color = 'red'
            icon = 'fire'
        
        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0;">{city['name']}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 5px 0;"><b>🌡️ Temp:</b> {weather_info.get('temp', 'N/A')}°C</p>
            <p style="margin: 5px 0;"><b>☁️ Condition:</b> {weather_info.get('condition', 'N/A')}</p>
            <p style="margin: 5px 0;"><b>💧 Humidity:</b> {weather_info.get('humidity', 'N/A')}%</p>
            <p style="margin: 5px 0;"><b>💨 Wind:</b> {weather_info.get('wind', 'N/A')} km/h</p>
            <small>Click anywhere to view full details</small>
        </div>
        """
        
        folium.Marker(
            location=[city["lat"], city["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{city['name']}: {weather_info.get('temp', 'N/A')}°C",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)
    
    # Add measure control
    plugins.MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
    
    # Add fullscreen option
    plugins.Fullscreen(position='topleft').add_to(m)
    
    # Add mouse position
    plugins.MousePosition().add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=True).add_to(m)
    
    return m

# =========================
# GET QUICK WEATHER INFO
# =========================
def get_quick_weather(city: str) -> dict:
    """Get quick weather data for map markers"""
    try:
        url = "http://api.weatherapi.com/v1/current.json"
        params = {"key": WEATHERAPI_KEY, "q": city, "aqi": "no"}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return {
            'temp': data['current']['temp_c'],
            'condition': data['current']['condition']['text'],
            'humidity': data['current']['humidity'],
            'wind': data['current']['wind_kph']
        }
    except:
        return {'temp': 'N/A', 'condition': 'N/A', 'humidity': 'N/A', 'wind': 'N/A'}

# =========================
# FETCH WEATHER BY COORDINATES
# =========================
def fetch_weather_by_coords(lat: float, lon: float):
    """Fetch weather data using latitude and longitude"""
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": WEATHERAPI_KEY, 
        "q": f"{lat},{lon}", 
        "days": 2, 
        "aqi": "yes"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data
    except Exception as e:
        return {"error": str(e)}

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
        prediction_input = f"Weather forecast for {city}: Based on recent patterns showing {context}"
        
        # Get classification
        result = classifier(prediction_input[:512])
        
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
    common_cities = [
        'new york', 'london', 'tokyo', 'paris', 'sydney', 'berlin',
        'rome', 'madrid', 'beijing', 'moscow', 'dubai', 'singapore',
        'los angeles', 'chicago', 'toronto', 'mumbai', 'delhi',
        'bangkok', 'hong kong', 'seoul', 'amsterdam', 'barcelona'
    ]
    
    query_lower = query.lower()
    
    for city in common_cities:
        if city in query_lower:
            return city.title()
    
    patterns = [" in ", " for ", " at "]
    for pattern in patterns:
        if pattern in query_lower:
            parts = query_lower.split(pattern)
            if len(parts) > 1:
                potential_city = parts[1].split()[0].strip('?,.')
                return potential_city.title()
    
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
# CHAT HISTORY CSV
# =========================
CHAT_CSV = os.path.join(_PROJECT_ROOT, "chat_log.csv")

def save_chat_to_csv(role: str, content: str, city: str = ""):
    """Append a chat message to CSV file"""
    file_exists = os.path.exists(CHAT_CSV)
    with open(CHAT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = __import__('csv').writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "role", "content", "city"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), role, content, city])

def load_chat_from_csv():
    """Load chat history from CSV"""
    if not os.path.exists(CHAT_CSV):
        return []
    try:
        history = []
        with open(CHAT_CSV, "r", encoding="utf-8") as f:
            reader = __import__('csv').DictReader(f)
            for row in reader:
                if "role" in row and "content" in row:
                    history.append({"role": row["role"], "content": row["content"], "city": row.get("city", "")})
        return history
    except Exception:
        return []


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
        "country": loc.get("country", ""),
        "lat": loc.get("lat", 0),
        "lon": loc.get("lon", 0),
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
    # return times, temps, humidity

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
# AI OVERVIEW FUNCTION
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def get_ai_overview(_weather_json: str) -> str:
    """Generate a natural-language weather overview using the LLM."""
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a weather forecaster that responds in natural language."},
                {"role": "user", "content": f"How should one prepare for the weather today? What should they wear, pack, or keep in mind? The JSON data for the local weather forecast is below. Talk as if you're on air for one person. 350 characters maximum. {_weather_json}"},
            ],
            model=MODEL,
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to generate overview: {e}"

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
st.set_page_config(page_title="WeatherTwin AI", layout="wide")

st.markdown("""
<style>
    :root {
        --color-humidity: #5AC8FA;
        --color-uv: #FF9F0A;
        --color-temp: #FF453A;
        --color-wind: #30D158;
        --color-pressure: #BF5AF2;
        --color-live: #32D74B;
        --color-border: rgba(164, 164, 164, 0.34);
        --radius: 12px;
    }
    /* Card padding & radius */
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 20px !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--color-border) !important;
    }
    /* Metric styling */
    [data-testid="stMetricValue"] { font-weight: 500 !important; font-size: 1.5rem !important; }
    [data-testid="stMetricLabel"] { opacity: 0.55 !important; font-weight: 600 !important; font-size: 0.78rem !important; text-transform: uppercase !important; }
    /* Glassmorphism inputs */
    [data-testid="stTextInput"] > div > div > input, .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 4px !important;
        padding: 12px 16px !important;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 8px !important; font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border: 1px solid var(--color-border) !important;
    }
    .stButton > button:hover { transform: translateY(-1px) !important; }
    /* Nav bar */
    .top-nav { display:flex; align-items:center; justify-content:space-between; padding:8px 0 16px 0; gap:16px; flex-wrap:wrap; }
    .nav-brand { font-size:1.6rem; font-weight:700; margin:0; white-space:nowrap; }
    .nav-badges { display:flex; gap:12px; flex-shrink:0; }
    .nav-badge { display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:16px; font-size:0.72rem; font-weight:600; border:1px solid rgba(255,255,255,0.08); }
    .nb-green { background:rgba(50,215,75,0.12); color:#32D74B; }
    .nb-blue { background:rgba(90,200,250,0.12); color:#5AC8FA; }
    /* Chat bubbles */
    .chat-user { background:rgba(90,200,250,0.08); border:1px solid rgba(90,200,250,0.15); border-radius:12px 12px 4px 12px; padding:12px 16px; margin:8px 0 8px 40px; }
    .chat-ai { background:rgba(255,255,255,0.04); border:1px solid var(--color-border); border-radius:12px 12px 12px 4px; padding:16px; margin:8px 40px 8px 0; }
    .chat-role { font-size:0.72rem; font-weight:600; text-transform:uppercase; opacity:0.5; margin-bottom:4px; }
    /* Quick chips */
    .chip-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
    /* Delta chips */
    .delta-chip { display:inline-block; padding:2px 10px; border-radius:6px; font-size:0.75rem; font-weight:600; margin-top:4px; }
    .delta-good { background:rgba(50,215,75,0.15); color:#32D74B; }
    .delta-warn { background:rgba(255,159,10,0.15); color:#FF9F0A; }
    .delta-bad { background:rgba(255,69,58,0.15); color:#FF453A; }
    /* Hero temp */
    .hero-temp { font-size:2.5rem !important; font-weight:400; line-height:1; margin:0; }
    .hero-condition { font-size:1.1rem; font-weight:500; opacity:0.7; margin-top:4px; }
    /* Condition card */
    .condition-card { border-radius:var(--radius); padding:16px 24px; text-align:center; color:white; text-shadow:0 1px 3px rgba(0,0,0,0.3); }
    .condition-card .icon { font-size:3rem; margin-bottom:6px; }
    .condition-card .label { font-weight:700; font-size:1rem; }
    /* Radio inline */
    [data-testid="stRadio"] > div { flex-direction:row !important; gap:8px !important; }
    /* Expander */
    [data-testid="stExpander"] { border-radius:var(--radius) !important; border:1px solid var(--color-border) !important; }
    /* Arrow button border */
    .arrow-btn button { border:1.5px solid rgba(255,255,255,0.18) !important; }
    hr { border:none !important; border-top:1px solid rgba(255,255,255,0.06) !important; margin:12px 0 !important; }
    
    /* Fixed Forecast Dock */
    .forecast-dock {
        position:fixed; bottom:0; left:0; right:0; z-index:9999;
        background: var(--secondary-background-color);
        color: var(--text-color);
        backdrop-filter: blur(16px);
        border-top:1px solid rgba(49, 51, 63, 0.4);
        display:flex; align-items:center; padding:0; height:90px;
    }
    .dock-label {
        min-width:180px; padding:0 20px;
        border-right:1px solid rgba(255,255,255,0.1);
        display:flex; flex-direction:column; justify-content:center; height:100%;
    }
    .dock-label-title { font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; opacity:0.6; margin-bottom:6px; }
    .dock-tabs { display:flex; gap:0; }
    .dock-tab { padding:2px 10px; font-size:0.68rem; font-weight:600; cursor:pointer; border-radius:4px; opacity:0.6; transition: all 0.15s ease; }
    .dock-tab:hover { opacity:0.7; }
    .dock-tab.active { opacity:1; background:rgba(255,255,255,0.1); }
    .dock-items {
        display:flex; flex:1; overflow-x:auto; height:100%;
    }
    .dock-item {
        flex:1; min-width:70px; display:flex; flex-direction:column;
        align-items:center; justify-content:center; gap:2px;
        border-right:1px solid rgba(255,255,255,0.05); padding:6px 4px;
    }
    .dock-item:last-child { border-right:none; }
    .dock-icon { font-size:1.4rem; }
    .dock-time { font-size:0.65rem; opacity:0.5; font-weight:600; }
    .dock-val { font-size:0.82rem; font-weight:500; }
    /* Add bottom padding to main content so dock doesn't overlap */
    [data-testid="stMain"] { padding-bottom:100px !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if 'selected_lat' not in st.session_state:
    st.session_state.selected_lat = None
if 'selected_lon' not in st.session_state:
    st.session_state.selected_lon = None
if 'map_city' not in st.session_state:
    st.session_state.map_city = "New York"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_from_csv()
if 'all_plans' not in st.session_state:
    st.session_state.all_plans = [m for m in st.session_state.chat_history if m.get('role') == 'assistant']

with st.spinner("Loading AI model and weather dataset..."):
    classifier = load_weather_model()
    weather_df = load_weather_dataset()

# =========================
# TOP NAVIGATION BAR
# =========================
bert_ok = "✓" if classifier else "✗"
nav_col1, nav_col2, nav_col3 = st.columns([2, 4, 2])
with nav_col1:
    st.markdown('<p class="nav-brand">WeatherTwin AI</p>', unsafe_allow_html=True)
with nav_col2:
    city_input = st.text_input("Search", value=st.session_state.map_city, key="city_search",
                               label_visibility="collapsed", placeholder="Search for a city...")
with nav_col3:
    _n_records = len(weather_df) if weather_df is not None else 0
    _n_cities = weather_df['city'].nunique() if weather_df is not None and 'city' in weather_df.columns else 0
    st.markdown(f'''<div class="nav-badges">
        <span class="nav-badge nb-green">Model: BERT {bert_ok}</span>
        <span class="nav-badge nb-blue">Records: {_n_records}</span>
        <span class="nav-badge nb-blue">Cities: {_n_cities}</span>
    </div>''', unsafe_allow_html=True)

# Handle search
if city_input and city_input != st.session_state.map_city:
    st.session_state.map_city = city_input
    st.session_state.selected_lat = None
    st.session_state.selected_lon = None

# Fetch weather data globally
if st.session_state.selected_lat and st.session_state.selected_lon:
    weather_data = fetch_weather_by_coords(st.session_state.selected_lat, st.session_state.selected_lon)
else:
    weather_data = fetch_weather(st.session_state.map_city)


# =============================================
# TOP SECTION: AI Overview (left) + Weather Detail (right)
# =============================================
if "error" not in weather_data:
    cur = parse_current(weather_data)

    overview_col, detail_col = st.columns([2, 3])

    # ── LEFT: AI Overview ──
    with overview_col:
        with st.container(border=True):
            st.markdown("**🤖 AI Weather Overview**")
            st.caption(f"{cur['city']} · {cur['localtime']}")
            weather_json_str = json.dumps({
                "city": cur["city"],
                "temp_c": cur["temp_c"],
                "condition": cur["condition"],
                "humidity": cur["humidity"],
                "wind_kph": cur["wind_kph"],
                "uv": cur["uv"],
                "vis_km": cur["vis_km"],
                "sunrise": cur["sunrise"],
                "sunset": cur["sunset"],
            })
            with st.spinner("Generating overview..."):
                overview_text = get_ai_overview(weather_json_str)
            st.markdown(f'<div style="font-size:1.05rem; line-height:1.6; padding:8px 0;">{overview_text}</div>', unsafe_allow_html=True)

    # ── RIGHT: Weather Detail Card ──
    with detail_col:
        with st.container(border=True):
            # Unit toggle inside card header
            hero_col, toggle_col = st.columns([5, 1])
            with toggle_col:
                use_imperial = st.toggle("°F", key="unit_toggle")

            if use_imperial:
                temp_val = f"{(cur['temp_c'] * 9/5) + 32:.1f}"
                temp_unit = "°F"
                wind_disp = f"{cur['wind_kph'] * 0.621371:.1f} mph"
                vis_disp = f"{cur['vis_km'] * 0.621371:.1f} mi"
                press_disp = f"{cur['pressure_mb'] * 0.02953:.2f} inHg"
            else:
                temp_val = f"{cur['temp_c']}"
                temp_unit = "°C"
                wind_disp = f"{cur['wind_kph']} km/h"
                vis_disp = f"{cur['vis_km']} km"
                press_disp = f"{cur['pressure_mb']} mb"

            # Hero temp row
            cond_l = cur['condition'].lower()
            ic = "☀️" if ("sun" in cond_l or "clear" in cond_l) else "🌧️" if ("rain" in cond_l or "drizzle" in cond_l) else "☁️" if ("cloud" in cond_l or "overcast" in cond_l) else "❄️" if "snow" in cond_l else "⛈️" if ("storm" in cond_l or "thunder" in cond_l) else "🌤️"

            with hero_col:
                st.markdown(f"""<div style="display:flex; align-items:center; gap:16px;">
                    <div>
                        <p class="hero-temp">{temp_val}<span style="font-size:2rem; opacity:0.6;">{temp_unit}</span></p>
                        <p class="hero-condition">{ic} {cur['condition']}</p>
                    </div>
                    <div style="opacity:0.6; font-size:0.85rem;">
                        {cur['city']}, {cur['region']}<br>{cur['localtime']}
                    </div>
                </div>""", unsafe_allow_html=True)


            # 6 metric tiles in 3 columns — all with delta badges
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Humidity", f"{cur['humidity']}%")
                h_cls = "delta-good" if cur['humidity'] < 60 else ("delta-warn" if cur['humidity'] < 80 else "delta-bad")
                h_lbl = "Normal" if cur['humidity'] < 60 else ("High" if cur['humidity'] < 80 else "Very High")
                st.markdown(f'<span class="delta-chip {h_cls}">{h_lbl}</span>', unsafe_allow_html=True)
            with c2:
                st.metric("Wind", wind_disp)
                w_cls = "delta-good" if cur['wind_kph'] < 20 else ("delta-warn" if cur['wind_kph'] < 50 else "delta-bad")
                w_lbl = "Calm" if cur['wind_kph'] < 20 else ("Breezy" if cur['wind_kph'] < 50 else "Strong")
                st.markdown(f'<span class="delta-chip {w_cls}">{w_lbl}</span>', unsafe_allow_html=True)
            with c3:
                uv_val = float(cur['uv']) if str(cur['uv']).replace('.','',1).isdigit() else 0
                st.metric("UV Index", cur['uv'])
                uv_cls = "delta-good" if uv_val <= 2 else ("delta-warn" if uv_val <= 7 else "delta-bad")
                uv_lbl = "Low" if uv_val <= 2 else ("Moderate" if uv_val <= 5 else "High")
                st.markdown(f'<span class="delta-chip {uv_cls}">{uv_lbl}</span>', unsafe_allow_html=True)
            with c4:
                st.metric("Pressure", press_disp)
                p_val = cur['pressure_mb']
                p_cls = "delta-good" if 1000 <= p_val <= 1025 else ("delta-warn" if 980 <= p_val < 1000 or 1025 < p_val <= 1040 else "delta-bad")
                p_lbl = "Normal" if 1000 <= p_val <= 1025 else ("Low" if p_val < 1000 else "High")
                st.markdown(f'<span class="delta-chip {p_cls}">{p_lbl}</span>', unsafe_allow_html=True)
            with c5:
                st.metric("Visibility", vis_disp)
                v_val = cur['vis_km']
                v_cls = "delta-good" if v_val >= 10 else ("delta-warn" if v_val >= 5 else "delta-bad")
                v_lbl = "Clear" if v_val >= 10 else ("Moderate" if v_val >= 5 else "Poor")
                st.markdown(f'<span class="delta-chip {v_cls}">{v_lbl}</span>', unsafe_allow_html=True)
            with c6:
                aqi_raw = cur.get('aqi', 'N/A')
                st.metric("AQI", aqi_raw)
                if str(aqi_raw).isdigit():
                    aqi_v = int(aqi_raw)
                    a_cls = "delta-good" if aqi_v <= 2 else ("delta-warn" if aqi_v <= 4 else "delta-bad")
                    a_lbl = "Good" if aqi_v <= 2 else ("Moderate" if aqi_v <= 4 else "Unhealthy")
                    st.markdown(f'<span class="delta-chip {a_cls}">{a_lbl}</span>', unsafe_allow_html=True)
else:
    st.error(weather_data.get("error", "Could not fetch weather data."))

# =========================
# MAIN SPLIT-PANE LAYOUT
# =========================
left_pane, right_pane = st.columns([3, 2])

# ─────────────────────────────────────
# LEFT PANE: AI Planning Workspace
# ─────────────────────────────────────

#auto scroll working after refresh

@st.fragment #refreshes the chat section only
def chat_section():
    with st.container(border=True):
        # Header: radio left, title center-ish, action buttons right
        hdr_radio, hdr_title, hdr_actions = st.columns([2, 3, 2])
        with hdr_radio:
            query_mode = st.radio("Mode", ["BERT Forecast", "Live Weather"],
                                  label_visibility="collapsed")
        with hdr_title:
            st.markdown("**AI Weather Assistant**")
            st.caption("Powered by BERT + Historical Data")
        with hdr_actions:
            act1, act2 = st.columns(2)
            with act1:
                if st.button("＋ New", key="new_chat_btn", use_container_width=True):
                    for m in st.session_state.chat_history:
                        if m.get('role') == 'assistant':
                            st.session_state.all_plans.append(m)
                    if os.path.exists(CHAT_CSV) and os.path.getsize(CHAT_CSV) > 0:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        archive_name = os.path.join(_PROJECT_ROOT, f"chat_log_{ts}.csv")
                        os.rename(CHAT_CSV, archive_name)
                    st.session_state.chat_history = []
                    st.rerun()
            with act2:
                if st.button("🗑️", key="clear_chat_btn", use_container_width=True):
                    st.session_state.chat_history = []
                    if os.path.exists(CHAT_CSV):
                        os.remove(CHAT_CSV)
                    if os.path.exists(LOG_FILE):
                        os.remove(LOG_FILE)
                    st.rerun()

        st.divider()

        # Chat history display (auto-scrolls to bottom)
        chat_container = st.container(height=400)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("<div style='text-align:center; opacity:0.4; padding:60px 0;'>"
                            "Ask me about weather anywhere.<br>Try the quick actions below!</div>",
                            unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-user"><div class="chat-role">You</div>{msg["content"]}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-ai"><div class="chat-role">WeatherTwin AI</div>{msg["content"]}</div>',
                                    unsafe_allow_html=True)
                        # Show agent reasoning trace if available
                        agent_steps = msg.get("agent_steps", [])
                        agent_tools = msg.get("tools_used", [])
                        if agent_steps:
                            with st.expander(f"🔍 Agent Trace — {len(agent_steps)} step(s), tools: {', '.join(agent_tools) if agent_tools else 'none'}"):
                                for s in agent_steps:
                                    st.markdown(f"**Step {s['step']}** — `{s['action']}`")
                                    if s.get('thought'):
                                        st.caption(f"💭 {s['thought']}")
                                    if s.get('action_input') and s['action'] != 'final_answer':
                                        st.code(json.dumps(s['action_input'], indent=2), language='json')
                                    if s.get('observation'):
                                        obs = s['observation']
                                        if isinstance(obs, dict):
                                            st.json(obs)
                                        else:
                                            st.text(str(obs)[:500])
                                    st.divider()
                # Anchor at the bottom
                st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)

        # Auto-scroll using components.html — this actually runs JS
        if st.session_state.chat_history:
            components.html('''
            <script>
                function scrollChat() {
                    const doc = window.parent.document;
                    const anchor = doc.getElementById('chat-bottom-anchor');
                    if (anchor) {
                        anchor.scrollIntoView({behavior: 'smooth', block: 'end'});
                    } else {
                        // Fallback: scroll the inner scrollable container
                        const containers = doc.querySelectorAll('[data-testid="stVerticalBlock"]');
                        containers.forEach(c => {
                            if (c.closest('[style*="overflow"]') || c.scrollHeight > c.clientHeight) {
                                c.scrollTop = c.scrollHeight;
                            }
                        });
                    }
                }
                // Small delay to let Streamlit finish rendering
                setTimeout(scrollChat, 200);
                setTimeout(scrollChat, 600);
            </script>
            ''', height=0)

        st.divider()

        # Quick Action Chips
        chip_cols = st.columns(4)
        chip_prompts = {
            "☀️ Beach Day": "What's the best beach weather this weekend in Malibu?",
            "🥾 Hiking Trip": "Is it good weather for hiking at Yosemite this weekend?",
            "🎿 Ski Conditions": "What are the skiing conditions at Lake Tahoe?",
            "🌧️ Rain Check": f"Will it rain in {st.session_state.map_city} today?"
        }
        selected_chip = None
        for i, (label, prompt) in enumerate(chip_prompts.items()):
            with chip_cols[i]:
                if st.button(label, key=f"chip_{i}", use_container_width=True):
                    selected_chip = prompt

        # Input area with arrow button
        in_col, btn_col = st.columns([7, 1])
        with in_col:
            ai_input = st.text_area("Ask", key="ai_input", placeholder="Plan my day around the weather...",
                                    label_visibility="collapsed", height=50)
        with btn_col:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            ask_clicked = st.button("→", key="ask_btn")

        final_input = selected_chip if selected_chip else (ai_input if ask_clicked else None)

        if final_input:
            st.session_state.chat_history.append({"role": "user", "content": final_input, "city": st.session_state.map_city})
            save_chat_to_csv("user", final_input, st.session_state.map_city)

            with st.spinner("🤖 Agent is reasoning..."):
                if "BERT" in query_mode and classifier:
                    prediction, error = predict_weather_with_bert(final_input, weather_df, classifier)
                    if prediction:
                        response_text = prediction
                        agent_steps = []
                        agent_tools = ["predict_weather_bert (direct)"]
                        log_query(final_input, prediction, source="BERT")
                    else:
                        # Fall through to agent
                        agent = AgentRunner()
                        result = agent.run(final_input)
                        response_text = result["answer"]
                        agent_steps = result["steps"]
                        agent_tools = result["tools_used"]
                        log_query(final_input, response_text, source="Agent")
                else:
                    agent = AgentRunner()
                    result = agent.run(final_input)
                    response_text = result["answer"]
                    agent_steps = result["steps"]
                    agent_tools = result["tools_used"]
                    log_query(final_input, response_text, source="Agent")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "city": st.session_state.map_city,
                "agent_steps": agent_steps,
                "tools_used": agent_tools,
            })
            save_chat_to_csv("assistant", response_text, st.session_state.map_city)
            st.rerun(scope="fragment")

with left_pane:
    chat_section()

# ─────────────────────────────────────
# RIGHT PANE: Dynamic Supporting Data
# ─────────────────────────────────────
with right_pane:
    # Top Card: Compact Map
    with st.container(border=True):
        st.caption("Interactive Map")
        weather_map = create_weather_map()
        map_data = st_folium(weather_map, width=None, height=320, key="weather_map",
                             returned_objects=["last_clicked"])
        if map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            if (st.session_state.selected_lat != clicked_lat or
                st.session_state.selected_lon != clicked_lon):
                st.session_state.selected_lat = clicked_lat
                st.session_state.selected_lon = clicked_lon
                st.rerun()



    # Bottom Card: Recently Generated Plans
    with st.container(border=True):
        st.markdown("**Recently Generated Plans**")
        # Combine current session plans + archived plans
        current_ai = [m for m in st.session_state.chat_history if m.get('role') == 'assistant']
        all_plans = st.session_state.all_plans + current_ai
        # Deduplicate by content
        seen = set()
        unique_plans = []
        for m in all_plans:
            key = m.get('content', '')[:200]
            if key not in seen:
                seen.add(key)
                unique_plans.append(m)
        if unique_plans:
            with st.expander(f"View {len(unique_plans)} past responses", expanded=False):
                for idx, m in enumerate(reversed(unique_plans[-10:])):
                    preview = m["content"][:120] + "..." if len(m["content"]) > 120 else m["content"]
                    city_tag = f' · {m.get("city", "")}' if m.get("city") else ""
                    st.caption(f"#{len(unique_plans) - idx}{city_tag}")
                    st.markdown(preview)
                    st.divider()
            if st.button("Clear All History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.all_plans = []
                if os.path.exists(CHAT_CSV):
                    os.remove(CHAT_CSV)
                if os.path.exists(LOG_FILE):
                    os.remove(LOG_FILE)
                # Also remove archived logs
                import glob
                for f in glob.glob(os.path.join(_PROJECT_ROOT, "chat_log_*.csv")):
                    os.remove(f)
                st.rerun()
        else:
            st.caption("No plans generated yet. Ask the AI assistant!")

# =============================
# FIXED FORECAST DOCK (Bottom)
# =============================
# Read dock mode from query params (allows HTML tabs to toggle)
dock_mode = st.query_params.get("dock", "temp")

if 'weather_data' in locals() and 'error' not in weather_data:
    current_time_str = weather_data["location"]["localtime"]
    current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M")
    end_time = current_time + timedelta(hours=12)

    sunrise_str = weather_data["forecast"]["forecastday"][0]["astro"]["sunrise"]
    sunset_str = weather_data["forecast"]["forecastday"][0]["astro"]["sunset"]
    sunrise = datetime.strptime(f"{current_time.date()} {sunrise_str}", "%Y-%m-%d %I:%M %p")
    sunset = datetime.strptime(f"{current_time.date()} {sunset_str}", "%Y-%m-%d %I:%M %p")

    def weather_to_icon(condition: str, dt: datetime) -> str:
        condition = condition.lower()
        is_day = sunrise <= dt <= sunset
        if "sun" in condition or "clear" in condition:
            return "☀️" if is_day else "🌙"
        elif "cloud" in condition or "overcast" in condition:
            return "☁️" if is_day else "🌥️"
        elif "rain" in condition or "drizzle" in condition:
            return "🌧️"
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

    all_hours = []
    for day in weather_data["forecast"]["forecastday"]:
        all_hours.extend(day["hour"])

    forecast_items = []
    _imperial = st.session_state.get('unit_toggle', False)
    for h in all_hours:
        h_time = datetime.strptime(h["time"], "%Y-%m-%d %H:%M")
        if current_time <= h_time <= end_time:
            ic = weather_to_icon(h["condition"]["text"], h_time)
            if _imperial:
                t_display = f"{(h['temp_c'] * 9/5) + 32:.0f}°F"
            else:
                t_display = f"{h['temp_c']:.0f}°C"
            forecast_items.append({
                "time": h_time.strftime("%H:%M"),
                "icon": ic,
                "temp": t_display,
                "humidity": f"{h['humidity']}%"
            })

    # Build HTML with BOTH values in each item
    items_html = ""
    for item in forecast_items:
        items_html += f'''<div class="dock-item">
            <span class="dock-icon">{item["icon"]}</span>
            <span class="dock-val dock-val-temp">{item["temp"]}</span>
            <span class="dock-val dock-val-hum" style="display:none;">{item["humidity"]}</span>
            <span class="dock-time">{item["time"]}</span>
        </div>'''

    # Dock rendered with st.markdown — your existing CSS applies as-is
    st.markdown(f'''
    <div class="forecast-dock">
        <div class="dock-label">
            <span class="dock-label-title">12-HOUR FORECAST</span>
            <div class="dock-tabs">
                <span class="dock-tab active" id="dock-tab-temp">Temp</span>
                <span class="dock-tab" id="dock-tab-hum">Humidity</span>
            </div>
        </div>
        <div class="dock-items">{items_html}</div>
    </div>
    ''', unsafe_allow_html=True)

    # Invisible iframe that injects JS into the parent page
    components.html('''
    <script>
        function setupDockToggle() {
            const doc = window.parent.document;
            const tabTemp = doc.getElementById('dock-tab-temp');
            const tabHum = doc.getElementById('dock-tab-hum');

            if (!tabTemp || !tabHum) {
                setTimeout(setupDockToggle, 100);
                return;
            }

            // Avoid attaching duplicate listeners on rerun
            if (tabTemp.dataset.bound) return;
            tabTemp.dataset.bound = 'true';

            tabTemp.addEventListener('click', function() {
                doc.querySelectorAll('.dock-val-temp').forEach(el => el.style.display = 'inline');
                doc.querySelectorAll('.dock-val-hum').forEach(el => el.style.display = 'none');
                tabTemp.classList.add('active');
                tabHum.classList.remove('active');
            });

            tabHum.addEventListener('click', function() {
                doc.querySelectorAll('.dock-val-temp').forEach(el => el.style.display = 'none');
                doc.querySelectorAll('.dock-val-hum').forEach(el => el.style.display = 'inline');
                tabHum.classList.add('active');
                tabTemp.classList.remove('active');
            });
        }
        setupDockToggle();
    </script>
    ''', height=0)


