# app.py
import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
# from sentence_transformers import SentenceTransformer

import torch

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
# RAG MODEL SETUP
# =========================
@st.cache_resource


# Link: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
def load_rag_model():
    """Load RAG model for weather forecasting"""
    try:
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", 
            index_name="exact", 
            use_dummy_dataset=True
        )
        model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

        # Lightweight Option
        # model = SentenceTransformer('all-MiniLM-L6-v2')

        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading RAG model: {e}")
        return None, None

# =========================
# WEATHER DATASET LOADING
# =========================
@st.cache_data
def load_weather_dataset():
    """
    Load historical weather dataset from Hugging Face
    Dataset: https://huggingface.co/datasets/mongodb/weather
    """
    try:
        from datasets import load_dataset
        
        # Load weather dataset from Hugging Face
        # dataset = load_dataset("extreme-weather-impacts/classified_data_10Q", split="train")
        # df = pd.DataFrame(dataset)
        
        # Alternative: Use local CSV if dataset not available
        # You can also use: https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data

        df1 = pd.read_csv("/Users/sayush/Documents/cs5588/CS-5588/week-4/data/los_angeles.csv")
        df2 = pd.read_csv("/Users/sayush/Documents/cs5588/CS-5588/week-4/data/san_diego.csv")
        df3 = pd.read_csv("/Users/sayush/Documents/cs5588/CS-5588/week-4/data/san_francisco.csv")

        df = pd.concat([df1, df2, df3], ignore_index=True)
        
        return df
    except Exception as e:
        st.warning(f"Could not load HuggingFace dataset: {e}")
        
        # Fallback: Create sample weather data
        sample_data = {
            'city': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'] * 100,
            'temperature': [20 + i % 15 for i in range(500)],
            'humidity': [60 + i % 30 for i in range(500)],
            'wind_speed': [10 + i % 20 for i in range(500)],
            'condition': ['Sunny', 'Rainy', 'Cloudy', 'Snowy', 'Foggy'] * 100,
            'date': pd.date_range('2023-01-01', periods=500, freq='D')
        }
        return pd.DataFrame(sample_data)

# =========================
# WEATHER KNOWLEDGE BASE
# =========================
def create_weather_knowledge_base(df):
    """Create a searchable knowledge base from weather data"""
    knowledge = []
    
    for idx, row in df.iterrows():
        entry = f"In {row.get('city', 'Unknown')}, temperature was {row.get('temperature', 'N/A')}°C, "
        entry += f"humidity {row.get('humidity', 'N/A')}%, wind speed {row.get('wind_speed', 'N/A')} km/h, "
        entry += f"condition: {row.get('condition', 'N/A')}"
        knowledge.append(entry)
    
    return knowledge

# =========================
# RAG-BASED PREDICTION
# =========================
def predict_weather_with_rag(query: str, weather_df: pd.DataFrame, tokenizer, model):
    """Use RAG to predict weather based on historical data"""
    try:
        # Extract city from query
        city = extract_city_from_query(query)
        
        if not city:
            return None, "Could not identify city in query"
        
        # Filter dataset for the specific city
        city_data = weather_df[weather_df['city'].str.lower() == city.lower()]
        
        if city_data.empty:
            return None, f"No historical data found for {city}"
        
        # Create context from historical data
        context = create_weather_knowledge_base(city_data.tail(10))
        context_str = " ".join(context)
        
        # Prepare input for RAG model
        input_text = f"Based on historical weather data: {context_str}. Question: {query}"
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Statistical analysis
        stats = {
            'avg_temp': city_data['temperature'].mean(),
            'avg_humidity': city_data['humidity'].mean(),
            'avg_wind': city_data['wind_speed'].mean(),
            'common_condition': city_data['condition'].mode()[0] if not city_data['condition'].mode().empty else 'N/A'
        }
        
        forecast = f"""
📊 **Weather Forecast for {city}** (Based on Historical Data)

🤖 **RAG Model Prediction:** {prediction}

📈 **Statistical Analysis:**
- Average Temperature: {stats['avg_temp']:.1f}°C
- Average Humidity: {stats['avg_humidity']:.1f}%
- Average Wind Speed: {stats['avg_wind']:.1f} km/h
- Most Common Condition: {stats['common_condition']}

⚠️ This prediction is based on historical patterns from the dataset.
"""
        
        return forecast, None
        
    except Exception as e:
        return None, f"RAG prediction error: {str(e)}"

# =========================
# EXTRACT CITY FROM QUERY
# =========================
def extract_city_from_query(query: str) -> str:
    """Extract city name from user query"""
    # Simple extraction - can be improved with NER
    common_cities = ['new york', 'london', 'tokyo', 'paris', 'sydney', 
                     'berlin', 'rome', 'madrid', 'beijing', 'moscow']
    
    query_lower = query.lower()
    for city in common_cities:
        if city in query_lower:
            return city.title()
    
    # Try to extract using "in" keyword
    if " in " in query_lower:
        parts = query_lower.split(" in ")
        if len(parts) > 1:
            potential_city = parts[1].split()[0]
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
    hours = data["forecast"]["forecastday"][0]["hour"]
    times = [h["time"].split(" ")[1] for h in hours]
    temps = [h["temp_c"] for h in hours]
    humidity = [h["humidity"] for h in hours]
    return times, temps, humidity

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
st.title("🌦 Weather + AI Dashboard with RAG Forecasting")

# Load RAG model and dataset
with st.spinner("Loading RAG model and weather dataset..."):
    tokenizer, rag_model = load_rag_model()
    weather_df = load_weather_dataset()

col1, col2 = st.columns([3, 1])

# -------------------------
# Left: Weather Dashboard
# -------------------------
with col1:
    city_input = st.text_input("🔍 Enter a city:", value="New York")
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
        
        # =============================
        # SECTION 2: PLOTLY CHART
        # =============================
        with st.container(border=True):
            st.subheader("📊 24-Hour Forecast")
            
            times, temps, humidity = get_24h_data(weather_data)
            
            # Tab selection for different views
            tab1, tab2, tab3 = st.tabs(["📈 Combined", "🌡️ Temperature", "💧 Humidity"])
            
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=times, 
                    y=temps, 
                    name="Temp (°C)", 
                    line=dict(color='#ff6b6b', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.1)'
                ))
                fig.add_trace(go.Scatter(
                    x=times, 
                    y=humidity, 
                    name="Humidity %", 
                    line=dict(color='#4ecdc4', width=3),
                    yaxis="y2"
                ))
                fig.update_layout(
                    yaxis=dict(title="Temperature (°C)"),
                    yaxis2=dict(title="Humidity %", overlaying="y", side="right"),
                    xaxis_tickangle=-45,
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(
                    x=times,
                    y=temps,
                    name="Temperature",
                    line=dict(color='#ff6b6b', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.2)'
                ))
                fig_temp.update_layout(
                    yaxis=dict(title="Temperature (°C)"),
                    xaxis_tickangle=-45,
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with tab3:
                fig_hum = go.Figure()
                fig_hum.add_trace(go.Bar(
                    x=times,
                    y=humidity,
                    name="Humidity",
                    marker_color='#4ecdc4'
                ))
                fig_hum.update_layout(
                    yaxis=dict(title="Humidity %"),
                    xaxis_tickangle=-45,
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_hum, use_container_width=True)

# -------------------------
# Right: AI Assistant with RAG
# -------------------------
with col2:
    with st.container(border=True):
        st.subheader("🤖 AI Weather Assistant")
        st.caption("Uses RAG for predictions from historical data")
        
        query_mode = st.radio(
            "Query Mode:",
            ["🔮 RAG Forecast (Historical)", "🌐 Live Weather (API)"],
            label_visibility="collapsed"
        )
        
        ai_input = st.text_input(
            "Ask about weather:", 
            key="ai_input", 
            placeholder="e.g., Predict weather in Paris"
        )
        
        if st.button("🔍 Ask AI", key="ask_btn", use_container_width=True):
            if ai_input:
                with st.spinner("Processing..."):
                    if "RAG" in query_mode:
                        # Use RAG for prediction
                        if tokenizer and rag_model:
                            prediction, error = predict_weather_with_rag(
                                ai_input, weather_df, tokenizer, rag_model
                            )
                            
                            if prediction:
                                st.markdown(prediction)
                                log_query(ai_input, prediction, source="RAG")
                            else:
                                st.warning(error)
                                # Fallback to API
                                st.info("Falling back to live API...")
                                answer = run_agent(ai_input)
                                st.success(answer)
                        else:
                            st.error("RAG model not loaded. Using API fallback.")
                            answer = run_agent(ai_input)
                            st.success(answer)
                    else:
                        # Use API
                        answer = run_agent(ai_input)
                        st.success(answer)
            else:
                st.warning("Please enter a question.")
    
    with st.container(border=True):
        st.subheader("📊 Dataset Info")
        if weather_df is not None:
            st.metric("Records", len(weather_df))
            st.metric("Cities", weather_df['city'].nunique() if 'city' in weather_df.columns else 'N/A')
            
            with st.expander("View Sample Data"):
                st.dataframe(weather_df.head(10))
    
    with st.container(border=True):
        st.subheader("📋 Recent Logs")
        
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = f.read()
            
            with st.expander("View Logs", expanded=False):
                st.code(logs[-2000:], language=None)
            
            if st.button("🗑️ Clear Logs", use_container_width=True):
                os.remove(LOG_FILE)
                st.rerun()
        else:
            st.info("No logs yet.")