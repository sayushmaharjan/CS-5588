# app.py
import os
import json
import requests
import streamlit as st
from streamlit.components.v1 import html
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import plotly.graph_objects as go
# =========================
# LOAD ENV
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "REMOVEDlcSNJqwWbx9es7OKfT1zWGdyb3FYS9LlnX1iSFU71Wd6RNNG6uSM"
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY") or "2e5d4ea8dc9e429da37235056261202"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"
LOG_FILE = "chat_log.txt"

# =========================
# LOGGING FUNCTION
# =========================
def log_query(user_input: str, bot_response: str, weather_data: str = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"Timestamp   : {timestamp}\n")
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
    weather_info = None  # Track weather data for logging

    while step < max_steps:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0
            )
        except Exception as e:
            error_msg = f"❌ API Error: {str(e)}"
            log_query(user_input, error_msg)  # Log errors too
            return error_msg

        content = response.choices[0].message.content.strip()

        try:
            content_json = parse_agent_response(content)
            action = content_json.get("action", "")
            action_input = content_json.get("action_input", "")

            if action == "user_answer":
                # ✅ LOG THE QUERY AND RESPONSE
                log_query(user_input, action_input, weather_info)
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
            # Log raw response as answer
            log_query(user_input, content, weather_info)
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
st.title("🌦 Weather + AI Dashboard")

col1, col2 = st.columns([3,1])

# -------------------------
# Left: Weather Dashboard
# -------------------------
with col1:
    city_input = st.text_input("Enter a city:", value="New York")
    weather_data = fetch_weather(city_input)

    if "error" in weather_data:
        st.error(weather_data["error"])
    else:
        cur = parse_current(weather_data)
        st.subheader(f"{cur['city']}, {cur['region']}")
        st.caption(f"Local Time: {cur['localtime']}")
        # Flash card
        st.markdown(f"""
        **Condition:** {cur['condition']}  
        **Temp:** {cur['temp_c']}°C  
        **AQI:** {cur['aqi']}, **UV:** {cur['uv']}  
        **Sunrise/Sunset:** {cur['sunrise']} / {cur['sunset']}  
        **Humidity:** {cur['humidity']}%, **Wind:** {cur['wind_kph']} km/h  
        **Pressure:** {cur['pressure_mb']} mb, **Visibility:** {cur['vis_km']} km
        """)
        # 24h graph
        times, temps, humidity = get_24h_data(weather_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=temps, name="Temp (°C)", line=dict(color='firebrick')))
        fig.add_trace(go.Scatter(x=times, y=humidity, name="Humidity %", line=dict(color='royalblue'), yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="Temp (°C)"),
            yaxis2=dict(title="Humidity %", overlaying="y", side="right"),
            title="Next 24 Hours: Temp & Humidity",
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Right: AI Assistant
# -------------------------
with col2:
    st.subheader("🤖 AI Assistant")
    ai_input = st.text_input("Ask AI about weather:", key="ai_input")
    if st.button("Ask AI", key="ask_btn"):
        answer = run_agent(ai_input)
        st.success(answer)
        log_query(ai_input, answer)

    st.subheader("📋 Recent Logs")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = f.read()
        st.text_area("Logs", value=logs[-2000:], height=300)
        if st.button("Clear Logs"):
            os.remove(LOG_FILE)
            st.experimental_rerun()
    else:
        st.info("No logs yet.")

# # app.py
# import os
# import json
# import requests
# import streamlit as st
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# from openai import OpenAI
# import plotly.graph_objects as go
# # =========================
# # LOAD ENV
# # =========================
# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "REMOVEDlcSNJqwWbx9es7OKfT1zWGdyb3FYS9LlnX1iSFU71Wd6RNNG6uSM"
# WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY") or "2e5d4ea8dc9e429da37235056261202"

# client = OpenAI(
#     api_key=GROQ_API_KEY,
#     base_url="https://api.groq.com/openai/v1"
# )

# MODEL = "llama-3.3-70b-versatile"
# LOG_FILE = "chat_log.txt"

# # =========================
# # LOGGING FUNCTION
# # =========================
# def log_query(user_input: str, bot_response: str, weather_data: str = None):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write("="*60 + "\n")
#         f.write(f"Timestamp   : {timestamp}\n")
#         f.write(f"User Query  : {user_input}\n")
#         if weather_data:
#             f.write(f"Weather Data: {weather_data}\n")
#         f.write(f"Bot Response: {bot_response}\n")
#         f.write("="*60 + "\n\n")

# # =========================
# # WEATHER FUNCTIONS
# # =========================
# def fetch_weather(city: str):
#     url = "http://api.weatherapi.com/v1/forecast.json"
#     params = {"key": WEATHERAPI_KEY, "q": city, "days": 2, "aqi": "yes"}
#     try:
#         res = requests.get(url, params=params, timeout=10)
#         res.raise_for_status()
#         return res.json()
#     except Exception as e:
#         return {"error": str(e)}

# def parse_current(data):
#     cur = data["current"]
#     loc = data["location"]
#     return {
#         "city": loc["name"],
#         "region": loc["region"],
#         "localtime": loc["localtime"],
#         "temp_c": cur["temp_c"],
#         "condition": cur["condition"]["text"],
#         "humidity": cur["humidity"],
#         "wind_kph": cur["wind_kph"],
#         "pressure_mb": cur["pressure_mb"],
#         "vis_km": cur["vis_km"],
#         "uv": cur["uv"],
#         "aqi": cur.get("air_quality", {}).get("us-epa-index", "N/A"),
#         "sunrise": data["forecast"]["forecastday"][0]["astro"]["sunrise"],
#         "sunset": data["forecast"]["forecastday"][0]["astro"]["sunset"]
#     }

# def get_next24h(data):
#     """Return next 24 hours starting from current hour"""
#     all_hours = data["forecast"]["forecastday"][0]["hour"] + data["forecast"]["forecastday"][1]["hour"]
#     current_hour = datetime.strptime(data["location"]["localtime"], "%Y-%m-%d %H:%M").hour
#     next24 = all_hours[current_hour:current_hour+24]
#     times = [h["time"].split(" ")[1] for h in next24]
#     temps = [h["temp_c"] for h in next24]
#     humidity = [h["humidity"] for h in next24]
#     return times, temps, humidity

# # =========================
# # AI AGENT
# # =========================
# SYSTEM_PROMPT = """You are a helpful AI agent that can use tools to find weather information.
# Respond only in strict JSON, no markdown.
# Actions:
# 1. get_weather
# 2. user_answer
# Format:
# {"thought": "...", "action": "get_weather", "action_input": "city"}
# or
# {"thought": "...", "action": "user_answer", "action_input": "..."}
# """

# def run_agent(user_input: str) -> str:
#     messages = [{"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": user_input}]
#     try:
#         response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"API Error: {str(e)}"

# # =========================
# # STREAMLIT UI
# # =========================
# st.set_page_config(page_title="Weather + AI Dashboard", layout="wide")
# st.title("🌦 Weather + AI Dashboard")

# # from streamlit import experimental_rerun
# from streamlit_autorefresh import st_autorefresh

# # At the top of your script (after imports)
# # This will refresh the app every 5 minutes (300000 ms)
# count = st_autorefresh(interval=300_000, limit=None, keys="auto_refresh")


# # Auto-refresh every 5 minutes (300000 ms)
# # st_autorefresh = st.experimental_data_editor({"Refresh": [1]}, key="autorefresh", disabled=True)
# # st.experimental_rerun(interval=300000)  # Streamlit auto-refresh every 5 mins

# col1, col2 = st.columns([3,1])

# # -------------------------
# # Left: Weather Dashboard
# # -------------------------
# with col1:
#     city_input = st.text_input("Enter a city:", value="New York")
#     weather_data = fetch_weather(city_input)

#     if "error" in weather_data:
#         st.error(weather_data["error"])
#     else:
#         cur = parse_current(weather_data)
#         st.subheader(f"{cur['city']}, {cur['region']}")
#         st.caption(f"Local Time: {cur['localtime']}")
#         # Flash card
#         st.markdown(f"""
#         **Condition:** {cur['condition']}  
#         **Temp:** {cur['temp_c']}°C  
#         **AQI:** {cur['aqi']}, **UV:** {cur['uv']}  
#         **Sunrise/Sunset:** {cur['sunrise']} / {cur['sunset']}  
#         **Humidity:** {cur['humidity']}%, **Wind:** {cur['wind_kph']} km/h  
#         **Pressure:** {cur['pressure_mb']} mb, **Visibility:** {cur['vis_km']} km
#         """)
#         # 24h graph starting from now
#         times, temps, humidity = get_next24h(weather_data)
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=times, y=temps, name="Temp (°C)", line=dict(color='firebrick')))
#         fig.add_trace(go.Scatter(x=times, y=humidity, name="Humidity %", line=dict(color='royalblue'), yaxis="y2"))
#         fig.update_layout(
#             yaxis=dict(title="Temp (°C)"),
#             yaxis2=dict(title="Humidity %", overlaying="y", side="right"),
#             title="Next 24 Hours: Temp & Humidity",
#             xaxis_tickangle=-45,
#             height=400
#         )
#         st.plotly_chart(fig, use_container_width=True)

# # -------------------------
# # Right: AI Assistant
# # -------------------------
# with col2:
#     st.subheader("🤖 AI Assistant")
#     ai_input = st.text_input("Ask AI about weather:", key="ai_input")
#     if st.button("Ask AI", key="ask_btn"):
#         answer = run_agent(ai_input)
#         st.success(answer)
#         log_query(ai_input, answer)

#     st.subheader("📋 Recent Logs")
#     if os.path.exists(LOG_FILE):
#         with open(LOG_FILE, "r", encoding="utf-8") as f:
#             logs = f.read()
#         st.text_area("Logs", value=logs[-2000:], height=300)
#         if st.button("Clear Logs"):
#             os.remove(LOG_FILE)
#             st.experimental_rerun()
#     else:
#         st.info("No logs yet.")