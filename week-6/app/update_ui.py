import re
import sys

with open('app_map.py', 'r') as f:
    content = f.read()

# 1. Global CSS Injection
target_1 = 'st.set_page_config(page_title="Weather + AI Dashboard", layout="wide")\nst.title("🌦 Weather + AI Dashboard with Interactive Map")'
replacement_1 = '''st.set_page_config(page_title="Weather + AI Dashboard", layout="wide")

st.markdown("""
<style>
    /* Typographic Contrast */
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        opacity: 0.65 !important;
        font-weight: 600 !important;
    }
    /* Standardize Padding inside bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌦 Weather + AI Dashboard with Interactive Map")'''
content = content.replace(target_1, replacement_1)


# 2. Dataset Info Header Bar
target_2 = '''# Load BERT model and dataset
with st.spinner("🔄 Loading AI model and weather dataset..."):
    classifier = load_weather_model() #, tokenizer, model
    weather_df = load_weather_dataset()

col1, col2 = st.columns([3, 2])'''
replacement_2 = '''# Load BERT model and dataset
with st.spinner("🔄 Loading AI model and weather dataset..."):
    classifier = load_weather_model() #, tokenizer, model
    weather_df = load_weather_dataset()

# =========================
# HEADER: DATASET INFO & CONTROLS
# =========================
if weather_df is not None:
    with st.expander("📊 Dataset Info & Global Controls", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📝 Records", len(weather_df))
        with col_b:
            st.metric("🏙️ Cities Covered", weather_df['city'].nunique() if 'city' in weather_df.columns else 'N/A')
        with col_c:
            st.caption("Expand to view sample dataset rows below")
        st.dataframe(weather_df.head(5), use_container_width=True)

col1, col2 = st.columns([3, 2])'''

content = content.replace(target_2, replacement_2)


# 3. Unit Toggle and Condition Image
target_3 = '''        with st.container(border=True):
            col_title, col_source = st.columns([2, 1])
            with col_title:
                st.subheader(f"📍 {cur['city']}, {cur['region']}, {cur['country']}")
            with col_source:
                st.caption(source_label)
            
            st.caption(f"🕒 Local Time: {cur['localtime']}")
            st.caption(f"📌 Coordinates: {cur['lat']:.4f}, {cur['lon']:.4f}")
            
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
            
            st.divider()'''

replacement_3 = '''        with st.container(border=True):
            col_title, col_toggle = st.columns([3, 1])
            with col_title:
                st.subheader(f"📍 {cur['city']}, {cur['region']}, {cur['country']}")
                st.caption(f"{source_label} | 🕒 Local Time: {cur['localtime']} | 📌 Coordinates: {cur['lat']:.4f}, {cur['lon']:.4f}")
            with col_toggle:
                st.write("") # Spacing
                use_imperial = st.toggle("°F / mph", key="unit_toggle")
            
            # Calculate display values based on toggle
            if use_imperial:
                temp_disp = f"{(cur['temp_c'] * 9/5) + 32:.1f} °F"
                wind_disp = f"{cur['wind_kph'] * 0.621371:.1f} mph"
            else:
                temp_disp = f"{cur['temp_c']} °C"
                wind_disp = f"{cur['wind_kph']} km/h"
            
            # Main weather display
            main_col1, main_col2 = st.columns([1, 2])
            
            with main_col1:
                st.metric(
                    label="🌡️ Temperature",
                    value=temp_disp,
                    delta=cur['condition']
                )
            
            with main_col2:
                cond_l = cur['condition'].lower()
                if "sun" in cond_l or "clear" in cond_l:
                    bg = "linear-gradient(135deg, #f6d365 0%, #fda085 100%)"
                    ic = "☀️"
                elif "rain" in cond_l or "drizzle" in cond_l:
                    bg = "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)"
                    ic = "🌧️"
                elif "cloud" in cond_l or "overcast" in cond_l:
                    bg = "linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%)"
                    ic = "☁️"
                elif "snow" in cond_l:
                    bg = "linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)"
                    ic = "❄️"
                elif "storm" in cond_l or "thunder" in cond_l:
                    bg = "linear-gradient(135deg, #4b6cb7 0%, #182848 100%)"
                    ic = "⛈️"
                else:
                    bg = "linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%)"
                    ic = "🌡️"
                
                st.markdown(f\'\'\'
                <div style="background: {bg}; border-radius: 10px; padding: 1.5rem; text-align: center; color: white; text-shadow: 1px 1px 3px rgba(0,0,0,0.3);">
                    <h2 style="margin:0; font-size: 3rem;">{ic}</h2>
                    <h3 style="margin:0; font-weight: 700; letter-spacing: 1px;">{cur['condition']}</h3>
                </div>
                \'\'\', unsafe_allow_html=True)
            
            st.divider()'''

content = content.replace(target_3, replacement_3)


# 4. Progress Bars in Metrics
target_4 = '''            # Metrics Row 1
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
                st.metric(label="🌇 Sunset", value=cur['sunset'])'''

replacement_4 = '''            # Metrics Row 1
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric(label="💧 Humidity", value=f"{cur['humidity']}%")
                st.progress(min(cur['humidity'] / 100.0, 1.0))
            
            with m2:
                st.metric(label="💨 Wind", value=wind_disp) # Uses toggled unit
                st.progress(min(cur['wind_kph'] / 100.0, 1.0))
            
            with m3:
                st.metric(label="🌡️ Pressure", value=f"{cur['pressure_mb']} mb")
                p_val = max(0.0, min((cur['pressure_mb'] - 900) / 150.0, 1.0))
                st.progress(p_val)
            
            with m4:
                st.metric(label="👁️ Visibility", value=f"{cur['vis_km']} km")
                st.progress(min(cur['vis_km'] / 20.0, 1.0))
            
            # Metrics Row 2
            m5, m6, m7, m8 = st.columns(4)
            
            with m5:
                st.metric(label="☀️ UV Index", value=cur['uv'])
                uv_val = float(cur['uv']) if str(cur['uv']).replace('.','',1).isdigit() else 0
                st.progress(min(uv_val / 11.0, 1.0))
            
            with m6:
                aqi_val = cur.get('aqi', 0)
                if aqi_val == "N/A": aqi_val = 0
                st.metric(label="🌫️ AQI (US EPA)", value=aqi_val)
                st.progress(min(float(aqi_val) / 6.0, 1.0))
            
            with m7:
                st.metric(label="🌅 Sunrise", value=cur['sunrise'])
            
            with m8:
                st.metric(label="🌇 Sunset", value=cur['sunset'])'''

content = content.replace(target_4, replacement_4)


# 5. Remove Dataset Info from Right Sidebar
target_5 = '''    with st.container(border=True):
        st.subheader("📊 Dataset Info")
        if weather_df is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📝 Records", len(weather_df))
            with col_b:
                st.metric("🏙️ Cities", weather_df['city'].nunique() if 'city' in weather_df.columns else 'N/A')
            
            with st.expander("🔍 View Sample Data"):
                st.dataframe(weather_df.head(10), use_container_width=True)'''

content = content.replace(target_5, '')

# 6. Extract 12-Hour Forecast to Global Scope
# We'll locate the section between SECTION 2: PLOTLY ICON TIMELINE and right before the # Right: AI Assistant with BERT
# We will remove it from there, unindent it, and append it at the end of the file before Footer
import re

pattern = r"(        # =============================\n        # SECTION 2: PLOTLY ICON TIMELINE.*?)(?=        # -------------------------\n        # Right: AI Assistant with BERT)"
match = re.search(pattern, content, re.DOTALL)
if match:
    forecast_block = match.group(1)
    content = content.replace(forecast_block, '')
    
    # We also need to get the "weather_data" logic that is in forecast_block, but wait, weather_data is fetched globally per query
    # The whole forecast_block requires weather_data. Since we moved it out of `if not "error" in weather_data:` we need to re-wrap it below the columns
    
    # Let's clean the indentation of forecast block from 8 spaces to 4 spaces, so it can go inside an `if` block at the bottom
    lines = forecast_block.split('\n')
    unindented_lines = []
    for line in lines:
        if line.startswith('    '):  # If it has 4 spaces, remove 4 spaces to keep it at 1 level of indent (inside global if)
             unindented_lines.append(line[4:])
        else:
             unindented_lines.append(line)
             
    clean_forecast_block = '\n'.join(unindented_lines)
    
    # Now, append it after the right sidebar but before the footer
    footer_target = "# Footer\nst.divider()"
    
    new_forecast_wrapper = f'''# =============================
# GLOBAL 12-HOUR FORECAST
# =============================
if st.session_state.get('selected_lat') or st.session_state.get('map_city'):
    if 'weather_data' in locals() and "error" not in weather_data:
{clean_forecast_block}

# Footer
st.divider()'''

    content = content.replace(footer_target, new_forecast_wrapper)

with open('app_map.py', 'w') as f:
    f.write(content)

print("UI Updates applied successfully.")
