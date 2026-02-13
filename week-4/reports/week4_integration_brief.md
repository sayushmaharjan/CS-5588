# Week 4 Integration Brief — {cfg.project_title}

**Team:**   Sayush Maharjan, Harsha Sri
**Stakeholder/User:** Travellers, Everyday users, Urban planners,  Local government, Data analyst
**Problem:**   Most weather apps focus on short-term predictions like temperature, rain, alerts but don’t explain whether conditions are unusual, risky, or part of a larger trend. Users are left without context unsure if today’s heat is extreme, if recent rain signals a real shift, or how current conditions compare to what’s typical for that place and time of year.

This missing context leads to confusion, misjudged risk, and erosion of trust in weather information. WeatherTwin closes this gap by treating weather as an interpretable decision-support problem, not just a forecast. It grounds current conditions in historical climate patterns and transparently explains why they matter, helping users move from passively viewing forecasts to making informed, confident, climate-aware decisions.

---

## 1) Module placement in capstone system

### Upstream inputs:
- Live weather data from OpenWeather API
- Historical temperature and humidity CSV datasets (2021–2026)

### Module responsibilities:
- Fetch and process 24-hour live forecast data
- Compute historical monthly averages
- Compare live conditions to historical baselines
- Provide natural-language query responses
- Log all user interactions for evaluation

### Downstream outputs:
- Visual dashboard (temperature, humidity, wind graphs)
- Evidence-backed query responses
- Structured event logs for monitoring and evaluation

---

## 2) User workflow (end-to-end)
1. User selects a city from the dashboard.  
2. System retrieves live weather and 24-hour forecast data.  
3. Historical climate data is loaded and compared to current conditions.  
4. User submits a natural-language weather question.  
5. System generates an evidence-backed response and displays metrics (latency, confidence).  
6. Interaction is logged for monitoring and evaluation.  

---

## 3) Success metrics

### Product / impact metrics (required)
- **Time-to-decision:** Reduce weather analysis time from ~5 minutes (manual lookup) to under 30 seconds.  
- **Trust/verification signals:** Display historical comparisons and evidence sources alongside responses.  
- **Adoption/usage signal:** Number of completed queries per session.  

### Technical metrics (recommended)
- **Quality:** % of correct comparative responses vs historical baseline (validated on sample queries).  
- **Latency:** < 1.5 seconds per query.  
- **Failure rate:** < 5% API errors or empty responses.  

---

## 4) Failure & risk (what happens if wrong?)

### Likely failure:
- Weather API outage or incorrect forecast data.  
- Incorrect historical comparison due to data parsing errors.  

### Impact:
- Users may make poor travel or event-planning decisions.  
- Loss of trust in the system.  

### Mitigation:
- Detect API status codes and handle non-200 responses.  
- Display warning banner on API failure.  
- Fall back to cached recent forecast data.  
- Log `failure_flag=True` for monitoring.  

---

## 5) Next sprint plan

### Next feature:
- Multi-city comparison view  
- Severe weather alert detection  

### Data improvement:
- Expand historical dataset to 10+ years  
- Add precipitation and extreme-event tagging  

### Evaluation improvement:
- Automated accuracy testing against historical averages  
- Add user feedback scoring for response helpfulness  
