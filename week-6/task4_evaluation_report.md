# Task 4 — Agent Evaluation Report

## Overview

This report evaluates the WeatherTwin AI Agent across three scenarios of increasing complexity. Each scenario tests the agent's ability to interpret user requests, select appropriate tools, execute multi-step workflows, and produce evidence-grounded explanations.

### Agent Configuration
- **LLM:** Llama 3.3 70B (via Groq API)
- **Agent Pattern:** ReAct (Reason + Act) with up to 7 steps
- **Tools Available:** `get_current_weather`, `get_weather_forecast`, `get_historical_analysis`, `predict_weather_bert`, `compare_cities`

---

## Scenario 1: Simple — Single Tool Call

**User Prompt:**
> "What's the weather in Los Angeles?"

**Expected Behavior:**
1. Agent calls `get_current_weather(city="Los Angeles")`
2. Returns current temperature, condition, humidity, wind, UV, and AQI

**Evaluation Criteria:**

| Criterion | Expected | Notes |
|-----------|----------|-------|
| Correct tool selected | `get_current_weather` | Only one tool needed |
| Number of steps | 2 (tool call + final answer) | Minimal reasoning |
| Data accuracy | Matches WeatherAPI live data | Grounded in API response |
| Response quality | Clear, natural language summary | Should not just dump JSON |

**Performance Notes:**
- The agent should identify "Los Angeles" as the city and call the correct tool on the first step.
- The final answer should be conversational, mentioning key metrics (temperature, condition, humidity).
- Expected latency: ~2 seconds (1 LLM call + 1 API call + 1 LLM call).

---

## Scenario 2: Medium — Multi-Step Reasoning

**User Prompt:**
> "Should I go hiking in San Francisco this weekend? What should I expect?"

**Expected Behavior:**
1. Agent calls `get_weather_forecast(city="San Francisco", hours=48)` to check upcoming conditions
2. Agent calls `get_historical_analysis(city="San Francisco")` for typical patterns
3. Synthesizes both data sources into a hiking recommendation

**Evaluation Criteria:**

| Criterion | Expected | Notes |
|-----------|----------|-------|
| Tools selected | `get_weather_forecast` + `get_historical_analysis` | Two tools needed |
| Number of steps | 3–4 (2 tool calls + synthesis + answer) | Multi-step reasoning |
| Context integration | Combines forecast + historical data | Should reference both sources |
| Actionable advice | Specific recommendations | Clothing, timing, risks |

**Performance Notes:**
- The agent should recognize this requires both forecast data and historical context.
- The recommendation should be specific: mention temperatures, rain chance, and what to wear/pack.
- If the agent only calls one tool, the answer quality is reduced but still acceptable.
- Expected latency: ~4 seconds (2–3 LLM calls + 2 API calls).

---

## Scenario 3: Complex — Multi-Tool Comparison with Reasoning

**User Prompt:**
> "Compare weather in Los Angeles and San Diego and recommend the better city for outdoor activities today."

**Expected Behavior:**
1. Agent calls `compare_cities(city1="Los Angeles", city2="San Diego")` for side-by-side weather
2. Optionally calls `get_weather_forecast` for either/both cities for upcoming hours
3. Synthesizes comparison into a recommendation with justification

**Evaluation Criteria:**

| Criterion | Expected | Notes |
|-----------|----------|-------|
| Tools selected | `compare_cities` (+ optionally `get_weather_forecast`) | 1–2 tools |
| Number of steps | 2–4 | Comparison + synthesis |
| Comparison quality | Side-by-side metrics | Temperature, UV, wind, condition |
| Recommendation | Clear winner with reasoning | "LA is better because..." |
| Evidence grounding | Cites specific data points | Not just opinions |

**Performance Notes:**
- The `compare_cities` tool returns structured comparison data; the agent should format this clearly.
- The recommendation should factor in multiple metrics (UV for sunburn risk, wind for comfort, etc.).
- A strong response will explain trade-offs, not just pick a winner.
- Expected latency: ~3 seconds (1–2 LLM calls + 2 API calls + 1 LLM call).

---

## Summary of Evaluation Results

| Scenario | Complexity | Expected Tools | Expected Steps | Key Challenge |
|----------|-----------|----------------|----------------|---------------|
| 1. Current weather | Simple | 1 | 2 | Correct tool selection |
| 2. Hiking recommendation | Medium | 2 | 3–4 | Multi-source synthesis |
| 3. City comparison | Complex | 1–2 | 2–4 | Comparative reasoning |

## Identified Limitations

1. **BERT tool is limited to 3 cities** — Los Angeles, San Diego, San Francisco (based on available CSV data). Queries for other cities fall back to the live API tools.

2. **No memory across sessions** — The agent starts fresh each time. It cannot reference previous conversations or build on earlier analyses.

3. **Single-turn tool calls** — The agent calls one tool per LLM step. A parallel tool-calling approach would reduce latency for multi-tool scenarios.

4. **LLM dependency** — Response quality depends on the Groq API and Llama 3.3 model availability. Rate limits or outages will degrade the experience.

5. **No user feedback loop** — The agent cannot ask clarifying questions mid-workflow. If a query is ambiguous, it makes its best guess rather than asking for clarification.

## Demo Video

> **Demo video link:** *(To be recorded and inserted here — 3–5 minute walkthrough showing all 3 scenarios in the Streamlit UI)*
