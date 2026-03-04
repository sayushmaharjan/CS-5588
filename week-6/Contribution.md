# Contribution.md — Week 6: AI Agent Integration

**Project:** WeatherTwin AI  
**Course:** Spring 2026 Capstone  
**Date:** March 3, 2026  

---

## Team Member 1: Harsha Sri Neeriganti

### Week 6 Contributions
- **Task 1 — Antigravity IDE Setup:**
  - Connected Google Antigravity IDE to the project repository
  - Used the IDE to analyze existing codebase and plan the agent module architecture
  - Authored `task1_antigravity_report.md` documenting IDE usage and benefits
- **Task 2 — Agent Tools Development:**
  - Developed `agent/tools.py` with 5 callable weather tools (`get_current_weather`, `get_weather_forecast`, `get_historical_analysis`, `predict_weather_bert`, `compare_cities`)
  - Built `agent/agent_runner.py` implementing a multi-step ReAct agent using Groq API (`llama-3.3-70b-versatile`)
- **Task 4 — Agent Integration:**
  - Integrated `AgentRunner` into `app/app_map.py` with "Live Weather" mode
  - Added reasoning trace display (🔍 Agent Trace) to the chat UI
  - Set up authentication system (`app_auth.py`) with user management
  - Fixed cross-platform file path issues for data loading

### Reflection
Implementing the ReAct agent pattern was the most valuable learning experience this week. Structuring the reasoning loop — where the LLM thinks, selects a tool, observes the result, and iterates — gave me a deep understanding of how AI agents work in practice. Using Antigravity IDE to analyze and refactor the codebase into a modular agent package made the process significantly faster and more reliable.

---

## Team Member 2: Sayush Maharjan

### Week 6 Contributions
- **Task 2 — Agent Tool Schemas:**
  - Created `agent/tool_schemas.py` with JSON schemas for all 5 tools
  - Defined parameter specifications and descriptions for LLM tool selection
- **Task 3 — Application Interface Update:**
  - Implemented WeatherAPI integration for the agent's real-time weather tools
  - Developed BERT model setup and historical data loading pipeline for agent tools
  - Configured environment variables and API key management for Groq and WeatherAPI
- **Task 5 — Evaluation & Documentation:**
  - Designed 3 evaluation scenarios (simple, medium, complex) in `task4_evaluation_report.md`
  - Updated `README.md` with Week 6 agent architecture, tools table, and setup instructions
  - Updated `requirements.txt` with agent-related dependencies
  - Curated and organized historical weather CSV datasets for the agent's analysis tools

### Reflection
This week showed how modular design enables powerful extensions. Converting existing project components into standardized agent tools — each returning a consistent `{status, data, source}` dictionary — made it possible for the agent runner to orchestrate them seamlessly. The evaluation scenarios revealed both the agent's strengths in multi-step reasoning and areas for improvement, such as better weather-specific classification models.
