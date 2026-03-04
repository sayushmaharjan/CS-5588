# WeatherTwin

## Team Name
**Team 2 — WeatherTwin**

## Team Members
- **Harsha Sri Neeriganti**  
  GitHub: Harsha Sri Neeriganti	https://github.com/Harsha-Sri24

- **Sayush Maharjan**  
  GitHub: Sayush Maharjan	https://github.com/sayushmaharjan


## Project Summary
WeatherTwin is a GenAI-powered climate intelligence assistant that goes beyond basic weather forecasts to provide **personalized, context-aware, and explainable weather insights**.  
It combines large language models (LLMs) with **Retrieval-Augmented Generation (RAG)** to translate complex weather and climate data into clear, actionable explanations tailored to a user’s location, time horizon, and specific needs.

By grounding responses in historical climate records and trusted sources, WeatherTwin helps users understand whether current conditions are typical, unusual, or risky compared to long-term patterns. The system emphasizes **trust, transparency, and uncertainty awareness**, enabling informed decision-making for everyday users and professionals.

---

## Problem Statement & Impact
Most weather applications focus on short-term predictions (temperature, rain, alerts) without explaining **context** — whether conditions are unusual, risky, or part of a larger trend.  
This lack of explanation leads to confusion, misjudged risk, and reduced trust.

WeatherTwin addresses this gap by treating weather as a **decision-support problem**, grounding current conditions in historical climate patterns and clearly explaining why they matter.

---

## Target Users & Stakeholders
- **Everyday Users & Travelers**  
  Planning daily activities, travel, or outdoor events with better context.
- **Urban Planners & Local Governments**  
  Assessing risks, infrastructure planning, and climate-informed decisions.
- **Data Analysts & Educators**  
  Exploring climate trends with explainable and reproducible outputs.

### Stakeholder Needs
- Contextualized and reliable weather explanations  
- Evidence-backed and citation-based reasoning  
- Integration of historical data with current conditions

---

## Decisions & Workflows Improved
- **Daily Planning:** Understand if weather is unusual for a given place and time  
- **Risk Awareness:** Compare current conditions with historical patterns  
- **Trend Detection:** Identify extreme or emerging weather patterns early  
- **Trust & Verification:** Transparent sourcing and human feedback loops  

---

## Related Work
1. **Retrieval-Augmented Generation with Grounded Attribution**  
   Gao et al., 2023 — https://arxiv.org/abs/2305.14627  
   Informs citation-backed RAG design and grounding evaluation.

2. **ClimaX: A Foundation Model for Weather and Climate**  
   Nguyen et al., ICML 2023 — https://arxiv.org/abs/2301.10343  
   Inspires representation of multi-variable climate data.

3. **AutoAgent: Multi-Agent Framework for Complex Reasoning**  
   Li et al., 2023 — https://arxiv.org/abs/2309.17288  
   Motivates agent-based orchestration and verification.

---

## Data Sources
- **NOAA Global Historical Climate Network (GHCN)**  
  https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- **Kaggle Weather & Climate Datasets**  
  https://www.kaggle.com/datasets
- **IPCC Assessment Reports (AR5 / AR6)**  
  https://www.ipcc.ch/reports/
- **Hugging Face Climate & Weather Datasets**  
  https://huggingface.co/datasets

These datasets support grounding, retrieval, fine-tuning, and evaluation.

---

## GenAI System Architecture & Pipeline
1. **Data Ingestion & Knowledge Layer**
   - Chunking of documents and tabular data
   - Metadata tagging (location, date, variable)
   - Versioning and provenance tracking

2. **Retrieval, Generation & Fine-Tuning**
   - Embeddings and vector search
   - RAG pipeline with reranking and citations
   - Optional LoRA / PEFT fine-tuning
   - Multi-agent orchestration (retrieval, reasoning, verification)

3. **Interface & Delivery**
   - Flask API for model access
   - Dashboard or chat-based interface
   - Human-in-the-loop feedback
   - Transparent citations and uncertainty reporting

---

## Methods, Technologies & Tools
- **LLMs:** OpenAI GPT-4 / Hugging Face models  
- **RAG Frameworks:** LangChain, LlamaIndex  
- **Vector Search:** FAISS (local)  
- **Fine-Tuning:** LoRA / PEFT  
- **Interface & API:** Flask, Streamlit  
- **Collaboration:** GitHub, documentation & version control  

---

## Evaluation & Trust Metrics
- **Grounding & Hallucination Rate**
- **Citation Accuracy**
- **Task Success Rate**
- **User Trust & Satisfaction**
- **Latency & Cost**

Baselines include LLM-only and RAG-without-fine-tuning systems.

---

## Expected Deliverables
- Functional GenAI prototype
- Interactive demo (dashboard or chat)
- Fully versioned GitHub repository
- Final technical report with evaluation results

---

## Phase 1: Project Initiation (Weeks 1–3)
- Define GenAI workflow and system scope
- Collect and curate datasets
- Design RAG and fine-tuning strategy
- Build initial ingestion and retrieval pipeline

---

## Members, Roles & Project Tasks

### Harsha Sri Neeriganti
**Roles:** Backend, AI agents, data pipelines, dashboard integration  
**Tasks:**
- Build RAG pipeline and data ingestion workflows  
- Implement AI agents for personalized responses  
- Develop Flask API  
- Integrate backend with dashboard  
- Assist in evaluation and final reporting  

### Sayush Maharjan
**Roles:** Frontend, dashboard interface, AI agent integration  
**Tasks:**
- Design and implement frontend interface  
- Integrate Flask API for user interaction  
- Connect AI agents to frontend  
- Support evaluation and final documentation  

---

## Week 6 — AI Agent Integration

### Agent Architecture

WeatherTwin now includes a modular AI agent layer that transforms the dashboard into an intelligent decision-support system. The agent follows the **ReAct (Reason + Act)** pattern:

1. **User Query** → Interpreted by Llama 3.3 70B (via Groq API)
2. **Tool Selection** → Agent picks from 5 available tools based on JSON schemas
3. **Execution** → Tool functions call WeatherAPI, parse CSV data, or run BERT inference
4. **Observation** → Results feed back to the LLM for further reasoning or final answer
5. **Response** → Grounded answer with visible reasoning trace in the UI

### Agent Tools

| Tool | Description | Data Source |
|------|-------------|-------------|
| `get_current_weather` | Real-time weather conditions for any city | WeatherAPI (live) |
| `get_weather_forecast` | Hourly forecast for the next N hours | WeatherAPI (forecast) |
| `get_historical_analysis` | Statistical summary from historical data | CSV dataset |
| `predict_weather_bert` | BERT-based weather condition prediction | BERT model + CSV |
| `compare_cities` | Side-by-side comparison of two cities | WeatherAPI (comparison) |

### Project Structure

```
week-6_WeatherTwin_Antigravity_Integration/
├── agent/                       # NEW — Agent module
│   ├── __init__.py
│   ├── tools.py                 # 5 callable tool functions
│   ├── tool_schemas.py          # JSON schemas for LLM tool selection
│   └── agent_runner.py          # ReAct agent loop
├── app/
│   ├── app_map.py               # Main Streamlit dashboard (updated)
│   ├── app_auth.py              # Authentication module
│   └── app_bert.py              # BERT integration (reference)
├── auth/
│   └── users.csv                # User credentials
├── data/
│   ├── los_angeles.csv          # Historical weather data
│   ├── san_diego.csv
│   └── san_francisco.csv
├── task1_antigravity_report.md  # Antigravity IDE usage report
├── task4_evaluation_report.md   # Evaluation scenarios & analysis
├── requirements.txt
└── README.md
```

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd week-6_WeatherTwin_Antigravity_Integration
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install folium streamlit-folium
   ```

3. **Create a `.env` file** in the project root with the following keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   WEATHERAPI_KEY=your_weatherapi_key
   OPENWEATHER_API_KEY=your_openweathermap_key
   ```

4. **Run the application:**
   ```bash
   cd app
   streamlit run app_map.py
   ```

5. **Using the agent:**
   - Select **Live Weather** mode in the chat panel to use the full agent with all 5 tools
   - Select **BERT Forecast** mode for direct BERT predictions (falls back to agent if BERT can't handle the query)
   - Click the **🔍 Agent Trace** expander under any AI response to see the reasoning steps

### Demo Video

> https://drive.google.com/file/d/1ZIxZJZBqhN9NKkUsSSOrXI8CdceK8EMX/view?usp=sharing

