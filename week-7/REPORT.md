# Week 7 — Open-Source Climate Project Reproducibility Report

**Course:** CS 5588 — DataScience Capstone 
**Team:** Team 2 — WeatherTwin  
**Members:** Harsha Sri Neeriganti, Sayush Maharjan  
**Date:** March 10, 2026

---

## 1. Executive Summary

This report documents the reproducibility of two open-source climate-related projects and analyzes their relevance to our WeatherTwin project. Both projects were cloned, set up from source, and verified on a local macOS ARM64 system.

| Project | Domain | Status |
|---------|--------|--------|
| **CLIMADA** (CLIMate ADAptation) | Climate risk assessment framework | ✅ Fully reproduced |
| **Personalized Climate-Aware Health Navigator** | AI-powered weather-health advisory | ✅ Fully reproduced (with fixes) |

---

## 2. Project 1 — CLIMADA (CLIMate ADAptation)

### 2.1 Overview

[CLIMADA](https://github.com/CLIMADA-project/climada_python) is an open-source probabilistic natural catastrophe impact model developed by ETH Zurich. It provides global coverage of major climate-related extreme-weather hazards at high resolution (4×4 km) and supports climate risk assessment and adaptation option appraisal.

**Key capabilities:**
- Probabilistic impact calculations for natural hazards (tropical cyclones, floods, etc.)
- Exposure and vulnerability modeling (LitPop asset exposure)
- Cost-benefit analysis for adaptation measures
- Uncertainty quantification and forecasting
- Data API for hazard, exposure, and impact function datasets

### 2.2 Reproducibility Steps

| Step | Command | Outcome |
|------|---------|---------|
| 1. Clone | `git clone https://github.com/CLIMADA-project/climada_python.git` | ✅ Cloned (195 MB) |
| 2. Checkout branch | `git checkout develop` | ✅ `develop` branch |
| 3. Create conda env | `conda create -n climada_env "python=3.11.*" -y` | ✅ Python 3.11.14 |
| 4. Install deps | `conda env update -n climada_env -f requirements/env_climada.yml` | ✅ All dependencies |
| 5. Install package | `conda run -n climada_env python -m pip install -e ./` | ✅ v6.1.1.dev0 |
| 6. Verify | `python -m unittest climada.engine.test.test_impact` | ✅ 44/44 tests passed |

**Reproducibility Rating: (Excellent)**

- Well-documented installation guide with both simple and advanced paths
- Conda environment file (`env_climada.yml`) pins all dependencies precisely
- Built-in verification test suite
- Supports Python 3.10, 3.11, 3.12

### 2.3 Challenges Encountered

- The official docs recommend `mamba` for faster dependency solving, but `conda` worked without issues for our setup.
- First-time execution generates a directory tree in the home directory, which takes a few extra minutes.
- No significant issues were encountered during reproduction.

---

## 3. Project 2 — Personalized Climate-Aware Health Navigator

### 3.1 Overview

[Personalized Climate-Aware Health Navigator](https://github.com/VigneshwaranKbgv/Personalized-Climate-Aware-Health-Navigator) is a web-based AI-powered platform that provides real-time weather-based health and activity recommendations. It integrates OpenWeather API data with Google Gemini AI to deliver personalized health suggestions.

**Key capabilities:**
- User registration and authentication
- Real-time weather data integration via OpenWeather API
- AI-generated health and activity suggestions (originally Gemini, now Groq)
- Interactive dashboard with weather reports and suggestions
- MVC architecture with Java/JSP, MySQL, and Apache Tomcat

### 3.2 Reproducibility Steps

| Step | Command | Outcome |
|------|---------|---------|
| 1. Clone | `git clone https://github.com/VigneshwaranKbgv/Personalized-Climate-Aware-Health-Navigator.git` | ✅ Cloned |
| 2. Create conda env | `conda create -n health_nav_env -c conda-forge openjdk=17 maven mysql -y` | ✅ JDK 17, Maven 3.9.13 |
| 3. Build WAR | `conda run -n health_nav_env mvn clean package` | ✅ `climate-health.war` (12 MB) |
| 4. Create Docker Compose | MySQL 8 + Tomcat 10.1 containers | ✅ Created |
| 5. Create DB schema | `init.sql` with `users` and `health_data` tables | ✅ Created |
| 6. Docker Compose up | `docker compose up --build -d` | ✅ Both containers healthy |
| 7. Verify | `curl http://localhost:8080/` → HTTP 200 | ✅ App running |

**Reproducibility Rating: (Moderate)**

### 3.3 Challenges Encountered & Fixes Applied

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| No setup instructions in README | README only documents architecture, not how to run locally | Created `docker-compose.yml`, `init.sql`, and `.env` files |
| URL encoding bug | `WeatherAPI.java` didn't URL-encode locations with spaces (e.g., "New York") | Added `URLEncoder.encode()` |
| Deprecated Gemini model | Used `gemini-pro` which is deprecated | Updated to `gemini-2.0-flash` |
| Gemini API rate limiting | HTTP 429 errors on free-tier API key | Replaced Gemini with Groq API (llama-3.3-70b-versatile) |
| No database schema | No SQL init scripts provided | Reverse-engineered from DAO source code |
| Weather card hardcoded | `dashboard.jsp` shows static weather values | Identified as original design limitation |

---

## 4. Relation to WeatherTwin

### 4.1 Feature Comparison

| Feature | CLIMADA | Health Navigator | WeatherTwin |
|---------|---------|-----------------|-------------|
| **Weather data** | Historical hazard datasets | Real-time via OpenWeather API | Real-time + historical (NOAA GHCN) |
| **AI/ML** | Probabilistic impact models | LLM-powered suggestions (Groq/Gemini) | RAG + LLM with citations |
| **Personalization** | Regional risk profiles | Health-based (weight, height, temp) | Location + context-aware |
| **Data grounding** | Peer-reviewed climate data | Live API data only | RAG with source attribution |
| **Architecture** | Python library | Java MVC + Docker | Flask API + Streamlit/Dashboard |
| **Target users** | Researchers, policymakers | General public | Everyday users, planners, analysts |

### 4.2 Key Insights for WeatherTwin

**From CLIMADA:**
- **Hazard modeling patterns** — CLIMADA's probabilistic impact calculation framework could inform how WeatherTwin quantifies risk and uncertainty in weather explanations.
- **Data API design** — CLIMADA's data API for accessing hazard, exposure, and impact datasets demonstrates a clean pattern for integrating diverse climate data sources into a unified retrieval layer.
- **Uncertainty quantification** — CLIMADA's `unsequa` module for uncertainty and sensitivity analysis aligns directly with WeatherTwin's goal of "uncertainty awareness."

**From Health Navigator:**
- **LLM integration with weather data** — The pattern of fetching real-time weather data and passing it as context to an LLM for generating personalized recommendations is directly applicable to WeatherTwin's RAG pipeline.
- **User personalization** — Health Navigator's approach of combining user-specific data (location, health metrics) with weather conditions to generate tailored advice is similar to WeatherTwin's "context-aware" goal.
- **API key management** — Using environment variables for API keys in a Docker-based deployment is a clean pattern for WeatherTwin's Flask API deployment.

---

## 5. Integration Strategy for WeatherTwin

### 5.1 CLIMADA Integration

CLIMADA can enhance WeatherTwin's knowledge base by providing climate risk context.

**Integration approach:**
```
WeatherTwin RAG Pipeline
    ├── Current Weather Data (OpenWeather / NOAA)
    ├── Historical Climate Records (GHCN, Kaggle)
    └── Climate Risk Data (CLIMADA)  ← NEW
         ├── Hazard exposure for user's location
         ├── Historical impact data for similar events
         └── Risk percentiles and return periods
```

**Implementation steps:**
1. Use CLIMADA's Python API to precompute hazard exposure for major cities
2. Store results as structured documents in the RAG vector store (FAISS)
3. During retrieval, include climate risk context alongside historical weather data
4. LLM can reference CLIMADA data for risk-aware explanations (e.g., "This storm intensity has a 10-year return period for your area")

### 5.2 Health Navigator Pattern Integration

The Health Navigator's LLM-weather integration pattern can be adapted for WeatherTwin.

**Integration approach:**
```
User Query → Flask API
    ├── Retrieve weather data (current + historical)
    ├── Retrieve relevant documents (RAG)
    ├── Construct context-rich prompt
    └── Generate response with Groq/GPT-4
         ├── Weather explanation with historical context
         ├── Risk assessment with CLIMADA data
         └── Personalized recommendations
```

**Key adaptations from Health Navigator:**
1. Replace simple "weather → LLM → suggestion" pipeline with RAG-enhanced retrieval
2. Add historical grounding (NOAA GHCN data) for contextual explanations
3. Add citation tracking for transparency
4. Use Groq API (proven fast and reliable) as primary LLM, with GPT-4 as fallback

---

## 6. Individual Contributions

### Harsha Sri Neeriganti

| Task | Description | Status |
|------|-------------|--------|
| CLIMADA setup & verification | Cloned repository, created conda environment, installed dependencies, ran verification tests (44/44 passed) | ✅ Complete |
| CLIMADA architecture analysis | Analyzed CLIMADA's modular architecture (hazard, exposure, impact, adaptation layers) and documented integration opportunities with WeatherTwin's RAG pipeline | ✅ Complete |
| Integration strategy — CLIMADA | Designed the approach for incorporating CLIMADA's probabilistic risk data into WeatherTwin's knowledge base via FAISS vector store | ✅ Complete |
| Backend integration research | Identified how CLIMADA's Python API and data modules can be consumed by WeatherTwin's Flask backend for risk-aware weather explanations | ✅ Complete |

### Sayush Maharjan

| Task | Description | Status |
|------|-------------|--------|
| Health Navigator setup & verification | Cloned repository, created Docker Compose environment (MySQL + Tomcat), built WAR file, verified app at localhost:8080 | ✅ Complete |
| Bug fixes & Groq API integration | Fixed URL encoding bug in `WeatherAPI.java`, updated deprecated Gemini model, integrated Groq API (`llama-3.3-70b-versatile`) as replacement | ✅ Complete |
| Integration strategy — Health Navigator | Analyzed the LLM-weather integration pattern and documented how WeatherTwin can adapt the personalized suggestion pipeline with RAG enhancement | ✅ Complete |
| Frontend integration research | Identified how Health Navigator's dashboard pattern (real-time weather + AI suggestions) can inform WeatherTwin's Streamlit/dashboard interface design | ✅ Complete |

---

## 7. Summary & Lessons Learned

### Key Takeaways

1. **CLIMADA** is a mature, well-documented framework with excellent reproducibility. Its probabilistic hazard modeling and uncertainty quantification are directly relevant to WeatherTwin's goals of providing trustworthy, evidence-backed climate insights.

2. **Health Navigator** demonstrates a practical LLM-weather integration pattern but lacks reproducibility documentation. Several bugs were encountered and fixed during setup, highlighting the importance of comprehensive setup guides and automated testing.

3. **Both projects validate WeatherTwin's approach** — combining real-time weather data with AI/ML for personalized, context-aware insights. WeatherTwin's differentiator is the RAG-based grounding in historical data with citation tracking, which neither project fully implements.

### Reproducibility Best Practices Identified

- Provide `docker-compose.yml` or equivalent for complex multi-service setups
- Pin all dependency versions in environment files
- Include automated verification tests
- Document required API keys and environment variables
- Provide sample `.env` files with clear instructions

---

## 8. References

1. CLIMADA Documentation — https://climada-python.readthedocs.io/
2. CLIMADA GitHub — https://github.com/CLIMADA-project/climada_python
3. Health Navigator GitHub — https://github.com/VigneshwaranKbgv/Personalized-Climate-Aware-Health-Navigator