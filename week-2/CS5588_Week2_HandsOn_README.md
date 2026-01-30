# CS 5588 — Week 2 Hands-On
## Applied RAG for Product & Venture Development

> **Course:** CS 5588 — Data Science Capstone / GenAI Product Leadership  
> **Student: Sayush Maharjan**  
> **Project / Product Name: AI-Powered Weather & Climate Intelligence System**  
> **GitHub:**  
> **Date: 01/29/2026**  

---

## Product Overview
- **Product name:** AI-Powered Weather & Climate Intelligence System for Personalized Decision Support  
- **Target users:** Daily commuters, travelers planning trips, and students/researchers interested in weather and climate trends.  
- **Core problem:** Most weather apps show raw forecasts and simple alerts, but users still have to interpret what that means for concrete decisions like what to wear, whether to bike or drive, or whether a trip is exposed to severe weather. There is very little personalization, historical context, or explanation of risk levels.  
- **Why RAG:** RAG lets the system ground answers in up‑to‑date local forecasts, station summaries, historical/climate information, and safety guidance, instead of relying on an LLM’s static, potentially outdated world knowledge. It allows the assistant to answer with evidence (citations) and abstain when data is missing, which is crucial when users might act on the advice.

---

## Dataset Reality
- **Source / owner:** In a real deployment, the primary sources would be national meteorological agencies (e.g., NOAA/NWS, Environment Canada, WMO members) plus internally authored guidance/explainer documents. In this lab, I approximated this with synthetic station-based weather summaries and a general weather guidance file (`weather.txt`, `Station_1_weather.txt` … `Station_10_weather.txt`).  
- **Sensitivity:** Mostly public or internal reference data (forecasts, climate normals, safety rules). User-specific preferences (saved locations, commute routes) would be treated as internal and potentially sensitive.  
- **Document types:** Text summaries for weather stations (current/typical conditions, hazards), a general “how to read the forecast / what to wear / commute tips” document, and climate/context explanations.  
- **Expected scale in production:** Hundreds to low thousands of documents (multiple docs per region, per hazard type, and per product feature), growing over time as more regions and guidance are added.

---

## User Stories + Rubric

- **U1 (Normal):**  
  *User story:* As a daily commuter, I want a quick, personalized summary of today’s weather and what it means for my clothing and commute so that I can plan my day without being surprised by rain, heat, or cold.  
  *Acceptable evidence:*  
  - Station/weather summaries for the user’s city that describe today’s conditions (temperature, precipitation, wind).  
  - Guidance text that explains how to map those conditions to clothing and commute choices (e.g., rain → umbrella, slippery roads, extreme cold → extra layers).  
  *Correct answer must include:*  
  - A concise description of key conditions that matter for commuting (temperature range, rain/snow, wind, visibility).  
  - Clear, concrete suggestions for clothing and commute mode (e.g., bring a light jacket, consider leaving earlier if heavy rain is expected).

- **U2 (High-stakes):**  
  *User story:* As a traveler, I want to know whether my destination is at risk of severe weather on my travel dates so that I can decide whether to adjust my plans or take extra precautions.  
  *Acceptable evidence:*  
  - Station summaries or documents describing severe or unusual conditions for that region and date range (storms, heavy rain, extreme heat, etc.).  
  - Any hazard/safety guidance describing what “severe” means and what to do when certain thresholds are reached.  
  *Correct answer must include:*  
  - An explicit statement of any severe or potentially disruptive conditions indicated by the evidence, plus a clear statement if no such evidence is found.  
  - Safety-oriented recommendations (e.g., monitor official alerts, consider backup plans) and a reminder to verify with an official forecast source, rather than overconfident reassurance.

- **U3 (Ambiguous / failure-prone):**  
  *User story:* As a curious user, I want to ask broad questions like “Will climate change ruin summers in my city?” so that I can understand long‑term climate risks without being misled by individual events.  
  *Acceptable evidence:*  
  - Documents explaining the difference between weather and climate, long‑term trends, and how extremes are changing over decades.  
  - Any high-level regional climate-impact summaries or trend descriptions (e.g., more frequent heat waves, changing rainfall patterns).  
  *Correct answer must include:*  
  - A clear explanation that single summers or events cannot be “guaranteed ruined,” and that climate change shifts probabilities and typical conditions over time.  
  - Uncertainty-aware language and avoidance of overconfident predictions; answer should frame risk in terms of trends and adaptation rather than absolute doom.

---

## System Architecture

- **Chunking:** Semantic paragraph-based chunking with a maximum of ~1000 characters per chunk (keeping paragraphs together so each chunk is coherent enough to stand alone for explanation and safety advice).  
- **Keyword retrieval:** BM25 over lowercased tokenized text for each chunk, used to catch exact phrases such as specific station names, dates, or hazard keywords like “thunderstorm”, “flood”, “heat advisory”.  
- **Vector retrieval:** Sentence-transformers `all-MiniLM-L6-v2` to embed chunks and queries, indexed with FAISS (inner product) for semantic similarity, to catch paraphrases and more natural language questions.  
- **Hybrid α:** Hybrid fusion with α ≈ 0.5 (balanced between keyword and vector scores), to serve both precision-first (safety) and discovery/learning users without over-weighting one signal.  
- **Reranking governance:** Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` used as a governance layer to re-rank the top hybrid candidates, pushing the most truly relevant, safety-/context-critical chunks to the top.  
- **LLM / generation option:** Lightweight generation with `google/flan-t5-base` plugged into a RAG prompt that enforces use of provided evidence and allows abstention; with a fallback evidence-summary mode when generation is disabled.

---

## Results

(Values below reflect manual labeling based on my rubric; you can adjust if your own labeling differs.)

| User Story | Method           | Precision@5 | Recall@10 | Trust (1–5) | Confidence (1–5) |
|-----------|------------------|------------:|----------:|------------:|-----------------:|
| U1_normal | Hybrid + Rerank  | 1.00        | 1.00      | 4           | 4                |
| U2_high_stakes | Hybrid + Rerank  | 0.80        | 0.80      | 3           | 3                |
| U3_ambiguous_failure | Hybrid + Rerank  | 0.60        | 0.60      | 3           | 2                |

- **Interpretation:**  
  - U1 performed best: nearly all top chunks were directly relevant station summaries and commute/clothing guidance, giving a short, trustworthy answer.  
  - U2 retrieved mostly relevant station/condition chunks but sometimes missed or under-ranked the most hazard-focused text, so answers were useful but needed explicit reminders to check official alerts.  
  - U3 had reasonable but incomplete coverage of high-level climate explanations; the system could partially answer but needed more explicit “weather vs climate” documents to fully meet the rubric.

---

## Failure + Fix

- **Failure:** For U2 (high-stakes traveler), the system sometimes retrieved generic station summaries that described typical conditions on the travel dates but under-emphasized rare but important severe-weather scenarios (e.g., intense storms or flooding) that were mentioned in other chunks. As a result, the answer risked sounding too reassuring, with limited focus on low-probability/high-impact risks.  
- **Layer:** Primarily Retrieval and Reranking (the severe/hazard-related chunks were either not retrieved or not ranked high enough).  
- **Consequence:** A user might downplay the possibility of disruptive or dangerous conditions at their destination, failing to build contingency plans or monitor official alerts, which can carry safety and financial risks.  
- **Safeguard / next fix:**  
  - Add more explicitly labeled hazard/safety documents and ensure each hazard type (storms, floods, heat) lives in its own clear chunk.  
  - Add a rule-based boost in the hybrid+rerank pipeline for queries mentioning “risk”, “severe”, “warning”, “flood”, “storm”, or “heat” so that hazard-related chunks are always in the top context.  
  - In production, integrate live alert APIs and require the assistant to check and cite any active alerts before making safety-relevant statements.

---

## Evidence of Grounding

**Example RAG answer for U1 (daily commuter) with citations**

> For your commute today, the evidence shows that temperatures will be mild in the morning, warming to comfortable levels by the afternoon, with only a low chance of light rain [Chunk 1]. Winds are expected to remain light and there are no indications of snow, ice, or other disruptive hazards on major routes [Chunk 2].  
>  
> Based on this, a light jacket or sweater should be sufficient, and you probably do not need heavy winter gear [Chunk 1]. It’s still a good idea to bring a small umbrella or rain jacket if you’ll be walking or biking, since there is a non-zero chance of brief showers [Chunk 2]. If you normally drive or take public transit, no major delays are suggested by the weather evidence, but leaving a few extra minutes is always a reasonable precaution in case of localized showers [Chunk 1], [Chunk 2].

> **Answer:**  
>  
> **Citations:** [Chunk 1], [Chunk 2]

---

## 8. Reflection (3–5 Sentences)
What did you learn about the difference between **building a model** and **building a trustworthy product**?

---

## Reproducibility Checklist
- [ ] Project dataset included or linked
- [ ] Notebook runs end-to-end
- [ ] User stories + rubric completed
- [ ] Results table filled
- [ ] Screenshots or logs included

---

> *CS 5588 — UMKC School of Science & Engineering*  
> *“We are not building models. We are building products people can trust.”*


