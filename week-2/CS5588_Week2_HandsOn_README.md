# CS 5588 — Week 2 Hands-On
## Applied RAG for Product & Venture Development

> **Course:** CS 5588 — Data Science Capstone / GenAI Product Leadership  
> **Student:**  
> **Project / Product Name:**  
> **GitHub:**  
> **Date:**  

---

## 1. Product Overview
### Problem Statement
Describe the real-world problem your product is solving. What is broken, slow, risky, or expensive today?

### Target Users
Who will use this system in practice? (e.g., clinicians, analysts, compliance officers, educators, founders, NGOs, etc.)

### Value Proposition
Why would someone choose your AI system over existing tools, search engines, or manual workflows?

---

## 2. Dataset Reality
### Data Source & Ownership
- Source:
- Owner (public / company / agency / internal):

### Sensitivity & Ethics
- Sensitivity level (public / internal / regulated / confidential):
- Privacy / compliance concerns:

### Document Types
- Examples: policies, manuals, research, reports, SOPs, meeting notes, etc.

### Expected Scale in Production
- How many documents would this system realistically manage?

---

## 3. User Stories & Risk Awareness

### U1 — Normal Use Case
> As a ___, I want to ___ so that I can ___.

**Acceptable Evidence:**  
**Correct Answer Criteria:**  

### U2 — High-Stakes Case
> As a ___, I want to ___ so that I can ___.

**Why This Is High Risk:**  
**Acceptable Evidence:**  
**Correct Answer Criteria:**  

### U3 — Ambiguous / Failure Case
> As a ___, I want to ___ so that I can ___.

**What Could Go Wrong:**  
**Safeguard Needed:**  

---

## 4. System Architecture (Product View)

### Chunking Strategy
- Fixed or Semantic:
- Chunk size / overlap:
- Why this fits your product users:

### Retrieval Design
- Keyword layer (TF-IDF / BM25):
- Vector layer (embedding model + index):
- Hybrid α value(s):

### Governance Layer
- Re-ranking method (Cross-Encoder / LLM Judge / None):
- What risk this layer reduces:

### Generation Layer
- Model used:
- Grounding & citation strategy:
- Abstention policy (“Not enough evidence” behavior):

---

## 5. Results

| User Story | Method (Keyword / Vector / Hybrid) | Precision@5 | Recall@10 | Trust Score (1–5) | Confidence Score (1–5) |
|------------|-----------------------------------|-------------|-----------|-------------------|-------------------------|
| U1         |                                   |             |           |                   |                         |
| U2         |                                   |             |           |                   |                         |
| U3         |                                   |             |           |                   |                         |

---

## 6. Failure Case & Venture Fix

### Observed Failure
Describe one real failure you observed in your system.

### Real-World Consequence
What could happen if this system were deployed as-is? (legal, financial, ethical, safety, trust, etc.)

### Proposed System-Level Fix
What would you change next?
- Data
- Chunking
- Hybrid α
- Re-ranking
- Human-in-the-loop review

---

## 7. Evidence of Grounding

Paste one **RAG-grounded answer** below with citations.

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