
- **Team / Name: Sayush Maharjan, Harsha Sri Neeriganti**  
- **Project name (working title): WeatherTwin - Personalized Climate-Aware AI Assistant**  

### 0.1 Target user persona
- Who will use this? (role, context, pain point)

- Role: "Climate-Anxious Homeowner" or "Urban Renter."
- Environment: A city facing increasing climate volatility (e.g., Miami, Phoenix, or Jakarta).
- Pain Point: Generic weather apps give raw data (temperature/rainfall) but no context. Users don't know how a "2-inch rain event" translates to "my basement flooding" or how a "heatwave" impacts their specific building type.


### 0.2 Problem statement (1–2 sentences)
- What decision/task does your product support?

- Users cannot translate macro-level climate data into micro-level personal risks, leading to poor decisions regarding insurance, home safety, and daily commutes.


### 0.3 Value proposition (1 sentence)
- What improves (speed, accuracy, trust, cost, risk)?

- This module retrieves specific technical evidence (flood maps, building codes, climate models) and contextualizes it against the user's profile to deliver actionable, personal risk intelligence ("Your commute route crosses Zone X, which floods frequently").


### 0.4 Success metrics (pick 2–3)
- e.g., time-to-answer, citation coverage, % “not enough evidence” when missing, user satisfaction (1–5), precision@5

- Personalization Precision: % of retrieved results that are actually relevant to the user’s specific location/building type.
- Evidence Traceability: 100% of risk claims must cite a specific map zone, page in a climate report, or building code.
- Refusal Accuracy: System must refuse to predict risks for locations/buildings not covered in the documents (avoiding false safety assurances).


### Queries 
- Query 1: According to the FEMA Flood Safety guide, what actions should residents take during flooding to protect themselves and evacuate safely?
- Compare the claims in the Climate Action Technical Paper about global temperature increase with the temperature trend shown in the provided climate visualization. Does the visual evidence support the paper’s claims?
- What are the specific federal tax credit percentages for homeowners who install AI-integrated smart windows in the fiscal year 2028, according to the Natural Resources Canada and FEMA safety guides?

<img width="1280" height="591" alt="Screenshot 2026-02-05 at 6 48 21 PM" src="https://github.com/user-attachments/assets/0d060b36-ec16-4f40-8c4e-c126f44436db" />


### Failure Case
- Issue: The system misinterpreted a low-resolution flood map boundary, incorrectly classifying a high-risk property as "Safe" due to poor OCR/Captioning accuracy.
- Risk: Relying on this error, the user opted out of flood insurance, leading to significant financial loss and property damage during a flood event.
- Mitigation: To prevent this, the system will replace binary "Safe/Risk" labels with probability intervals, add visual disclaimers regarding map precision, and provide direct links to high-resolution official sources for human verification.




### Links

- ** Github Link:** https://github.com/sayushmaharjan/CS-5588/tree/main/week-3
- ** Colab File:** https://colab.research.google.com/github/sayushmaharjan/CS-5588/blob/main/week-3/CS5588_Week3_HandsOn.ipynb


