# Individual Contribution — Sayush Maharjan

I reproduced the **Personalized Climate-Aware Health Navigator**, a Java-based web app that combines OpenWeather API data with LLM-generated health recommendations. The README had **no setup instructions**, so I used antigravity to reverse-engineer the deployment by reading the source code, creating a **Docker Compose setup** (MySQL 8 + Tomcat 10.1), writing the database schema, and building the WAR file with Maven. I verified the app running at `localhost:8080`.

I fixed several issues during reproduction: a **URL encoding bug** causing API failures for cities with spaces, a **deprecated Gemini model** (`gemini-pro`), as I was using a free tier, and **API rate-limiting** on the free tier. I replaced Gemini entirely with the **Groq API** (`llama-3.3-70b-versatile`), which was faster and more reliable.

I also analyzed how the project's LLM-weather pattern relates to WeatherTwin, noting that it lacks historical grounding, citations, and retrieval augmentation — gaps our RAG architecture addresses.

## Reflection

This exercise showed how critical good documentation is for reproducibility — unlike CLIMADA, this project required significant reverse-engineering. The Gemini deprecation reinforced the need for a provider-agnostic LLM layer in WeatherTwin, a pattern I validated by successfully swapping APIs. Seeing the Health Navigator's limitations — no historical context, no source attribution — confirmed that WeatherTwin's RAG-based design with FAISS retrieval and citation tracking is the right approach.