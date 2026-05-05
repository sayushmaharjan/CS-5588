"""
story_generator.py
~~~~~~~~~~~~~~~~~~
Replaces LyricsProcessor.

Calls an LLM to generate a complete children's story broken into pages,
with illustration prompts and mood/SFX metadata per page.

Reuses the EXACT same _call_llm() / _detect_provider() multi-provider
routing logic from the original lyrics_processor.py — only the prompts
and output parsing are different.
"""

import os
import json
import time
import sys
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from storybook_generator.utils.story_schemas import Storybook, StoryPage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STORY_SYSTEM_PROMPT = """\
You are an award-winning children's book author.
Your task: write an original, magical, age-appropriate children's story
from a simple idea. The story must be warm, imaginative, and suitable
for a bedtime read-aloud.

Output ONLY valid JSON. No markdown fences. No explanations."""

STORY_USER_PROMPT = """\
Write a children's bedtime story with the following details:

Child's name: {child_name}
Child's age: {age}
Story theme / idea: {theme}
Number of pages: {num_pages}
Tone: calm, magical, bedtime-friendly, age-appropriate for {age}-year-olds

Requirements for each page:
- Story text: 2–4 sentences, simple vocabulary, warm and descriptive
- The child's name ({child_name}) must appear naturally in the story
- Illustration prompt: vivid, Pixar-style description of what to show
- Mood: one of [calm, happy, magical, adventure, sleepy]
- Sound effects: a list of 0–2 ambient sounds from:
  [forest, magic, ocean, wind, rain, birds, night, cozy, stars, sparkle]

Return JSON in this exact schema:
{{
  "title": "Story title here",
  "music_prompt": "MusicGen prompt for gentle ambient background music (calm, instrumental)",
  "color_palette": "soft pastels, warm, dreamy",
  "art_style": "Pixar-style children's book illustration",
  "pages": [
    {{
      "page_number": 1,
      "text": "Story text for this page...",
      "illustration_prompt": "Detailed scene for illustration, Pixar-style, soft lighting...",
      "mood": "calm",
      "sound_effects": ["forest"]
    }}
  ]
}}

Rules:
- Exactly {num_pages} pages
- The story must have a beginning, middle, and a calm/happy ending
- Each page transitions naturally to the next
- Page {num_pages} should feel sleepy and resolved (bedtime-friendly)
- Include the child named {child_name} as the main character
"""


def _clean_json_response(text: str) -> str:
    """Extract JSON from a response that might include markdown or extra text.
    Identical to the helper in lyrics_processor.py."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return text.strip()


# ---------------------------------------------------------------------------
# StoryGenerator class
# ---------------------------------------------------------------------------

class StoryGenerator:
    """
    Generates a complete Storybook from a simple idea + child info.

    Multi-provider LLM routing is identical to LyricsProcessor:
    Anthropic | HuggingFace Inference | OpenAI-compatible (Groq, Together, Ollama…) | local pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
    ):
        self.api_key = api_key or ""
        if not self.api_key:
            self.api_key = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("GROQ_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or ""
            )

        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        self.model = model or os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

        self.client = None
        self.pipe = None
        self.provider = self._detect_provider()

        if self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")

        elif self.provider == "local":
            if not os.path.isdir(self.model):
                raise ValueError(f"Local model path not found: {self.model}")

    # ── Provider detection (verbatim from LyricsProcessor) ──────────────────

    def _detect_provider(self) -> str:
        """Detect which LLM provider to use. Identical to LyricsProcessor._detect_provider."""
        if self.base_url:
            if "huggingface" in self.base_url:
                return "huggingface"
            elif "anthropic" in self.base_url:
                return "anthropic"
            else:
                return "openai_compatible"
        if self.api_key.startswith("hf_"):
            return "huggingface"
        if self.api_key.startswith("sk-ant-"):
            return "anthropic"
        if "/" in self.model and not self.model.startswith(("gpt-", "claude-", "o1", "o3")):
            return "huggingface"
        if os.path.isdir(self.model):
            return "local"
        return "openai_compatible"

    # ── LLM call (verbatim from LyricsProcessor, all providers) ─────────────

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
        """Call LLM — identical routing to LyricsProcessor._call_llm."""
        if self.provider == "anthropic":
            if self.client is None:
                raise RuntimeError("Anthropic client not initialized")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        elif self.provider == "huggingface":
            import requests
            url = f"https://api-inference.huggingface.co/models/{self.model}/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.8,
            }
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code != 200:
                try:
                    err = response.json()
                    err_msg = err.get("error", err.get("message", str(err)))
                except Exception:
                    err_msg = response.text[:500]
                raise RuntimeError(
                    f"HuggingFace API error ({response.status_code}): {err_msg}\n"
                    f"Model: {self.model}"
                )
            return response.json()["choices"][0]["message"]["content"]

        elif self.provider == "local":
            if self.pipe is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                tokenizer = AutoTokenizer.from_pretrained(self.model)
                model_obj = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.pipe = pipeline("text-generation", model=model_obj, tokenizer=tokenizer)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = self.pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = self.pipe(
                prompt_text,
                max_new_tokens=max_tokens,
                temperature=0.8,
                do_sample=True,
                return_full_text=False,
            )
            return outputs[0]["generated_text"]

        else:
            # OpenAI-compatible (Groq, Together, OpenAI, Ollama, etc.)
            import requests
            if self.base_url:
                url = self.base_url.rstrip("/") + "/chat/completions"
            else:
                url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.8,
            }
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                if response.status_code == 429:
                    print(f"Rate limit hit. Retrying in 15s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(15)
                    continue
                if response.status_code != 200:
                    try:
                        err = response.json()
                        err_msg = err.get("error", {}).get("message", str(err))
                    except Exception:
                        err_msg = response.text[:500]
                    raise RuntimeError(
                        f"API error ({response.status_code}): {err_msg}\nModel: {self.model}"
                    )
                return response.json()["choices"][0]["message"]["content"]
            raise RuntimeError("Exceeded maximum retries for rate limits.")

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        child_name: str,
        age: int,
        theme: str,
        num_pages: int = 4,
    ) -> Storybook:
        """
        Generate a complete Storybook from a child's name, age, and theme idea.

        Returns a Storybook dataclass with StoryPage objects ready for
        illustration, narration, and rendering.
        """
        logger.info(f"Generating story for {child_name} (age {age}): '{theme}' ({num_pages} pages)")

        user_prompt = STORY_USER_PROMPT.format(
            child_name=child_name,
            age=age,
            theme=theme,
            num_pages=num_pages,
        )

        response_text = self._call_llm(
            system_prompt=STORY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=4000,
        )

        cleaned = _clean_json_response(response_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw response (first 500 chars):\n{cleaned[:500]}")
            raise

        return self._parse_to_storybook(data, child_name, age, theme)

    def _parse_to_storybook(self, data: dict, child_name: str, age: int, theme: str) -> Storybook:
        """Parse LLM JSON response into a Storybook dataclass."""
        pages = []
        for p in data.get("pages", []):
            page = StoryPage(
                page_number=p.get("page_number", len(pages) + 1),
                text=p.get("text", ""),
                illustration_prompt=p.get("illustration_prompt", ""),
                mood=p.get("mood", "calm"),
                sound_effects=p.get("sound_effects", []),
            )
            pages.append(page)

        return Storybook(
            title=data.get("title", f"{child_name}'s Adventure"),
            child_name=child_name,
            age=age,
            theme=theme,
            pages=pages,
            music_prompt=data.get(
                "music_prompt",
                "gentle ambient piano, soft, calming, children's bedtime music, slow tempo",
            ),
            color_palette=data.get("color_palette", "soft pastels, warm, dreamy"),
            art_style=data.get("art_style", "Pixar-style children's book illustration"),
        )
