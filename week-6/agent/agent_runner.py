"""
WeatherTwin Agent — Agent Runner
Multi-step ReAct-style agent that interprets user queries, selects tools,
executes them, and synthesises a grounded answer.
"""

import json
import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

from agent.tool_schemas import TOOL_SCHEMAS, get_tools_prompt
from agent.tools import TOOL_FUNCTIONS

load_dotenv()

# ── LLM client ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("AGENT_MODEL", "llama-3.3-70b-versatile")

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _client


# ── System prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are WeatherTwin AI — an intelligent weather assistant.
You have access to weather tools you can call to gather data before answering.

{get_tools_prompt()}

## How to respond

You MUST reply with **exactly one** JSON object per message (no markdown fences, no extra text).

To call a tool:
{{"thought": "why you need this tool", "action": "<tool_name>", "action_input": {{"param": "value"}}}}

To give the final answer (after you have all the data you need):
{{"thought": "summarising findings", "action": "final_answer", "action_input": "Your complete answer to the user in rich markdown."}}

## Rules
1. Always call at least one tool before giving a final answer — never guess.
2. You may call multiple tools in sequence (one per message) to build context.
3. When comparing cities use the compare_cities tool.
4. For historical or BERT questions, prefer get_historical_analysis or predict_weather_bert.
5. Keep final answers clear, organised, and evidence-grounded.
"""


# ── JSON parser (robust) ───────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Extract a JSON object from potentially messy LLM output."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find the first {...} block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from: {text[:200]}")


# ── Agent runner ────────────────────────────────────────────────────────────

class AgentRunner:
    """
    Multi-step ReAct agent.
    
    Usage
    -----
    >>> runner = AgentRunner()
    >>> result = runner.run("What's the weather in Tokyo?")
    >>> print(result["answer"])
    >>> print(result["steps"])   # reasoning trace
    """

    def __init__(self, max_steps: int = 7):
        self.max_steps = max_steps
        self.client = _get_client()

    def run(self, user_input: str) -> dict:
        """
        Execute the agent loop.
        
        Returns
        -------
        dict with keys:
            answer      – final text answer (str)
            steps       – list of step dicts (thought/action/observation)
            tools_used  – list of tool names invoked
            error       – error message if something went wrong, else None
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        steps = []
        tools_used = []

        for step_num in range(self.max_steps):
            # ── Ask the LLM ──
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=1024,
                )
            except Exception as e:
                return {
                    "answer": f"❌ API Error: {e}",
                    "steps": steps,
                    "tools_used": tools_used,
                    "error": str(e),
                }

            content = response.choices[0].message.content.strip()

            # ── Parse ──
            try:
                parsed = _parse_json(content)
            except ValueError:
                # If we can't parse, return the raw text as the answer
                return {
                    "answer": content,
                    "steps": steps,
                    "tools_used": tools_used,
                    "error": None,
                }

            thought = parsed.get("thought", "")
            action = parsed.get("action", "")
            action_input = parsed.get("action_input", "")

            # ── Final answer ──
            if action == "final_answer":
                steps.append({
                    "step": step_num + 1,
                    "thought": thought,
                    "action": "final_answer",
                    "action_input": action_input,
                    "observation": None,
                })
                return {
                    "answer": action_input,
                    "steps": steps,
                    "tools_used": tools_used,
                    "error": None,
                }

            # ── Legacy fallback (old "user_answer" action) ──
            if action == "user_answer":
                steps.append({
                    "step": step_num + 1,
                    "thought": thought,
                    "action": "final_answer",
                    "action_input": action_input,
                    "observation": None,
                })
                return {
                    "answer": action_input,
                    "steps": steps,
                    "tools_used": tools_used,
                    "error": None,
                }

            # ── Tool call ──
            if action in TOOL_FUNCTIONS:
                tool_fn = TOOL_FUNCTIONS[action]
                tools_used.append(action)

                # Build kwargs from action_input
                if isinstance(action_input, dict):
                    kwargs = action_input
                elif isinstance(action_input, str):
                    # Simple string → first required param
                    schema = next((s for s in TOOL_SCHEMAS if s["name"] == action), None)
                    if schema:
                        first_param = schema["parameters"].get("required", ["city"])[0]
                        kwargs = {first_param: action_input}
                    else:
                        kwargs = {"city": action_input}
                else:
                    kwargs = {"city": str(action_input)}

                try:
                    observation = tool_fn(**kwargs)
                except Exception as e:
                    observation = {"status": "error", "data": str(e), "source": action}

                observation_str = json.dumps(observation, default=str)

                steps.append({
                    "step": step_num + 1,
                    "thought": thought,
                    "action": action,
                    "action_input": kwargs,
                    "observation": observation,
                })

                # Feed observation back
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Observation from {action}:\n{observation_str}\n\n"
                        "Continue: call another tool if needed, or give a final_answer."
                    ),
                })
            else:
                # Unknown action
                return {
                    "answer": f"Unknown tool: {action}. Available: {list(TOOL_FUNCTIONS.keys())}",
                    "steps": steps,
                    "tools_used": tools_used,
                    "error": f"Unknown action: {action}",
                }

        # Exhausted steps
        return {
            "answer": "⚠️ Agent reached maximum reasoning steps. Please try a simpler question.",
            "steps": steps,
            "tools_used": tools_used,
            "error": "max_steps_reached",
        }


# ── Convenience wrapper ────────────────────────────────────────────────────

def run_agent(user_input: str) -> dict:
    """Module-level shortcut compatible with the old API."""
    runner = AgentRunner()
    return runner.run(user_input)
