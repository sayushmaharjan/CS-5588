"""
WeatherTwin Agent — Tool Schemas
Defines JSON-schema descriptions for each agent tool so the LLM knows
what tools are available, what arguments they expect, and when to use them.
"""

TOOL_SCHEMAS = [
    {
        "name": "get_current_weather",
        "description": (
            "Get real-time current weather conditions for a city. "
            "Returns temperature, humidity, wind speed, UV index, air quality, "
            "and weather condition. Use this when the user asks about the "
            "current or today's weather in a specific location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g. 'Los Angeles', 'New York', 'Tokyo')"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_weather_forecast",
        "description": (
            "Get an hourly weather forecast for the next N hours for a city. "
            "Returns temperature, humidity, condition, and wind for each hour. "
            "Use this when the user asks about upcoming weather, forecasts, "
            "or planning for a specific time window."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                },
                "hours": {
                    "type": "integer",
                    "description": "Number of hours to forecast (default 12, max 48)",
                    "default": 12
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_historical_analysis",
        "description": (
            "Get a statistical analysis of historical weather data for a city. "
            "Returns average temperature, temperature range, average wind speed, "
            "most common condition, and number of records analysed. "
            "Available cities: Los Angeles, San Diego, San Francisco. "
            "Use this to provide context about typical weather patterns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (Los Angeles, San Diego, or San Francisco)"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "predict_weather_bert",
        "description": (
            "Use a BERT deep-learning model to classify and predict weather "
            "conditions for a city based on historical data patterns. "
            "Returns predicted condition, confidence score, and statistical "
            "summary. Available cities: Los Angeles, San Diego, San Francisco. "
            "Use this when the user asks for an AI-powered prediction or forecast."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (Los Angeles, San Diego, or San Francisco)"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "compare_cities",
        "description": (
            "Compare current weather conditions between two cities side-by-side. "
            "Returns both cities' temperature, humidity, wind, UV, and conditions "
            "plus a textual comparison. Use this when the user wants to compare "
            "weather across locations or decide between destinations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city1": {
                    "type": "string",
                    "description": "First city name"
                },
                "city2": {
                    "type": "string",
                    "description": "Second city name"
                }
            },
            "required": ["city1", "city2"]
        }
    }
]


def get_tools_prompt() -> str:
    """Build a formatted string listing all tools for the system prompt."""
    lines = ["Available tools:\n"]
    for schema in TOOL_SCHEMAS:
        params = schema["parameters"]["properties"]
        param_list = ", ".join(
            f'{k} ({v["type"]}{"" if k in schema["parameters"].get("required", []) else ", optional"})'
            for k, v in params.items()
        )
        lines.append(f'- **{schema["name"]}**({param_list}): {schema["description"]}')
    return "\n".join(lines)
