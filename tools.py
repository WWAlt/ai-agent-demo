"""External tools the agent can call. Both APIs are free and need no key."""
import httpx

TIMEOUT = 10.0


class ToolError(Exception):
    """Raised when a tool fails in a way the agent should be told about."""


def _get(url: str, params: dict, retries: int = 2) -> dict:
    last = None
    for _ in range(retries + 1):
        try:
            r = httpx.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e  # network problem -> retry
        except httpx.HTTPStatusError as e:
            raise ToolError(f"API returned {e.response.status_code}") from e
    raise ToolError(f"Network error after retries: {last}")


def get_weather(city: str) -> dict:
    geo = _get("https://geocoding-api.open-meteo.com/v1/search",
               {"name": city, "count": 1})
    if not geo.get("results"):
        raise ToolError(f"City not found: {city}")
    place = geo["results"][0]
    data = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,wind_speed_10m",
    })
    cur = data["current"]
    return {"city": place["name"], "country": place.get("country"),
            "temperature_c": cur["temperature_2m"],
            "wind_kmh": cur["wind_speed_10m"]}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    data = _get("https://api.frankfurter.app/latest", {
        "amount": amount, "from": from_currency.upper(), "to": to_currency.upper(),
    })
    rates = data.get("rates", {})
    if to_currency.upper() not in rates:
        raise ToolError(f"Unsupported currency: {to_currency}")
    return {"amount": amount, "from": from_currency.upper(),
            "to": to_currency.upper(), "result": rates[to_currency.upper()],
            "date": data.get("date")}


# Tool definitions sent to the LLM
TOOL_SPECS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {"type": "object",
                         "properties": {"city": {"type": "string"}},
                         "required": ["city"]},
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between currencies (ISO codes like EUR, USD).",
        "input_schema": {"type": "object", "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"}},
            "required": ["amount", "from_currency", "to_currency"]},
    },
]

TOOLS = {"get_weather": get_weather, "convert_currency": convert_currency}


def run_tool(name: str, args: dict) -> dict:
    """Run a tool and always return a dict (errors become {'error': ...})."""
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOLS[name](**args)
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}
