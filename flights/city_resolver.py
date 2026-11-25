"""City to Airport Entity Resolver (Data-Decoupled Architecture).

Pure algorithmic entity resolver:
- Loads reference datasets dynamically from external JSON files in data/
- Maintains persistent city_cache.json for dynamically discovered airports
- Resolves unlisted destinations via Gemini LLM with zero hardcoded code-bloat.
"""
import json
import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AIRPORTS_FILE = DATA_DIR / "airports.json"
STATES_FILE = DATA_DIR / "states.json"
CITY_CACHE_FILE = BASE_DIR / "city_cache.json"

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
load_dotenv()


def _load_json_dataset(file_path: Path) -> dict[str, Any]:
    """Safely load external reference JSON datasets."""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# Load decoupled reference datasets on demand
CITY_AIRPORT_REGISTRY: dict[str, Any] = _load_json_dataset(AIRPORTS_FILE)
INDIAN_STATES_DIRECTORY: dict[str, Any] = _load_json_dataset(STATES_FILE)
DYNAMIC_CITY_CACHE: dict[str, Any] = _load_json_dataset(CITY_CACHE_FILE)

# Unique list of popular display options for UI dropdowns
POPULAR_CITY_OPTIONS = [
    "Hyderabad / Secunderabad (HYD)",
    "Goa - All Airports (GOI/GOX)",
    "Goa - Dabolim South (GOI)",
    "Goa - Manohar Mopa North (GOX)",
    "Delhi / NCR (DEL)",
    "Mumbai (BOM)",
    "Bengaluru / Bangalore (BLR)",
    "Jaipur (JAI)",
    "Kochi / Cochin (COK)",
    "Chennai (MAA)",
    "Kolkata (CCU)",
    "Pune (PNQ)",
    "Ahmedabad (AMD)",
    "Visakhapatnam (VTZ)",
    "Vijayawada (VGA)",
    "Tirupati (TIR)",
    "Varanasi (VNS)",
    "Srinagar (SXR)",
    "Amritsar (ATQ)",
    "Chandigarh (IXC)",
    "Lucknow (LKO)",
    "Guwahati (GAU)",
    "Udaipur (UDR)",
    "Dehradun / Rishikesh (DED)",
    "Bhubaneswar (BBI)",
    "Thiruvananthapuram (TRV)",
    "Munnar (via Kochi COK)",
    "Manali (via Chandigarh IXC)",
]


def _save_dynamic_cache() -> None:
    """Persist newly discovered cities to disk."""
    try:
        with open(CITY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(DYNAMIC_CITY_CACHE, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def normalize_input(raw: str) -> str:
    """Normalize user input string."""
    if not raw:
        return ""
    return re.sub(r"[^\w\s/]", "", str(raw).lower()).strip()


def is_state_or_region(query: str) -> tuple[bool, str, list[dict[str, str]]]:
    """Check if query is a known state or broad multi-airport region."""
    cleaned = normalize_input(query)
    for state, airports in INDIAN_STATES_DIRECTORY.items():
        if state == cleaned or state in cleaned:
            return True, state.title(), airports
    return False, "", []


def resolve_via_llm(query: str) -> dict[str, Any] | None:
    """Dynamically resolve any unlisted city, hill station, or international destination via Gemini."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            return None

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            google_api_key=gemini_key,
        )

        prompt = (
            "You are an expert aviation and geographic entity resolver.\n"
            f"Analyze the location query: '{query}'\n"
            "Return a strictly valid JSON object with:\n"
            "{\n"
            '  "is_state_or_region": boolean (true ONLY if query represents an entire large state, province, or wide multi-airport region like Andhra Pradesh, Rajasthan, Kerala, Bavaria),\n'
            '  "primary_iata": 3-letter IATA code of the commercial airport or nearest commercial passenger airport (e.g. ZRH for Zurich, KUU for Spiti/Manali, DXB for Dubai). Set null ONLY if it is an ambiguous multi-airport state/province,\n'
            '  "canonical_city": standard city name,\n'
            '  "airport_name": official airport name,\n'
            '  "display_name": human-friendly label e.g. "Zurich (ZRH)" or "Spiti (via Bhuntar KUU, ~230 km)"\n'
            "}\n"
            "Output JSON only."
        )

        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = content[0].get("text", "")
        clean_json = (
            content.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        return json.loads(clean_json)
    except Exception:
        return None


def resolve_city_to_airport(query: str, allow_llm: bool = True) -> tuple[str, str, str]:
    """Resolve a city or place name into (IATA_code, Display_City, Airport_Name).

    Returns:
        (iata_code, display_city, airport_name)
    Raises:
        ValueError if the input is an ambiguous state/region or unresolvable.
    """
    if not query or not query.strip():
        raise ValueError("Location query cannot be empty.")

    cleaned = normalize_input(query)

    # 1. State / Region check
    is_region, region_name, airports = is_state_or_region(cleaned)
    if is_region:
        airport_str = ", ".join([f"{a['city']} ({a['code']})" for a in airports])
        raise ValueError(
            f"'{query}' is a state or broad region with multiple airports ({airport_str}). "
            f"Please specify a specific city."
        )

    # 2. Reference Dataset Direct Match (loaded from data/airports.json)
    if cleaned in CITY_AIRPORT_REGISTRY:
        info = CITY_AIRPORT_REGISTRY[cleaned]
        return (info["primary_code"], info["city"], info["airport_name"])

    # 3. Dynamic Local Cache Direct Match (city_cache.json)
    if cleaned in DYNAMIC_CITY_CACHE:
        cached = DYNAMIC_CITY_CACHE[cleaned]
        return (cached["iata"], cached["city"], cached["airport_name"])

    # 4. Check if a 3-letter IATA code was extracted in parentheses e.g. "Zurich (ZRH)"
    paren_match = re.search(r"\(([A-Za-z]{3})\)", query)
    if paren_match:
        code_cand = paren_match.group(1).upper()
        for info in CITY_AIRPORT_REGISTRY.values():
            if info.get("primary_code") == code_cand or code_cand in info.get("alternate_codes", []):
                return (code_cand, info["city"], info["airport_name"])
        return (code_cand, f"{code_cand} Airport", f"Airport ({code_cand})")

    # If the user passed a pure 3-letter uppercase IATA code (e.g. "ZRH", "HYD", "DEL")
    stripped_raw = query.strip()
    if stripped_raw.isupper() and len(stripped_raw) == 3 and stripped_raw.isalpha():
        for info in CITY_AIRPORT_REGISTRY.values():
            if info.get("primary_code") == stripped_raw or stripped_raw in info.get("alternate_codes", []):
                return (stripped_raw, info["city"], info["airport_name"])
        return (stripped_raw, f"{stripped_raw} Airport", f"Airport ({stripped_raw})")

    # 5. Substring matching in dataset keys
    for key, info in CITY_AIRPORT_REGISTRY.items():
        if key in cleaned or cleaned in key:
            return (info["primary_code"], info["city"], info["airport_name"])

    # 6. Dynamic Tier: AI Entity Resolver via Gemini
    if allow_llm:
        data = resolve_via_llm(query)
        if data:
            iata = data.get("primary_iata")
            if data.get("is_state_or_region") and not iata:
                raise ValueError(
                    f"'{query}' is a state or broad region with multiple airports, not a single destination. Please specify a specific city."
                )

            if iata and len(iata) == 3:
                iata = iata.upper()
                city_name = data.get("canonical_city") or query.strip().title()
                airport_name = data.get("airport_name") or f"{city_name} Airport"
                display_label = data.get("display_name") or f"{city_name} ({iata})"

                # Save to dynamic in-memory & persistent cache
                entry = {
                    "iata": iata,
                    "city": city_name,
                    "airport_name": airport_name,
                    "display_label": display_label,
                }
                DYNAMIC_CITY_CACHE[cleaned] = entry
                _save_dynamic_cache()

                if display_label not in POPULAR_CITY_OPTIONS:
                    POPULAR_CITY_OPTIONS.append(display_label)

                return (iata, city_name, airport_name)

    # 7. Fallback: Pure 3-letter IATA string passed directly
    upper_raw = query.strip().upper()
    if len(upper_raw) == 3 and upper_raw.isalpha():
        return (upper_raw, upper_raw, f"Airport {upper_raw}")

    raise ValueError(
        f"Unable to resolve airport code for '{query}'. Please enter a valid city name or 3-letter airport code."
    )
