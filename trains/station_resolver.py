"""IRCTC Railway Station Entity Resolver (Data-Decoupled Architecture).

Pure algorithmic entity resolver:
- Loads reference datasets dynamically from external JSON files in data/
- Maintains persistent station_cache.json for dynamically discovered stations
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
STATIONS_FILE = DATA_DIR / "stations.json"
STATES_FILE = DATA_DIR / "states.json"
STATION_CACHE_FILE = BASE_DIR / "station_cache.json"

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
STATION_REGISTRY: dict[str, Any] = _load_json_dataset(STATIONS_FILE)
INDIAN_STATES_DIRECTORY: dict[str, Any] = _load_json_dataset(STATES_FILE)
DYNAMIC_STATION_CACHE: dict[str, Any] = _load_json_dataset(STATION_CACHE_FILE)

# Popular railway hubs for UI dropdown selection
POPULAR_CITY_STATION_OPTIONS = [
    "Hyderabad / Secunderabad (SC / HYB / KCG)",
    "Vijayawada (BZA)",
    "Visakhapatnam (VSKP)",
    "Tirupati (TPTY / RU)",
    "Goa / Madgaon (MAO / KRMI / THVM)",
    "Delhi NCR (NDLS / NZM / DLI / ANVT)",
    "Mumbai (CSMT / MMCT / LTT / BDTS)",
    "Bengaluru / Bangalore (SBC / YPR / SMVB)",
    "Chennai (MAS / MS)",
    "Kolkata (HWH / SDAH / KOAA)",
    "Jaipur (JP)",
    "Varanasi (BSB / DDU)",
    "Ahmedabad (ADI)",
    "Pune (PUNE)",
    "Kochi / Ernakulam (ERS / ERN)",
    "Lucknow (LKO / LJN)",
    "Patna (PNBE)",
    "Bhubaneswar (BBS)",
    "Amritsar (ASR)",
    "Chandigarh (CDG)",
    "Guwahati (GHY)",
    "Agra (AGC / AF)",
    "Hosapete / Hampi (HPT)",
    "Gokarna (GOK)",
]


def _save_dynamic_cache() -> None:
    """Persist newly discovered stations to disk."""
    try:
        with open(STATION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(DYNAMIC_STATION_CACHE, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def normalize_input(raw: str) -> str:
    """Normalize user input string."""
    if not raw:
        return ""
    return re.sub(r"[^\w\s/]", "", str(raw).lower()).strip()


def is_state_or_region(query: str) -> tuple[bool, str, list[dict[str, str]]]:
    """Check if query is a known state or multi-station region."""
    cleaned = normalize_input(query)
    for state, stations in INDIAN_STATES_DIRECTORY.items():
        if state == cleaned or state in cleaned:
            return True, state.title(), stations
    return False, "", []


def resolve_via_llm(query: str) -> dict[str, Any] | None:
    """Dynamically resolve any unlisted station, city, or hill station via Gemini."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            return None

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            google_api_key=gemini_key,
        )

        prompt = (
            "You are an expert Indian Railways (IRCTC) station entity resolver.\n"
            f"Analyze the location query: '{query}'\n"
            "Return a strictly valid JSON object with:\n"
            "{\n"
            '  "is_state_or_region": boolean (true ONLY if query represents an entire broad state/province like Andhra Pradesh, Kerala, Bavaria),\n'
            '  "primary_code": 2-5 letter standard IRCTC station code (e.g. NDLS for New Delhi, CSMT for Mumbai, SC for Secunderabad, BZA for Vijayawada, KGM for Nainital). Set null ONLY if state/region,\n'
            '  "cluster_codes": list of station codes serving this city/metro area (e.g. ["SC", "HYB", "KCG"] for Hyderabad),\n'
            '  "canonical_city": standard city name,\n'
            '  "station_name": official railway station name,\n'
            '  "display_label": human-friendly label e.g. "Hyderabad - Secunderabad (SC)" or "Nainital (via Kathgodam KGM)"\n'
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


def resolve_city_to_station(query: str, allow_llm: bool = True) -> tuple[str, list[str], str, str]:
    """Resolve a city or place name into (primary_code, cluster_codes, display_city, station_name).

    Returns:
        (primary_code, cluster_codes, display_city, station_name)
    Raises:
        ValueError if the query is an ambiguous state/region or unresolvable.
    """
    if not query or not query.strip():
        raise ValueError("Station query cannot be empty.")

    cleaned = normalize_input(query)

    # 1. State / Region check
    is_region, region_name, stations = is_state_or_region(cleaned)
    if is_region:
        stn_str = ", ".join([f"{s['city']} ({s['code']})" for s in stations])
        raise ValueError(
            f"'{query}' is a state or broad region with multiple railway junctions ({stn_str}). "
            f"Please specify a specific city."
        )

    # 2. Reference Dataset Direct Match (loaded from data/stations.json)
    if cleaned in STATION_REGISTRY:
        info = STATION_REGISTRY[cleaned]
        return (info["primary_code"], info["cluster_codes"], info["city"], info["station_name"])

    # 3. Dynamic Local Cache Direct Match (station_cache.json)
    if cleaned in DYNAMIC_STATION_CACHE:
        cached = DYNAMIC_STATION_CACHE[cleaned]
        return (
            cached["primary_code"],
            cached.get("cluster_codes") or [cached["primary_code"]],
            cached["city"],
            cached["station_name"],
        )

    # 4. Check if station code is explicitly provided in parentheses e.g. "Hospet (HPT)" or "New Delhi (NDLS)"
    paren_match = re.search(r"\(([A-Za-z]{2,5})\)", query)
    if paren_match:
        code_cand = paren_match.group(1).upper()
        for info in STATION_REGISTRY.values():
            if info.get("primary_code") == code_cand or code_cand in info.get("cluster_codes", []):
                return (code_cand, info.get("cluster_codes", [code_cand]), info["city"], info["station_name"])
        return (code_cand, [code_cand], f"{code_cand} Station", f"Station ({code_cand})")

    # If the user directly passed a pure uppercase IRCTC station code (e.g. "SC", "BZA", "CSMT", "NDLS")
    stripped_raw = query.strip()
    if stripped_raw.isupper() and 2 <= len(stripped_raw) <= 5 and stripped_raw.isalpha():
        for info in STATION_REGISTRY.values():
            if info.get("primary_code") == stripped_raw or stripped_raw in info.get("cluster_codes", []):
                return (stripped_raw, info.get("cluster_codes", [stripped_raw]), info["city"], info["station_name"])
        return (stripped_raw, [stripped_raw], f"{stripped_raw} Station", f"Station ({stripped_raw})")

    # 5. Substring matching in dataset keys
    for key, info in STATION_REGISTRY.items():
        if key in cleaned or cleaned in key:
            return (info["primary_code"], info["cluster_codes"], info["city"], info["station_name"])

    # 6. Dynamic Tier: AI Entity Resolver via Gemini
    if allow_llm:
        data = resolve_via_llm(query)
        if data:
            primary = data.get("primary_code")
            if data.get("is_state_or_region") and not primary:
                raise ValueError(
                    f"'{query}' is a state or broad region with multiple railway stations. Please specify a specific city."
                )

            if primary and (2 <= len(primary) <= 5):
                primary = primary.upper()
                clusters = [c.upper() for c in (data.get("cluster_codes") or [primary])]
                if primary not in clusters:
                    clusters.insert(0, primary)
                city_name = data.get("canonical_city") or query.strip().title()
                station_name = data.get("station_name") or f"{city_name} Station"
                display_label = data.get("display_label") or f"{city_name} ({primary})"

                # Save to cache
                entry = {
                    "primary_code": primary,
                    "cluster_codes": clusters,
                    "city": city_name,
                    "station_name": station_name,
                    "display_label": display_label,
                }
                DYNAMIC_STATION_CACHE[cleaned] = entry
                _save_dynamic_cache()

                return (primary, clusters, city_name, station_name)

    # 7. Fallback: Pure 2-5 letter string passed directly
    upper_raw = query.strip().upper()
    if 2 <= len(upper_raw) <= 5 and upper_raw.isalpha():
        return (upper_raw, [upper_raw], upper_raw, f"Station ({upper_raw})")

    raise ValueError(
        f"Unable to resolve railway station for '{query}'. Please provide a valid city name or IRCTC station code."
    )
