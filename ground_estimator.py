"""Ground Transit & Last-Mile Road Estimator (Decoupled & Self-Learning Architecture).

Pure algorithmic engine:
- Loads seed reference destinations from external `data/gateways.json`
- Maintains persistent dynamic cache `gateway_cache.json` on disk
- Resolves any new or unlisted destination dynamically via Gemini LLM
- Calculates realistic road transit times, distances, and calibrated Indian market fares
  (Auto-Rickshaw, State RTC Bus, Rapido Bike, and Station Cabs/Innova)
- Strictly marks all estimates with `is_live_api: False` to eliminate hallucinations.
"""
import json
import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from maps_service import find_nearest_transit_hub, get_exact_route

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GATEWAYS_FILE = DATA_DIR / "gateways.json"
GATEWAY_CACHE_FILE = BASE_DIR / "gateway_cache.json"

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
load_dotenv()


def _load_json_dataset(file_path: Path) -> dict[str, Any]:
    """Safely load external JSON reference datasets."""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


REFERENCE_GATEWAYS: dict[str, Any] = _load_json_dataset(GATEWAYS_FILE)
DYNAMIC_GATEWAY_CACHE: dict[str, Any] = _load_json_dataset(GATEWAY_CACHE_FILE)


def _save_dynamic_cache() -> None:
    """Persist newly discovered destination gateways to disk."""
    try:
        with open(GATEWAY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(DYNAMIC_GATEWAY_CACHE, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _resolve_gateway_via_llm(query: str) -> dict[str, Any] | None:
    """Dynamically resolve nearest railway station and last-mile road transfers for ANY place in India.

    Uses Google Places API (New) for verified ground-truth station discovery, then resolves IRCTC codes.
    """
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return None

    # Step 1: Query Google Places for verified ground-truth station near destination
    places_hint = ""
    hub = find_nearest_transit_hub(query, hub_type="railway_station")
    if hub and hub.get("hub_name"):
        places_hint = f"\nGoogle Places API verified nearest railway hub: '{hub['hub_name']}' located at '{hub.get('formatted_address', '')}'."

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            google_api_key=gemini_key,
            temperature=0.0,
        )
        prompt = f"""You are an Indian Geographic Transportation Specialist.
Given the tourist destination, hill station, pilgrimage spot, or village: "{query}"{places_hint}

Identify the closest transportation routing:
1. Destination full formal name
2. Primary broad-gauge Railway Junction (Name and IRCTC Station Code, e.g. "Warangal (WL)", "Kalka (KLK)", "Madanapalle Road (MPL)", "Visakhapatnam (VSKP)")
3. Station code (e.g. "WL", "CDG", "HPT")
4. Intermediate bus hub or nearest connecting town
5. Direct road distance from that railway station to the destination in km (float)
6. Step-by-step local road legs (List of dicts: leg_name, mode, transit_type, from_point, to_point, distance_km, notes)

Return ONLY valid JSON with keys:
"destination_name": string,
"primary_rail_hub": string,
"rail_station_code": string,
"intermediate_bus_hub": string,
"direct_cab_distance_km": float,
"legs": list of objects
"""
        resp = llm.invoke(prompt)
        text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if parsed.get("primary_rail_hub") and parsed.get("rail_station_code"):
                # If Google Routes API can compute exact road distance from station to destination, use it!
                exact_rt = get_exact_route(parsed["primary_rail_hub"], parsed["destination_name"])
                if exact_rt and exact_rt.get("distance_km"):
                    parsed["direct_cab_distance_km"] = float(exact_rt["distance_km"])
                return parsed
    except Exception:
        pass
    return None


def calculate_generic_road_transit(
    from_place: str,
    to_place: str,
    approx_distance_km: float = 0.0,
    passengers: int = 1,
) -> dict[str, Any]:
    """Calculate road transit guidance using Google Routes API with fallback to calibrated Indian pricing formulas."""
    pax = max(1, int(passengers))
    live_route = None
    dist = max(1.0, float(approx_distance_km or 0.0))

    # 1. Attempt Live Google Routes API for exact distance & traffic time (<0.3s cached)
    if from_place and to_place:
        live_route = get_exact_route(from_place, to_place, travel_mode="DRIVE")
        if live_route and live_route.get("distance_km"):
            dist = float(live_route["distance_km"])

    # 2. Derive traffic-aware duration
    if live_route and live_route.get("duration_minutes"):
        duration_mins = int(live_route["duration_minutes"])
        dur_fmt = str(live_route.get("duration_formatted", f"{duration_mins} mins"))
        is_live_route = True
        route_source = "google_maps_routes_api"
    else:
        speed_kmh = 25.0 if dist <= 15 else (40.0 if dist <= 100 else 50.0)
        duration_mins = int((dist / speed_kmh) * 60)
        hours = duration_mins // 60
        mins = duration_mins % 60
        dur_fmt = f"{hours}h {mins}m" if hours > 0 else f"{mins} mins"
        is_live_route = False
        route_source = "calibrated_estimation_model"

    bus_fare_per_pax = int(dist * 1.60)
    total_bus_fare = bus_fare_per_pax * pax
    auto_fare = int(30 + (dist * 16.0)) if dist <= 25 else int(dist * 15.0)

    if pax >= 4:
        cab_type = "SUV / Innova"
        cab_fare = max(1400, int(dist * 24.0 + 300))
    else:
        cab_type = "Sedan (Dzire / Etios)"
        cab_fare = max(900, int(dist * 18.0 + 200))

    return {
        "is_live_api": is_live_route,
        "data_type": "LIVE_GOOGLE_ROUTES" if is_live_route else "ESTIMATED_ROAD_GUIDANCE",
        "route_source": route_source,
        "from_point": from_place,
        "to_point": to_place,
        "distance_km": round(dist, 1),
        "estimated_duration_formatted": dur_fmt,
        "estimated_duration_minutes": duration_mins,
        "options": {
            "bus": {
                "available": dist >= 10,
                "transit_type": "State RTC Bus (TSRTC / APSRTC / HRTC)",
                "fare_per_passenger_inr": bus_fare_per_pax,
                "total_fare_inr": total_bus_fare,
                "notes": "Bus fare based on state RTC stage tariffs",
            },
            "auto": {
                "available": dist <= 30,
                "transit_type": "Local Auto-Rickshaw",
                "total_fare_inr": auto_fare,
                "notes": "Ideal for local transfer within city/town limits",
            },
            "private_cab": {
                "available": True,
                "transit_type": cab_type,
                "total_fare_inr": cab_fare,
                "notes": f"Full private vehicle drop for {pax} passenger(s)",
            },
        },
    }


def find_off_grid_destination(dest_name: str, passengers: int = 1) -> dict[str, Any] | None:
    """Identify gateway hub and calculate dynamic multi-leg road options for ANY destination."""
    if not dest_name:
        return None
    cleaned = dest_name.lower().strip()
    pax = max(1, int(passengers))

    gateway_data = None

    # 1. Check in external reference dataset
    for key, data in REFERENCE_GATEWAYS.items():
        if key in cleaned or cleaned in key:
            gateway_data = data
            break

    # 2. Check in dynamic cache
    if not gateway_data:
        for key, data in DYNAMIC_GATEWAY_CACHE.items():
            if key in cleaned or cleaned in key:
                gateway_data = data
                break

    # 3. Dynamic AI Discovery for any unlisted place (e.g. Horsley Hills, Araku Valley, Coorg)
    if not gateway_data:
        discovered = _resolve_gateway_via_llm(cleaned)
        if discovered:
            DYNAMIC_GATEWAY_CACHE[cleaned] = discovered
            _save_dynamic_cache()
            gateway_data = discovered

    if not gateway_data:
        return None

    # Calculate dynamic pricing from distances
    direct_cab_info = calculate_generic_road_transit(
        from_place=gateway_data.get("primary_rail_hub", "Railway Station"),
        to_place=gateway_data.get("destination_name", dest_name),
        approx_distance_km=float(gateway_data.get("direct_cab_distance_km", 50.0)),
        passengers=pax,
    )
    dist_total = float(direct_cab_info.get("distance_km") or gateway_data.get("direct_cab_distance_km", 50.0))

    # Format legs with dynamically calculated duration and cost models
    computed_legs = []
    for l in gateway_data.get("legs", []):
        transit_calc = calculate_generic_road_transit(
            from_place=l.get("from_point", "Origin"),
            to_place=l.get("to_point", "Destination"),
            approx_distance_km=float(l.get("distance_km", 10.0)),
            passengers=pax,
        )
        l_dist = float(transit_calc.get("distance_km") or l.get("distance_km", 10.0))
        computed_legs.append({
            "leg_name": l.get("leg_name", "Local Transfer"),
            "mode": l.get("mode", "ROAD_TRANSIT"),
            "transit_type": l.get("transit_type", "Local Bus / Auto"),
            "from_point": l.get("from_point"),
            "to_point": l.get("to_point"),
            "distance_km": l_dist,
            "duration_formatted": transit_calc.get("estimated_duration_formatted"),
            "duration_minutes": transit_calc.get("estimated_duration_minutes"),
            "cost_per_passenger_inr": transit_calc.get("options", {}).get("bus", {}).get("fare_per_passenger_inr", 50),
            "base_cost_inr": transit_calc.get("options", {}).get("auto", {}).get("total_fare_inr", 150),
            "notes": l.get("notes", "Locally available transit"),
            "route_source": transit_calc.get("route_source"),
        })

    return {
        "destination_name": gateway_data.get("destination_name", dest_name),
        "primary_rail_hub": gateway_data.get("primary_rail_hub"),
        "rail_station_code": gateway_data.get("rail_station_code"),
        "intermediate_bus_hub": gateway_data.get("intermediate_bus_hub"),
        "legs": computed_legs,
        "direct_station_cab": {
            "mode": "DIRECT_CAB",
            "transit_type": direct_cab_info.get("options", {}).get("private_cab", {}).get("transit_type", "Station Taxi"),
            "from_point": gateway_data.get("primary_rail_hub"),
            "to_point": gateway_data.get("destination_name"),
            "distance_km": dist_total,
            "duration_formatted": direct_cab_info.get("estimated_duration_formatted"),
            "duration_minutes": direct_cab_info.get("estimated_duration_minutes"),
            "sedan_fare_inr": int(dist_total * 18.0 + 200),
            "suv_innova_fare_inr": int(dist_total * 24.0 + 300),
            "notes": "Direct private vehicle drop from station to destination.",
        },
    }
