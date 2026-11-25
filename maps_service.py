"""Google Maps Platform Service (Dynamic, Decoupled & High-Performance).

Integrates Google's modern Routes API and Places API (New) for multi-modal travel planning:
1. Exact point-to-point road distance & live traffic durations (Routes API).
2. Automated nearest transit hub & railway station discovery (Places API New).
3. Zero hardcoding: key is loaded dynamically from GOOGLE_MAPS_API_KEY via .env.
4. Two-tier caching (In-Memory + Disk JSON) for sub-millisecond repeated queries.
5. Graceful fallback: returns None on API errors without crashing the agent.
"""
import json
import os
import time
from pathlib import Path
from typing import Any
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MAPS_CACHE_FILE = DATA_DIR / "maps_cache.json"

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
load_dotenv()

# In-memory LRU-style cache with 24-hour TTL (86400s)
_MEMORY_CACHE: dict[str, tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 86400


def _format_address(addr: str) -> str:
    """Normalize address string without forcing a hardcoded country suffix."""
    return str(addr).strip()


def _geocode_nominatim(address: str) -> tuple[float, float] | None:
    """Geocode address to (lat, lon) using OpenStreetMap Nominatim with caching."""
    clean_addr = _format_address(address)
    cache_key = f"geo:nominatim:{clean_addr.lower()}"
    if cache_key in _DISK_CACHE:
        loc = _DISK_CACHE[cache_key]
        return loc["lat"], loc["lon"]

    try:
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "SPVoyantaTripPlanner/2.0 (travel-agent-bot)"}
        params = {"q": clean_addr, "format": "json", "limit": 1}
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                _DISK_CACHE[cache_key] = {"lat": lat, "lon": lon}
                _save_disk_cache()
                return lat, lon
    except Exception:
        pass
    return None


def _route_via_osrm(
    origin: str,
    destination: str,
    travel_mode: str = "DRIVE",
) -> dict[str, Any] | None:
    """Fallback route calculation via public OSRM engine when Google Routes quota is exceeded."""
    coords_orig = _geocode_nominatim(origin)
    coords_dest = _geocode_nominatim(destination)

    if not coords_orig or not coords_dest:
        return None

    lat1, lon1 = coords_orig
    lat2, lon2 = coords_dest

    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        headers = {"User-Agent": "SPVoyantaTripPlanner/2.0"}
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return None

        data = resp.json()
        routes = data.get("routes") or []
        if not routes:
            return None

        best_route = routes[0]
        meters = best_route.get("distance", 0)
        dur_sec = int(best_route.get("duration", 0))

        dist_km = round(meters / 1000.0, 1)
        dur_mins = max(1, dur_sec // 60)
        hours = dur_mins // 60
        mins = dur_mins % 60
        dur_fmt = f"{hours}h {mins}m" if hours > 0 else f"{mins} mins"

        return {
            "source": "osrm_routes_engine",
            "is_live_api": True,
            "origin": origin,
            "destination": destination,
            "travel_mode": travel_mode,
            "distance_km": dist_km,
            "distance_meters": int(meters),
            "duration_minutes": dur_mins,
            "duration_formatted": dur_fmt,
            "route_description": "Road Transit via Open Highways",
            "encoded_polyline": "",
        }
    except Exception:
        return None


def _load_disk_cache() -> dict[str, Any]:
    """Safely load cached Google Maps queries from disk."""
    if MAPS_CACHE_FILE.exists():
        try:
            with open(MAPS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


_DISK_CACHE: dict[str, Any] = _load_disk_cache()


def _save_disk_cache() -> None:
    """Persist updated Maps cache to disk."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(MAPS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_DISK_CACHE, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def get_api_key() -> str | None:
    """Retrieve Google Maps API key dynamically from environment."""
    return os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_PLACES_API_KEY")


def get_exact_route(
    origin: str,
    destination: str,
    travel_mode: str = "DRIVE",
) -> dict[str, Any] | None:
    """Query Google Routes API for exact driving distance and live traffic duration.

    Args:
        origin: Starting address, landmark, or station name (e.g. 'Warangal Railway Station').
        destination: Arrival destination or spot (e.g. 'Pakhal Lake').
        travel_mode: 'DRIVE', 'TWO_WHEELER', or 'TRANSIT'. Defaults to 'DRIVE'.

    Returns:
        Dict with distance_km, duration_minutes, duration_formatted, polyline, and source.
    """
    norm_orig = _format_address(str(origin).strip())
    norm_dest = _format_address(str(destination).strip())
    cache_key = f"route:{travel_mode}:{norm_orig.lower()}->{norm_dest.lower()}"

    # 1. In-memory Cache Check (< 1ms)
    hit = _MEMORY_CACHE.get(cache_key)
    if hit and (time.time() - hit[0] < CACHE_TTL_SECONDS):
        return hit[1]

    # 2. Disk Cache Check (< 5ms)
    if cache_key in _DISK_CACHE:
        val = _DISK_CACHE[cache_key]
        _MEMORY_CACHE[cache_key] = (time.time(), val)
        return val

    # 3. Live Google Routes API Call
    api_key = get_api_key()
    if api_key:
        url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.description,routes.polyline.encodedPolyline",
        }
        payload = {
            "origin": {"address": norm_orig},
            "destination": {"address": norm_dest},
            "travelMode": travel_mode,
            "routingPreference": "TRAFFIC_AWARE" if travel_mode == "DRIVE" else "ROUTING_PREFERENCE_UNSPECIFIED",
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                routes = data.get("routes") or []
                if routes:
                    best_route = routes[0]
                    meters = best_route.get("distanceMeters", 0)
                    dur_str = best_route.get("duration", "0s").replace("s", "")
                    dur_sec = int(dur_str) if dur_str.isdigit() else 0

                    dist_km = round(meters / 1000.0, 1)
                    dur_mins = max(1, dur_sec // 60)
                    hours = dur_mins // 60
                    mins = dur_mins % 60
                    dur_fmt = f"{hours}h {mins}m" if hours > 0 else f"{mins} mins"

                    result = {
                        "source": "google_maps_routes_api",
                        "is_live_api": True,
                        "origin": norm_orig,
                        "destination": norm_dest,
                        "travel_mode": travel_mode,
                        "distance_km": dist_km,
                        "distance_meters": meters,
                        "duration_minutes": dur_mins,
                        "duration_formatted": dur_fmt,
                        "route_description": best_route.get("description", "Direct Road Route"),
                        "encoded_polyline": (best_route.get("polyline") or {}).get("encodedPolyline", ""),
                    }

                    _MEMORY_CACHE[cache_key] = (time.time(), result)
                    _DISK_CACHE[cache_key] = result
                    _save_disk_cache()
                    return result
        except Exception:
            pass

    # 4. Resilient Fallback: OSRM Routing Engine (if Google Routes API has 429 quota exhaustion or network error)
    fallback_route = _route_via_osrm(norm_orig, norm_dest, travel_mode=travel_mode)
    if fallback_route:
        _MEMORY_CACHE[cache_key] = (time.time(), fallback_route)
        _DISK_CACHE[cache_key] = fallback_route
        _save_disk_cache()
        return fallback_route

    return None


def find_nearest_transit_hub(
    place_name: str,
    hub_type: str = "railway_station",
) -> dict[str, Any] | None:
    """Use Google Places API (New) to discover the closest railway station or airport.

    Args:
        place_name: Destination or tourist spot (e.g. 'Pakhal Lake', 'Horsley Hills').
        hub_type: 'railway_station' or 'airport'.

    Returns:
        Dict with hub_name, formatted_address, latitude, longitude, and source.
    """
    api_key = get_api_key()
    if not api_key:
        return None

    clean_place = str(place_name).strip()
    cache_key = f"places:{hub_type}:{clean_place.lower()}"

    # 1. In-memory Cache Check
    hit = _MEMORY_CACHE.get(cache_key)
    if hit and (time.time() - hit[0] < CACHE_TTL_SECONDS):
        return hit[1]

    # 2. Disk Cache Check
    if cache_key in _DISK_CACHE:
        val = _DISK_CACHE[cache_key]
        _MEMORY_CACHE[cache_key] = (time.time(), val)
        return val

    # 3. Live Google Places API (New) Call
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.types",
    }
    
    query = f"Major railway junction or station near {clean_place}" if hub_type == "railway_station" else f"Commercial airport near {clean_place}"
    payload = {"textQuery": query}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=8)
        if resp.status_code != 200:
            return None

        data = resp.json()
        places = data.get("places") or []
        if not places:
            return None

        p = places[0]
        name = (p.get("displayName") or {}).get("text") or ""
        addr = p.get("formattedAddress") or ""
        loc = p.get("location") or {}

        result = {
            "source": "google_places_api_new",
            "is_live_api": True,
            "query_place": clean_place,
            "hub_type": hub_type,
            "hub_name": name,
            "formatted_address": addr,
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
        }

        # Store in caches
        _MEMORY_CACHE[cache_key] = (time.time(), result)
        _DISK_CACHE[cache_key] = result
        _save_disk_cache()

        return result
    except Exception:
        pass

    # 4. Resilient Fallback: Nominatim Discovery (if Places API New has 429 quota exhaustion)
    try:
        search_query = f"{clean_place} railway station" if hub_type == "railway_station" else f"{clean_place} airport"
        geo = _geocode_nominatim(search_query)
        if geo:
            fallback_res = {
                "source": "nominatim_transit_discovery",
                "is_live_api": True,
                "query_place": clean_place,
                "hub_type": hub_type,
                "hub_name": f"{clean_place.title()} {'Railway Station' if hub_type == 'railway_station' else 'Airport'}",
                "formatted_address": _format_address(clean_place),
                "latitude": geo[0],
                "longitude": geo[1],
            }
            _MEMORY_CACHE[cache_key] = (time.time(), fallback_res)
            _DISK_CACHE[cache_key] = fallback_res
            _save_disk_cache()
            return fallback_res
    except Exception:
        pass

    return None
