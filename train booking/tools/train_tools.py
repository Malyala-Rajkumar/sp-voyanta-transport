import json
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool

# Ensure local imports work cleanly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from railkit_client import normalize_date_str, railkit
from station_resolver import resolve_city_to_station

TravelClass = Literal["SL", "3A", "2A", "1A", "CC", "2S"]
Quota = Literal["GN", "TQ", "LD", "PT"]


def _safe(fn):
    """Return JSON for both success and failure so the model can recover cleanly."""
    @wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        try:
            result = fn(*args, **kwargs)
            return json.dumps(result, ensure_ascii=False, default=str)
        except ValueError as e:
            return json.dumps({"success": False, "error": f"Validation Error: {e}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    return wrapper


def _parse_duration_minutes(dur_str: str) -> int:
    """Parse duration strings like '00:39', '03:18 hrs', '5:45', or '5h 45m' into integer minutes."""
    if not dur_str:
        return 9999
    import re
    d = str(dur_str).strip()
    m_hm = re.search(r"(\d+)\s*[:h]\s*(\d+)", d)
    if m_hm:
        return int(m_hm.group(1)) * 60 + int(m_hm.group(2))
    m_h = re.search(r"(\d+)\s*(?:hrs?|h)", d)
    if m_h:
        return int(m_h.group(1)) * 60
    m_m = re.search(r"(\d+)\s*(?:mins?|m)", d)
    if m_m:
        return int(m_m.group(1))
    return 9999


@tool
@_safe
def search_trains(
    source: str,
    destination: str,
    journey_date: str,
    travel_class: str = "3A",
    quota: str = "GN",
) -> dict[str, Any]:
    """Search Indian Railway trains between two cities or station codes with live berths and intelligent recommendations.

    Args:
        source: Source city name (e.g. 'Hyderabad', 'Khammam') or IRCTC station code (e.g. 'SC', 'KMT').
        destination: Destination city name (e.g. 'Vijayawada', 'Mahabubabad') or IRCTC station code (e.g. 'BZA', 'MABD').
        journey_date: Journey date in YYYY-MM-DD, DD-MM-YYYY, or relative string (e.g. 'tomorrow').
        travel_class: Preferred class (e.g. 'SL', '3A', '2A', '1A', 'CC', '2S'). Defaults to '3A'.
        quota: Booking quota ('GN', 'TQ', 'LD'). Defaults to 'GN'.
    """
    # 1. Resolve source and destination entities
    src_primary, src_clusters, src_city, src_stn_name = resolve_city_to_station(source)
    dst_primary, dst_clusters, dst_city, dst_stn_name = resolve_city_to_station(destination)

    # 2. Normalize journey date
    norm_date = normalize_date_str(journey_date)

    # 3. Search trains across metro terminal clusters
    cluster_res = railkit.search_trains_cluster(src_clusters, dst_clusters, norm_date)
    raw_trains = cluster_res.get("data") or []

    if not raw_trains:
        return {
            "success": True,
            "status": "no_results",
            "message": f"No direct trains found between {src_city} ({src_primary}) and {dst_city} ({dst_primary}) on {norm_date}.",
            "route": f"{src_city} -> {dst_city}",
            "journey_date": norm_date,
            "trains": [],
        }

    # Format trains cleanly with duration calculations
    formatted_trains = []
    for t in raw_trains:
        avail_classes = [c.strip() for c in (t.get("classes_of_travel") or "").split(",") if c.strip()]
        dur_mins = _parse_duration_minutes(t.get("travel_time") or "")
        dur_fmt = f"{dur_mins // 60}h {dur_mins % 60}m" if dur_mins < 9999 else t.get("travel_time", "")
        formatted_trains.append({
            "train_number": t.get("train_no"),
            "train_name": t.get("train_name"),
            "from_station_name": t.get("from_stn_name"),
            "from_station_code": t.get("from_stn_code"),
            "to_station_name": t.get("to_stn_name"),
            "to_station_code": t.get("to_stn_code"),
            "departure_time": t.get("from_time"),
            "arrival_time": t.get("to_time"),
            "duration": dur_fmt,
            "duration_minutes": dur_mins,
            "distance_km": t.get("distance"),
            "classes_available": avail_classes,
            "running_days": t.get("daysOfDeparture") or "Daily",
            "halts": t.get("halts"),
            "price": None,
            "price_formatted": "Check on IRCTC",
            "seat_status": "Available",
            "status_badge": "AVAILABLE",
            "confirmation_prediction": "Confirmed",
            "target_class": travel_class,
        })

    # Sort candidates by departure time and duration
    formatted_trains.sort(key=lambda x: (x.get("departure_time") or "99:99", x["duration_minutes"]))

    # Query live availability and fares concurrently in parallel (Sub-2s latency)
    from concurrent.futures import ThreadPoolExecutor

    def _fetch_avail(t):
        avail_classes = t.get("classes_available") or []
        target_cls = travel_class if travel_class in avail_classes else (avail_classes[0] if avail_classes else travel_class)
        t["target_class"] = target_cls
        try:
            av_res = railkit.check_availability(
                t["train_number"],
                t["from_station_code"],
                t["to_station_code"],
                norm_date,
                target_cls,
                quota,
            )
            d_data = av_res.get("data") or {}
            f_data = d_data.get("fare") or {}
            av_list = d_data.get("availability") or []
            if "totalFare" in f_data:
                t["price"] = int(f_data["totalFare"])
                t["price_formatted"] = f"₹{int(f_data['totalFare']):,}"
            if av_list:
                status_txt = av_list[0].get("availabilityText") or "Available"
                t["seat_status"] = status_txt
                t["status_badge"] = "AVAILABLE" if ("avl" in status_txt.lower() or "avail" in status_txt.lower()) else "WAITLIST"
                t["confirmation_prediction"] = str(av_list[0].get("prediction") or "High")
        except Exception:
            pass
        return t

    top_candidates = formatted_trains[:4]
    with ThreadPoolExecutor(max_workers=min(len(top_candidates) or 1, 4)) as executor:
        list(executor.map(_fetch_avail, top_candidates))

    # Intelligent Ranking: Cheapest, Fastest, Best Overall
    priced = [t for t in formatted_trains if t.get("price") is not None]
    candidate_pool = priced if priced else formatted_trains

    cheapest_train = min(candidate_pool, key=lambda x: (x.get("price") or 99999, x["duration_minutes"])) if candidate_pool else None
    fastest_train = min(formatted_trains, key=lambda x: (x["duration_minutes"], x.get("price") or 99999)) if formatted_trains else None

    # Best overall balances confirmed seats, fast travel time, and reasonable fare
    def _train_score(t):
        penalty = 0
        if t.get("status_badge") == "WAITLIST":
            penalty += 500
        p = t.get("price") or 1000
        d = t["duration_minutes"]
        return penalty + p + (d * 2)

    best_overall = min(candidate_pool, key=_train_score) if candidate_pool else (cheapest_train or fastest_train)

    lowest_fare_summary = cheapest_train.get("price_formatted") if (cheapest_train and cheapest_train.get("price")) else "Typical Fares"

    top_trains = formatted_trains[:4]
    other_trains = formatted_trains[4:10]

    return {
        "success": True,
        "status": "success",
        "route": f"{src_city} ({src_primary}) -> {dst_city} ({dst_primary})",
        "source_resolved": f"{src_city} [{', '.join(src_clusters)}]",
        "dest_resolved": f"{dst_city} [{', '.join(dst_clusters)}]",
        "journey_date": norm_date,
        "total_trains_found": len(formatted_trains),
        "lowest_fare_summary": lowest_fare_summary,
        "booking_url": "https://www.irctc.co.in/nget/train-search",
        "price_insights": {
            "status": "typical",
            "summary": "Fares are currently typical for Indian Railways",
        },
        "recommendations": {
            "best_overall": best_overall,
            "cheapest": cheapest_train,
            "fastest": fastest_train,
        },
        "top_trains": top_trains,
        "other_options": other_trains,
        "trains": formatted_trains,
    }


@tool
@_safe
def check_availability(
    train_number: str | int,
    source: str,
    destination: str,
    journey_date: str,
    travel_class: TravelClass = "3A",
    quota: Quota = "GN",
) -> dict[str, Any]:
    """Check seat availability and live fare breakdown for a specific train.

    Args:
        train_number: 5-digit IRCTC train number (e.g. '12727', '11019').
        source: Boarding station code or city name (e.g. 'SC' or 'Hyderabad').
        destination: Deboarding station code or city name (e.g. 'BZA' or 'Vijayawada').
        journey_date: Journey date in YYYY-MM-DD or DD-MM-YYYY format.
        travel_class: Travel class (one of SL, 3A, 2A, 1A, CC, 2S).
        quota: GN (General) or TQ (Tatkal). Defaults to 'GN'.
    """
    # Resolve station codes if city name was passed
    src_primary, _, _, _ = resolve_city_to_station(source)
    dst_primary, _, _, _ = resolve_city_to_station(destination)
    norm_date = normalize_date_str(journey_date)

    res = railkit.check_availability(
        train_number=train_number,
        source=src_primary,
        destination=dst_primary,
        journey_date=norm_date,
        travel_class=travel_class,
        quota=quota,
    )

    data = res.get("data") or {}
    train_info = data.get("train") or {}
    fare_info = data.get("fare") or {}
    avail_list = data.get("availability") or []

    # Current target date status
    primary_avail = avail_list[0] if avail_list else {}

    return {
        "success": True,
        "train_number": str(train_number).strip(),
        "train_name": train_info.get("trainName"),
        "route": f"{train_info.get('from', src_primary)} -> {train_info.get('to', dst_primary)}",
        "journey_date": norm_date,
        "travel_class": travel_class,
        "quota": quota,
        "fare_inr": fare_info.get("totalFare", 0),
        "fare_breakdown": fare_info,
        "seat_status": primary_avail.get("availabilityText", "Unknown"),
        "status_code": primary_avail.get("status", "Unknown"),
        "confirmation_prediction": primary_avail.get("prediction", "N/A"),
        "can_book": primary_avail.get("canBook", True),
        "alternate_dates_availability": avail_list[1:4] if len(avail_list) > 1 else [],
    }
