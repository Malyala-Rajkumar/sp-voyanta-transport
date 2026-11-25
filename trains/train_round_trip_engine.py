"""Round-Trip Train Engine & Flexible Window Planner.

Computes return dates, iterates over search windows, pairs outbound and return trains,
checks seat availability/fares, and outputs the Itinerary Team's standardized payload.
Matches the exact schema and interface used in flights/round_trip_engine.py.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from railkit_client import normalize_date_str, railkit
from station_resolver import resolve_city_to_station

IST = ZoneInfo("Asia/Kolkata")


def calculate_return_date(departure_date: date, trip_duration_days: int) -> date:
    """Calculate return date based on stay duration."""
    if trip_duration_days <= 1:
        return departure_date
    return departure_date + timedelta(days=trip_duration_days)


def _to_date(val: Any) -> date:
    """Convert string or date object into datetime.date."""
    if isinstance(val, (datetime, date)):
        return val if isinstance(val, date) else val.date()
    s = str(val).strip()
    # Try YYYY-MM-DD
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        pass
    # Try DD-MM-YYYY
    try:
        return datetime.strptime(s, "%d-%m-%Y").date()
    except ValueError:
        pass
    return datetime.now(IST).date()


def _build_train_segment(
    train_record: dict,
    avail_record: dict | None,
    travel_class: str = "3A",
    quota: str = "GN",
    mode: str = "train",
) -> dict[str, Any]:
    """Convert an IRCTC train option into the Itinerary Team's standardized segment schema."""
    if not train_record:
        return {}

    fare = 0
    seat_status = "Available / Check RailKit"
    if avail_record:
        fare = avail_record.get("fare_inr") or 0
        seat_status = f"{avail_record.get('seat_status', '')} ({avail_record.get('confirmation_prediction', '')})".strip()

    t_no = train_record.get("train_no") or train_record.get("train_number") or ""
    t_name = train_record.get("train_name") or ""
    from_name = train_record.get("from_stn_name") or train_record.get("from_station_name") or train_record.get("from_stn_code") or ""
    to_name = train_record.get("to_stn_name") or train_record.get("to_station_name") or train_record.get("to_stn_code") or ""
    dep_t = train_record.get("from_time") or train_record.get("departure_time") or ""
    arr_t = train_record.get("to_time") or train_record.get("arrival_time") or ""
    dur = train_record.get("travel_time") or train_record.get("duration") or ""

    return {
        "mode": mode,
        "carrier": f"Indian Railways ({t_name.title()})",
        "train_number": t_no,
        "from_place": from_name,
        "to_place": to_name,
        "departure_time": dep_t,
        "arrival_time": arr_t,
        "duration": dur,
        "travel_class": travel_class,
        "quota": quota,
        "seat_status": seat_status,
        "cost_inr": int(fare),
        "details": f"Direct train journey via {t_name.title()} ({travel_class})",
        "url": "https://www.irctc.co.in",
        "source": "railkit_api",
    }


def plan_round_trip_trains(
    source: str,
    destination: str,
    start_date: Any,
    trip_duration_days: int = 5,
    search_window_days: int = 3,
    passengers: int = 1,
    travel_class: str = "3A",
    quota: str = "GN",
    budget: float | None = None,
) -> dict[str, Any]:
    """Execute a round-trip train search across a flexible window of departure dates.

    Matches the exact input signature and output schema of flights/round_trip_engine.py.
    Produces:
    - journey: { onward, return, total_inr, recommendation, alternatives }
    - all_date_options: list of candidate departure date packages in the search window.
    """
    # 1. Resolve cities / stations
    try:
        src_primary, src_clusters, src_city, src_stn_name = resolve_city_to_station(source)
        dst_primary, dst_clusters, dst_city, dst_stn_name = resolve_city_to_station(destination)
    except ValueError as ve:
        return {
            "status": "error",
            "message": str(ve),
        }

    dep_date_obj = _to_date(start_date)
    duration = max(1, int(trip_duration_days))
    window = max(1, min(int(search_window_days), 7))

    date_options = []

    def _evaluate_offset(offset: int) -> dict | None:
        curr_dep_date = dep_date_obj + timedelta(days=offset)
        curr_ret_date = calculate_return_date(curr_dep_date, duration)

        dep_str = normalize_date_str(curr_dep_date)
        ret_str = normalize_date_str(curr_ret_date)

        # Search Outbound & Return Trains in parallel concurrently
        out_trains, ret_trains = [], []
        with ThreadPoolExecutor(max_workers=2) as sub_exec:
            f_out = sub_exec.submit(railkit.search_trains_cluster, src_clusters, dst_clusters, dep_str)
            f_ret = sub_exec.submit(railkit.search_trains_cluster, dst_clusters, src_clusters, ret_str)
            try:
                out_trains = f_out.result().get("data") or []
            except Exception:
                pass
            try:
                ret_trains = f_ret.result().get("data") or []
            except Exception:
                pass

        if not out_trains or not ret_trains:
            return None

        # Pick top outbound & top return train
        best_out = out_trains[0]
        best_ret = ret_trains[0]

        # Check live availability & fares on top trains concurrently in parallel
        def _get_out_avail():
            try:
                av = railkit.check_availability(
                    best_out.get("train_no"),
                    best_out.get("from_stn_code", src_primary),
                    best_out.get("to_stn_code", dst_primary),
                    dep_str,
                    travel_class,
                    quota,
                )
                return av.get("data") or {}
            except Exception:
                return {}

        def _get_ret_avail():
            try:
                av = railkit.check_availability(
                    best_ret.get("train_no"),
                    best_ret.get("from_stn_code", dst_primary),
                    best_ret.get("to_stn_code", src_primary),
                    ret_str,
                    travel_class,
                    quota,
                )
                return av.get("data") or {}
            except Exception:
                return {}

        with ThreadPoolExecutor(max_workers=2) as avail_exec:
            f_av_out = avail_exec.submit(_get_out_avail)
            f_av_ret = avail_exec.submit(_get_ret_avail)
            d_out = f_av_out.result()
            d_ret = f_av_ret.result()

        out_fare = None
        ret_fare = None
        f_data_out = d_out.get("fare") or {}
        if "totalFare" in f_data_out:
            out_fare = int(f_data_out["totalFare"])
        av_list_out = d_out.get("availability") or []
        out_avail = {
            "fare_inr": out_fare or 0,
            "seat_status": av_list_out[0].get("availabilityText", "Available") if av_list_out else "Available",
            "confirmation_prediction": av_list_out[0].get("prediction", "High") if av_list_out else "Confirmed",
        }

        f_data_ret = d_ret.get("fare") or {}
        if "totalFare" in f_data_ret:
            ret_fare = int(f_data_ret["totalFare"])
        av_list_ret = d_ret.get("availability") or []
        ret_avail = {
            "fare_inr": ret_fare or 0,
            "seat_status": av_list_ret[0].get("availabilityText", "Available") if av_list_ret else "Available",
            "confirmation_prediction": av_list_ret[0].get("prediction", "High") if av_list_ret else "Confirmed",
        }

        # Calculate group fare from verified amounts
        verified_out = out_fare if out_fare is not None else 0
        verified_ret = ret_fare if ret_fare is not None else 0
        total_person_fare = verified_out + verified_ret
        total_group_fare = total_person_fare * int(passengers)

        meets_budget = True
        budget_reason = None
        if budget is not None and total_group_fare > budget:
            meets_budget = False
            budget_reason = f"Exceeds budget of ₹{int(budget):,} by ₹{total_group_fare - int(budget):,}"

        return {
            "option_id": offset + 1,
            "departure_date": curr_dep_date.isoformat(),
            "departure_date_formatted": curr_dep_date.strftime("%a, %b %d, %Y"),
            "return_date": curr_ret_date.isoformat(),
            "return_date_formatted": curr_ret_date.strftime("%a, %b %d, %Y"),
            "stay_duration_days": duration,
            "outbound_train": best_out,
            "outbound_avail": out_avail,
            "return_train": best_ret,
            "return_avail": ret_avail,
            "outbound_fare_person": int(out_fare or 0),
            "return_fare_person": int(ret_fare or 0),
            "total_round_trip_person": total_person_fare,
            "total_round_trip_inr": total_group_fare,
            "total_round_trip_formatted": f"₹{total_group_fare:,}",
            "meets_budget": meets_budget,
            "budget_reason": budget_reason,
        }

    # Execute all window offsets concurrently in parallel
    with ThreadPoolExecutor(max_workers=min(window, 4)) as win_exec:
        futures = [win_exec.submit(_evaluate_offset, off) for off in range(window)]
        for f in futures:
            res = f.result()
            if res:
                date_options.append(res)

    if not date_options:
        return {
            "status": "no_results",
            "message": f"No matching direct round-trip train combinations found for {src_city} -> {dst_city} across the {window}-day departure window.",
            "source_location": source,
            "destination_location": destination,
            "start_date": dep_date_obj.isoformat(),
        }

    # Rank candidate options: budget conforming first, then cheapest total fare
    budget_conforming = [o for o in date_options if o["meets_budget"]]
    pool = budget_conforming if budget_conforming else date_options
    pool.sort(key=lambda x: x["total_round_trip_inr"])
    best_option = pool[0]

    # Build standardized segments
    out_seg = _build_train_segment(
        best_option["outbound_train"],
        best_option["outbound_avail"],
        travel_class=travel_class,
        quota=quota,
    )
    ret_seg = _build_train_segment(
        best_option["return_train"],
        best_option["return_avail"],
        travel_class=travel_class,
        quota=quota,
    )

    # Standardized Itinerary Team Payload
    primary_journey = {
        "onward": {
            "route": f"{src_city}, India -> {dst_city}, India",
            "date": best_option["departure_date"],
            "departure_date_formatted": best_option["departure_date_formatted"],
            "train_summary": f"{best_option['outbound_train'].get('train_name')} (#{best_option['outbound_train'].get('train_no')})",
            "segments": [out_seg],
            "total_inr": best_option["outbound_fare_person"] * int(passengers),
            "booking_url": "https://www.irctc.co.in",
        },
        "return": {
            "route": f"{dst_city}, India -> {src_city}, India",
            "date": best_option["return_date"],
            "return_date_formatted": best_option["return_date_formatted"],
            "train_summary": f"{best_option['return_train'].get('train_name')} (#{best_option['return_train'].get('train_no')})",
            "segments": [ret_seg],
            "total_inr": best_option["return_fare_person"] * int(passengers),
            "booking_url": "https://www.irctc.co.in",
        },
        "total_inr": best_option["total_round_trip_inr"],
        "recommendation": (
            f"**Recommended Mode:** Indian Railways Express ({travel_class}) ≈ {best_option['total_round_trip_formatted']} round-trip for {passengers} traveler(s).\n\n"
            f"• **Onward Journey ({best_option['departure_date_formatted']}):** {best_option['outbound_train'].get('train_name')} (#{best_option['outbound_train'].get('train_no')}) departing at {best_option['outbound_train'].get('from_time')} ({best_option['outbound_train'].get('travel_time')}).\n"
            f"• **Return Journey ({best_option['return_date_formatted']}):** {best_option['return_train'].get('train_name')} (#{best_option['return_train'].get('train_no')}) departing at {best_option['return_train'].get('from_time')} ({best_option['return_train'].get('travel_time')}).\n"
            f"• **Seat & Fare Insight:** Highly budget-friendly travel leaving significant budget for local hotels and dining."
        ),
        "alternatives": [
            {
                "mode": "train",
                "departure_date": opt["departure_date_formatted"],
                "return_date": opt["return_date_formatted"],
                "total_inr": opt["total_round_trip_inr"],
                "note": f"Option {opt['option_id']} · Total {opt['total_round_trip_formatted']}",
            }
            for opt in date_options[1:4]
        ],
    }

    return {
        "status": "success",
        "search_parameters": {
            "source_location": source,
            "resolved_source": f"{src_city} ({src_primary})",
            "destination_location": destination,
            "resolved_destination": f"{dst_city} ({dst_primary})",
            "start_date": dep_date_obj.isoformat(),
            "trip_duration_days": duration,
            "search_window_days": window,
            "passengers": passengers,
            "travel_class": travel_class,
            "quota": quota,
            "budget": budget,
        },
        "cheapest_package": {
            "departure_date": best_option["departure_date_formatted"],
            "return_date": best_option["return_date_formatted"],
            "total_fare": best_option["total_round_trip_formatted"],
        },
        "journey": primary_journey,
        "all_date_options": date_options,
    }
