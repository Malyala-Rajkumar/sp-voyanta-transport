"""Round-Trip Flight Engine & Flexible Window Planner.

Computes return dates, iterates over search windows, pairs outbound and return
flights, and outputs the Itinerary Team's standardized payload.
"""
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any

from city_resolver import resolve_city_to_airport

# Import flight fetching logic from flight_v4
from flight_v4 import _fetch_options, evaluate_flights, fmt_minutes, fmt_time_12h, parse_dt


def calculate_return_date(departure_date: date, trip_duration_days: int) -> date:
    """Calculate the return date based on trip duration.

    If trip_duration_days == 0 or 1 (same day trip), return the same departure date.
    Otherwise, return departure_date + trip_duration_days.
    """
    if trip_duration_days <= 1:
        # Same-day trip
        return departure_date
    return departure_date + timedelta(days=trip_duration_days)


def _build_segments(flight_option: dict, mode: str = "flight") -> list[dict]:
    """Convert a normalized flight itinerary into the Itinerary Team's segment schema."""
    if not flight_option:
        return []

    # If layovers exist, return multiple segments, else single segment
    layovers = flight_option.get("layovers") or []
    flight_nums = flight_option.get("flight_numbers") or [flight_option.get("flight_number", "")]
    airlines = flight_option.get("airlines") or [flight_option.get("airline", "Airline")]

    # Single segment if direct
    if not layovers:
        return [
            {
                "mode": mode,
                "carrier": flight_option.get("airline"),
                "flight_number": flight_option.get("flight_number"),
                "from_place": flight_option.get("departure_airport"),
                "to_place": flight_option.get("arrival_airport"),
                "departure_time": flight_option.get("departure_time"),
                "arrival_time": flight_option.get("arrival_time"),
                "duration": flight_option.get("duration_formatted"),
                "cost_inr": int(flight_option.get("price") or 0),
                "details": f"Non-stop commercial flight via {flight_option.get('airline')}",
                "url": flight_option.get("booking_url", ""),
                "source": "google_flights_api",
            }
        ]

    # Multi-leg connecting flights: Decompose into individual flight segments
    raw_legs = flight_option.get("legs") or []
    if raw_legs and len(raw_legs) > 1:
        segments = []
        for idx, leg in enumerate(raw_legs):
            dep_info = leg.get("departure_airport") or {}
            arr_info = leg.get("arrival_airport") or {}
            dep_t = fmt_time_12h(parse_dt(dep_info.get("time")))
            arr_t = fmt_time_12h(parse_dt(arr_info.get("time")))
            leg_dur = fmt_minutes(leg.get("duration") or 0)
            from_p = dep_info.get("name") or dep_info.get("id") or flight_option.get("departure_airport")
            to_p = arr_info.get("name") or arr_info.get("id") or flight_option.get("arrival_airport")

            # Layover note if not the final leg
            if idx < len(layovers):
                lo = layovers[idx]
                details = f"Leg {idx+1}/{len(raw_legs)}: {leg.get('airline', 'Airline')} ({leg.get('flight_number', '')}) to {to_p}. Layover: {lo.get('duration_formatted')} at {lo.get('transit_airport')}"
            else:
                details = f"Leg {idx+1}/{len(raw_legs)} (Final): {leg.get('airline', 'Airline')} ({leg.get('flight_number', '')}) arriving at {to_p}"

            segments.append({
                "mode": mode,
                "carrier": leg.get("airline") or flight_option.get("airline"),
                "flight_number": leg.get("flight_number") or flight_option.get("flight_number"),
                "from_place": from_p,
                "to_place": to_p,
                "departure_time": dep_t,
                "arrival_time": arr_t,
                "duration": leg_dur,
                "cost_inr": int(flight_option.get("price") or 0) if idx == 0 else 0,  # ticket covers all legs
                "details": details,
                "url": flight_option.get("booking_url", ""),
                "source": "google_flights_api",
            })
        return segments

    # Fallback single multi-stop segment representation
    segments = []
    segments.append({
        "mode": mode,
        "carrier": flight_option.get("airline"),
        "flight_number": flight_option.get("flight_number"),
        "from_place": flight_option.get("departure_airport"),
        "to_place": flight_option.get("arrival_airport"),
        "departure_time": flight_option.get("departure_time"),
        "arrival_time": flight_option.get("arrival_time"),
        "duration": flight_option.get("duration_formatted"),
        "cost_inr": int(flight_option.get("price") or 0),
        "details": f"{flight_option.get('stops_summary')} via {flight_option.get('airline')}",
        "url": flight_option.get("booking_url", ""),
        "source": "google_flights_api",
    })
    return segments


def plan_round_trip(
    source: str,
    destination: str,
    start_date: date,
    trip_duration_days: int = 5,
    search_window_days: int = 3,
    passengers: int = 1,
    travel_class: str = "Economy",
    budget: float | None = None,
    max_stops: int | None = None,
) -> dict[str, Any]:
    """Execute a round-trip search across a flexible window of departure dates.

    Produces the standardized Itinerary Team payload containing:
    - journey: { onward, return, total_inr, recommendation, alternatives }
    - date_options: list of all candidate departure date packages in the window.
    """
    # 1. Resolve cities to airport codes
    try:
        dep_iata, source_city, dep_airport_name = resolve_city_to_airport(source)
        arr_iata, dest_city, arr_airport_name = resolve_city_to_airport(destination)
    except ValueError as ve:
        return {
            "status": "error",
            "message": str(ve),
        }

    # Bound search window safely between 1 and 7 days to preserve API rate limits
    window = max(1, min(int(search_window_days), 7))
    duration = max(1, int(trip_duration_days))

    date_options = []

    for offset in range(window):
        curr_dep_date = start_date + timedelta(days=offset)
        curr_ret_date = calculate_return_date(curr_dep_date, duration)

        # 2. Fetch Outbound Flights (Source ➔ Destination)
        out_best, out_other, out_url, _, out_err = _fetch_options(
            dep_iata, arr_iata, curr_dep_date, travel_class, passengers
        )

        # 3. Fetch Return Flights (Destination ➔ Source)
        ret_best, ret_other, ret_url, _, ret_err = _fetch_options(
            arr_iata, dep_iata, curr_ret_date, travel_class, passengers
        )

        if not out_best and not out_other:
            continue
        if not ret_best and not ret_other:
            continue

        # Evaluate and pick best/cheapest outbound and return
        out_target = out_best if out_best else out_other
        ret_target = ret_best if ret_best else ret_other

        # Rank outbound
        out_eval = evaluate_flights(out_target, budget=None, arrival_deadline=None, max_stops=max_stops)
        top_outbound = out_eval["recommendations"]["cheapest"] or out_target[0]
        top_outbound["booking_url"] = out_url

        # Rank return
        ret_eval = evaluate_flights(ret_target, budget=None, arrival_deadline=None, max_stops=max_stops)
        top_return = ret_eval["recommendations"]["cheapest"] or ret_target[0]
        top_return["booking_url"] = ret_url

        out_price = float(top_outbound.get("price") or 0)
        ret_price = float(top_return.get("price") or 0)
        total_round_trip = out_price + ret_price

        # Check budget
        meets_budget = True
        budget_reason = None
        if budget is not None and budget > 0:
            if total_round_trip > budget:
                meets_budget = False
                budget_reason = f"Exceeds budget by ₹{int(total_round_trip - budget):,}"

        # Standardized onward and return structures
        onward_payload = {
            "route": f"{source_city} ({dep_iata}) ➔ {dest_city} ({arr_iata})",
            "date": curr_dep_date.isoformat(),
            "departure_date_formatted": curr_dep_date.strftime("%a, %b %d, %Y"),
            "total_inr": int(out_price),
            "booking_url": out_url,
            "flight_summary": f"{top_outbound.get('airline')} ({top_outbound.get('flight_number')}) · {top_outbound.get('departure_time')} ➔ {top_outbound.get('arrival_time')} ({top_outbound.get('duration_formatted')})",
            "segments": _build_segments(top_outbound),
        }

        return_payload = {
            "route": f"{dest_city} ({arr_iata}) ➔ {source_city} ({dep_iata})",
            "date": curr_ret_date.isoformat(),
            "return_date_formatted": curr_ret_date.strftime("%a, %b %d, %Y"),
            "total_inr": int(ret_price),
            "booking_url": ret_url,
            "flight_summary": f"{top_return.get('airline')} ({top_return.get('flight_number')}) · {top_return.get('departure_time')} ➔ {top_return.get('arrival_time')} ({top_return.get('duration_formatted')})",
            "segments": _build_segments(top_return),
        }

        date_options.append({
            "option_id": offset + 1,
            "departure_date": curr_dep_date.isoformat(),
            "return_date": curr_ret_date.isoformat(),
            "departure_date_formatted": curr_dep_date.strftime("%a, %b %d"),
            "return_date_formatted": curr_ret_date.strftime("%a, %b %d"),
            "trip_duration_days": duration,
            "total_round_trip_inr": int(total_round_trip),
            "total_round_trip_formatted": f"₹{int(total_round_trip):,}",
            "meets_budget": meets_budget,
            "budget_reason": budget_reason,
            "onward": onward_payload,
            "return": return_payload,
        })

    if not date_options:
        return {
            "status": "no_results",
            "message": f"No flights found between {source_city} and {dest_city} for the selected window.",
        }

    # Sort options by cheapest total price
    date_options.sort(key=lambda x: x["total_round_trip_inr"])
    best_option = date_options[0]

    # Generate Itinerary Team recommendation text
    recommendation_text = (
        f"**Recommended Journey:** {best_option['onward']['flight_summary']} (Onward) and "
        f"{best_option['return']['flight_summary']} (Return) for **{best_option['total_round_trip_formatted']}** total.\n\n"
        f"**Why this combination?** Lowest combined round-trip fare found across the {window}-day search window. "
        f"Departs on {best_option['departure_date_formatted']} and returns on {best_option['return_date_formatted']} ({duration}-day stay)."
    )

    # Build primary Journey object (matching Itinerary Team reference format)
    primary_journey = {
        "onward": best_option["onward"],
        "return": best_option["return"],
        "total_inr": best_option["total_round_trip_inr"],
        "trip_duration_days": duration,
        "recommendation": recommendation_text,
        "alternatives": [
            {
                "mode": "flight_different_dates",
                "dates": f"{opt['departure_date_formatted']} ➔ {opt['return_date_formatted']}",
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
            "resolved_source": f"{source_city} ({dep_iata})",
            "destination_location": destination,
            "resolved_destination": f"{dest_city} ({arr_iata})",
            "start_date": start_date.isoformat(),
            "trip_duration_days": duration,
            "search_window_days": window,
            "passengers": passengers,
            "travel_class": travel_class,
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
