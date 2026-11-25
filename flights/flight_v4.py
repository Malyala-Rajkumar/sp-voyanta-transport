"""SP Voyanta Flight Agent (v4): High-fidelity flight search with Google Flights UI structure."""
import asyncio
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import serpapi
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from common.custom_exceptions import CustomException
from common.logger import get_logger

# Load .env from script dir as well as root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

logger = get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# Tunables
SHORT_LAYOVER_MIN = 60
LONG_LAYOVER_MIN = 240
CACHE_TTL_SECONDS = 600
SERPAPI_TIMEOUT = 25
SERPAPI_RETRIES = 3
MAX_HISTORY_MESSAGES = 30

TravelClass = Literal["Economy", "Premium Economy", "Business", "First"]
CLASS_CODES = {"Economy": 1, "Premium Economy": 2, "Business": 3, "First": 4}


def today_ist() -> date:
    return datetime.now(IST).date()


# ============================================================================
# 1. TOOL INPUT VALIDATION
# ============================================================================

class FlightSearchInput(BaseModel):
    departure_iata: str = Field(description="3-letter departure IATA airport code (e.g. JAI, HYD, DEL, BOM)")
    arrival_iata: str = Field(description="3-letter destination IATA airport code (e.g. COK, GOI, GOX, BLR)")
    travel_date: date = Field(description="Departure date in YYYY-MM-DD format")
    budget: float | None = Field(default=None, description="Maximum ticket budget in INR")
    arrival_deadline: datetime | None = Field(default=None, description="Latest acceptable arrival, e.g. 2026-11-20T18:00")
    max_stops: int | None = Field(default=None, ge=0, description="Maximum permitted stops (0 = non-stop only)")
    passengers: int = Field(default=1, ge=1, le=9, description="Number of adult passengers (1 to 9)")
    travel_class: TravelClass = Field(default="Economy", description="Cabin class")

    @field_validator("departure_iata", "arrival_iata")
    @classmethod
    def validate_iata(cls, value: str) -> str:
        clean = value.strip().upper()
        if len(clean) != 3 or not clean.isalpha():
            raise ValueError(f"'{value}' is invalid. A valid 3-letter IATA code is required (e.g. HYD, DEL).")
        return clean

    @field_validator("budget")
    @classmethod
    def validate_budget(cls, value):
        if value is not None and value <= 0:
            raise ValueError("Budget must be greater than zero.")
        return value

    @model_validator(mode="after")
    def validate_route_and_date(self):
        if self.departure_iata == self.arrival_iata:
            raise ValueError("Origin and destination airports cannot be identical.")
        if self.travel_date < today_ist():
            raise ValueError("Travel date cannot be in the past.")
        return self


# ============================================================================
# 2. PARSING, FORMATTING & EVALUATION HELPERS
# ============================================================================

def parse_dt(value):
    """Parse SerpApi datetime strings into naive datetime (local airport time)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def fmt_time_12h(dt_or_str) -> str:
    """Format datetime into clean 12-hour AM/PM format (e.g. 12:45 PM)."""
    if not dt_or_str:
        return ""
    dt = dt_or_str if isinstance(dt_or_str, datetime) else parse_dt(dt_or_str)
    if not dt:
        return str(dt_or_str)
    return dt.strftime("%I:%M %p").lstrip("0")


def fmt_minutes(mins) -> str:
    if mins is None:
        return "N/A"
    return f"{mins // 60}h {mins % 60}m"


def extract_flight(flight):
    """Normalize one SerpApi itinerary into a clean dict with human-readable fields."""
    legs = flight.get("flights") or []
    if not legs:
        return None

    first, last = legs[0], legs[-1]
    dep_airport = first.get("departure_airport") or {}
    arr_airport = last.get("arrival_airport") or {}

    dep_dt = parse_dt(dep_airport.get("time"))
    arr_dt = parse_dt(arr_airport.get("time"))
    arrives_next_day = bool(dep_dt and arr_dt and arr_dt.date() > dep_dt.date())

    dep_formatted = fmt_time_12h(dep_dt)
    arr_formatted = fmt_time_12h(arr_dt) + (" (+1 day)" if arrives_next_day else "")

    layovers = []
    layover_summaries = []
    for cur, nxt in zip(legs, legs[1:]):
        arr = parse_dt((cur.get("arrival_airport") or {}).get("time"))
        dep = parse_dt((nxt.get("departure_airport") or {}).get("time"))
        airport_name = (cur.get("arrival_airport") or {}).get("name", "Unknown")
        airport_id = (cur.get("arrival_airport") or {}).get("id", "")
        if arr and dep:
            mins = int((dep - arr).total_seconds() / 60)
            flags = []
            if mins < SHORT_LAYOVER_MIN:
                flags.append("short_layover")
            if mins > LONG_LAYOVER_MIN:
                flags.append("long_layover")
            layovers.append({
                "transit_airport": airport_name,
                "transit_id": airport_id,
                "duration_minutes": mins,
                "duration_formatted": fmt_minutes(mins),
                "flags": flags,
            })
            layover_summaries.append(f"{fmt_minutes(mins)} in {airport_name} ({airport_id})")

    try:
        price = float(flight["price"]) if flight.get("price") is not None else None
    except (ValueError, TypeError):
        price = None

    duration = flight.get("total_duration")
    airlines = list(dict.fromkeys(l.get("airline") for l in legs if l.get("airline")))
    flight_numbers = [l.get("flight_number") for l in legs if l.get("flight_number")]

    stops_count = len(legs) - 1
    stops_text = "Non-stop" if stops_count == 0 else f"{stops_count} stop ({', '.join(layover_summaries)})"

    return {
        "airlines": airlines,
        "airline": " + ".join(airlines) if airlines else "Unknown Airline",
        "flight_numbers": flight_numbers,
        "flight_number": ", ".join(flight_numbers) if flight_numbers else "",
        "departure_airport": dep_airport.get("id"),
        "departure_time_raw": dep_airport.get("time"),
        "departure_time": dep_formatted,
        "arrival_airport": arr_airport.get("id"),
        "arrival_time_raw": arr_airport.get("time"),
        "arrival_time": arr_formatted,
        "arrives_next_day": arrives_next_day,
        "duration_minutes": duration,
        "duration_formatted": fmt_minutes(duration),
        "price": price,
        "price_formatted": f"₹{int(price):,}" if price is not None else "Unavailable",
        "stops": stops_count,
        "stops_summary": stops_text,
        "layovers": layovers,
        "has_layover_warning": any(l["flags"] for l in layovers),
        "legs": legs,
    }


def _norm(value, lo, hi):
    return 0.0 if hi <= lo else (value - lo) / (hi - lo)


def rank_flights(flights):
    """Find cheapest, fastest, and weighted best-overall flight."""
    valid = [f for f in flights if f.get("price") is not None and f.get("duration_minutes") is not None]
    if not valid:
        return {"cheapest": None, "fastest": None, "best_overall": None}

    prices = [f["price"] for f in valid]
    durations = [f["duration_minutes"] for f in valid]
    p_lo, p_hi, d_lo, d_hi = min(prices), max(prices), min(durations), max(durations)

    def score(f):
        s = 0.50 * _norm(f["price"], p_lo, p_hi) + 0.35 * _norm(f["duration_minutes"], d_lo, d_hi)
        s += 0.10 * f["stops"]
        if f["has_layover_warning"]:
            s += 0.15
        return s

    cheapest_pick = min(valid, key=lambda f: (f["price"], f["duration_minutes"]))
    fastest_pick = min(valid, key=lambda f: (f["duration_minutes"], f["stops"], f["price"]))
    best_pick = min(valid, key=lambda f: (score(f), f["price"]))

    return {
        "cheapest": cheapest_pick,
        "fastest": fastest_pick,
        "best_overall": best_pick,
    }


def evaluate_flights(flights, budget=None, arrival_deadline=None, max_stops=None, absolute_cheapest_price=None):
    """Filter flights into matching vs alternatives based on user constraints."""
    matching, alternatives = [], []
    deadline = arrival_deadline.replace(tzinfo=None) if arrival_deadline else None

    for f in flights:
        reasons = []
        price, stops = f.get("price"), f.get("stops")
        arrival = parse_dt(f.get("arrival_time_raw"))

        if budget is not None and (price is None or price > budget):
            reasons.append("Exceeds budget")
        if deadline is not None and (arrival is None or arrival > deadline):
            reasons.append("Arrives after deadline")
        if max_stops is not None and (stops is None or stops > max_stops):
            reasons.append(f"Exceeds max stops ({stops} > {max_stops})")

        if reasons:
            alternatives.append({**f, "unmet_requirements": reasons})
        else:
            matching.append(f)

    sort_key = lambda f: (f.get("price") is None, f.get("price") or float("inf"))
    matching.sort(key=sort_key)
    alternatives.sort(key=sort_key)

    target = matching if matching else alternatives
    recs = rank_flights(target)

    # Determine what price qualifies for the "Cheapest" badge
    qualifying_cheap_price = absolute_cheapest_price if absolute_cheapest_price is not None else (recs["cheapest"]["price"] if recs["cheapest"] else None)

    # Attach badge labels to flights
    for f in matching:
        badges = []
        if recs["best_overall"] and f["flight_number"] == recs["best_overall"]["flight_number"] and f["departure_time"] == recs["best_overall"]["departure_time"]:
            badges.append("⭐ Best Overall")
        # Only assign Cheapest badge if this flight matches the absolute lowest fare
        if qualifying_cheap_price is not None and f.get("price") == qualifying_cheap_price:
            badges.append("🏷️ Cheapest")
        if recs["fastest"] and f["flight_number"] == recs["fastest"]["flight_number"] and f["departure_time"] == recs["fastest"]["departure_time"]:
            badges.append("⚡ Fastest")
        f["badge"] = " · ".join(badges) if badges else None

    return {
        "matching_flights": matching,
        "alternative_flights": alternatives,
        "recommendations": recs,
        "recommendations_source": "matching" if matching else "alternatives",
    }


# ============================================================================
# 3. SERPAPI ACCESS WITH POINT-OF-SALE FIX ("gl": "in")
# ============================================================================

_CACHE: dict[tuple, tuple[float, dict]] = {}


def _cache_get(key):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]
    _CACHE.pop(key, None)
    return None


def _cache_set(key, value):
    if len(_CACHE) > 500:
        _CACHE.clear()
    _CACHE[key] = (time.time(), value)


def _serpapi_search(params: dict) -> dict:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise CustomException("SERPAPI_API_KEY is missing from environment.")

    client = serpapi.Client(api_key=api_key, timeout=SERPAPI_TIMEOUT)
    last_err = None
    for attempt in range(SERPAPI_RETRIES):
        try:
            results = client.search(params)
            return results.as_dict() if hasattr(results, "as_dict") else dict(results)
        except Exception as e:
            last_err = e
            logger.warning("SerpApi attempt %d/%d failed: %s", attempt + 1, SERPAPI_RETRIES, e)
            time.sleep(2 ** attempt)
    raise last_err


def _fetch_options(dep, arr, travel_date, travel_class, passengers):
    """Fetch flights with gl='in' to guarantee domestic Indian promotional fares."""
    key = (dep, arr, travel_date.isoformat(), travel_class, passengers)
    cached = _cache_get(key)
    if cached:
        return cached["best_options"], cached["other_options"], cached["booking_url"], cached["price_insights"], None

    # CRITICAL: gl="in" ensures Google Flights queries the Indian point-of-sale inventory
    data = _serpapi_search({
        "engine": "google_flights",
        "departure_id": dep,
        "arrival_id": arr,
        "outbound_date": travel_date.isoformat(),
        "type": "2",  # one-way
        "currency": "INR",
        "hl": "en",
        "gl": "in",   # <--- India point-of-sale lock
        "adults": passengers,
        "travel_class": CLASS_CODES[travel_class],
    })

    if data.get("error"):
        msg = str(data["error"])
        if "hasn't returned any results" in msg.lower():
            return [], [], "", None, None
        return [], [], "", None, msg

    def extract_group(raw_list):
        options, seen = [], set()
        for f in raw_list:
            ext = extract_flight(f)
            if not ext:
                continue
            sig = (tuple(ext["flight_numbers"]), ext["departure_time_raw"])
            if sig not in seen:
                seen.add(sig)
                options.append(ext)
        return options

    best_options = extract_group(data.get("best_flights") or [])
    other_options = extract_group(data.get("other_flights") or [])
    booking_url = (data.get("search_metadata") or {}).get("google_flights_url", "")
    price_insights = data.get("price_insights") or {}

    _cache_set(key, {
        "best_options": best_options,
        "other_options": other_options,
        "booking_url": booking_url,
        "price_insights": price_insights,
    })

    return best_options, other_options, booking_url, price_insights, None


# ============================================================================
# 4. TOOLS (One-Way & Round-Trip Planning)
# ============================================================================

class RoundTripPlanInput(BaseModel):
    source_city: str = Field(description="Origin city or airport name (e.g. Hyderabad, Jaipur, Delhi, Mumbai)")
    destination_city: str = Field(description="Destination city or airport name (e.g. Goa, Kochi, Bengaluru)")
    start_date: date = Field(description="Earliest departure date in YYYY-MM-DD format")
    trip_duration_days: int = Field(default=5, ge=1, le=60, description="Duration of stay in days at destination")
    search_window_days: int = Field(default=3, ge=1, le=7, description="Number of departure dates to check in window (1 to 7)")
    passengers: int = Field(default=1, ge=1, le=9, description="Number of adult passengers")
    travel_class: TravelClass = Field(default="Economy", description="Cabin class")
    budget: float | None = Field(default=None, description="Max total round-trip budget in INR")
    max_stops: int | None = Field(default=None, ge=0, description="Max permitted stops per leg")


@tool(args_schema=RoundTripPlanInput)
def plan_round_trip_flights(
    source_city: str,
    destination_city: str,
    start_date: date,
    trip_duration_days: int = 5,
    search_window_days: int = 3,
    passengers: int = 1,
    travel_class: str = "Economy",
    budget: float | None = None,
    max_stops: int | None = None,
) -> str:
    """Plan complete round-trip flight itineraries across a flexible departure window, automatically calculating return dates and pairing onward + return flights."""
    from round_trip_engine import plan_round_trip as execute_round_trip
    logger.info("Round-trip plan: %s -> %s | Start: %s | Duration: %d days | Window: %d days",
                source_city, destination_city, start_date, trip_duration_days, search_window_days)
    try:
        result = execute_round_trip(
            source=source_city,
            destination=destination_city,
            start_date=start_date,
            trip_duration_days=trip_duration_days,
            search_window_days=search_window_days,
            passengers=passengers,
            travel_class=travel_class,
            budget=budget,
            max_stops=max_stops,
        )
        onward_url = result.get("journey", {}).get("onward", {}).get("booking_url", "")
        result["booking_url"] = onward_url
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        err = CustomException(f"Round-trip planning failed for {source_city} -> {destination_city}", e)
        logger.exception(str(err))
        return json.dumps({"status": "error", "message": str(err)})


@tool(args_schema=FlightSearchInput)
def search_flights(
    departure_iata: str,
    arrival_iata: str,
    travel_date: date,
    budget: float | None = None,
    arrival_deadline: datetime | None = None,
    max_stops: int | None = None,
    passengers: int = 1,
    travel_class: str = "Economy",
) -> str:
    """Search live one-way flights (Google Flights via SerpApi) with Google-structured ranking."""
    logger.info("Flight search: %s -> %s | %s | pax=%s | %s",
                departure_iata, arrival_iata, travel_date, passengers, travel_class)
    try:
        best_options, other_options, booking_url, price_insights, error = _fetch_options(
            departure_iata, arrival_iata, travel_date, travel_class, passengers
        )
        if error:
            return json.dumps({"status": "error", "message": error})
        if not best_options and not other_options:
            return json.dumps({
                "status": "no_results",
                "route": f"{departure_iata} to {arrival_iata}",
                "departure_date": travel_date.isoformat(),
            })

        # Determine absolute lowest fare across ALL returned options (both best and other)
        all_options = best_options + other_options
        all_prices = [f["price"] for f in all_options if f.get("price") is not None]
        absolute_cheapest_price = min(all_prices) if all_prices else None

        # Evaluate flights against constraints with true absolute lowest fare
        best_evaluation = evaluate_flights(best_options, budget, arrival_deadline, max_stops, absolute_cheapest_price)
        other_evaluation = evaluate_flights(other_options, budget, arrival_deadline, max_stops, absolute_cheapest_price)

        # Lowest price insight across all options
        lowest_fare = None
        if absolute_cheapest_price is not None:
            lowest_fare = f"₹{int(absolute_cheapest_price):,}"
        elif price_insights and "lowest_price" in price_insights:
            lowest_fare = f"₹{int(price_insights['lowest_price']):,}"

        return json.dumps({
            "status": "success",
            "booking_url": booking_url,
            "route": f"{departure_iata} to {arrival_iata}",
            "departure_date": travel_date.strftime("%A, %b %d, %Y"),
            "passengers": passengers,
            "travel_class": travel_class,
            "lowest_fare_summary": lowest_fare,
            "recommendations": best_evaluation["recommendations"],
            "top_flights": best_evaluation["matching_flights"][:4],
            "other_options": other_evaluation["matching_flights"][:3],
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        err = CustomException(f"Flight search failed for {departure_iata} -> {arrival_iata}", e)
        logger.exception(str(err))
        return json.dumps({"status": "error", "message": "Flight search is temporarily unavailable."})


# ============================================================================
# 5. HIGH-FIDELITY STRUCTURED SYSTEM PROMPT
# ============================================================================

def build_system_prompt() -> str:
    now = datetime.now(IST)
    return f"""You are the Lead Flight Specialist in SP Voyanta, a multi-agent trip planner.
Today is {now:%A}, {now:%Y-%m-%d} (IST). All fares are in Indian Rupees (₹).

## 1. SCOPE & TOOL SELECTION
- Use `search_flights`: for one-way journeys, relocations, or when the user explicitly says they want to "stay there" indefinitely.
- Use `plan_round_trip_flights`: when the user asks for a vacation, holiday, round-trip, or flexible trip with a stay duration and search window (e.g., 3-5 days stay).

## 2. STRICT PARAMETER GATING (MANDATORY CHECKS BEFORE TOOL CALL)
You MUST NOT call any tool until you have confirmed: (1) Origin, (2) Destination, and (3) Exact Departure Date.
If ANY parameter is missing or ambiguous, ask a polite clarification question and DO NOT call any tool.

CRITICAL GATING & TRIP LOGIC RULES:
1. NO DATE ➔ ASK DATE:
   - If the user did not specify an exact date (e.g. 2026-11-26) or a relative date (e.g. 'tomorrow', 'next Friday'):
     NEVER guess, assume, or default to tomorrow or today!
     Ask: "What date are you planning to travel?"
   - Relative dates must be calculated relative to today ({now:%Y-%m-%d}).

2. STATE OR REGION INPUT ➔ ALWAYS ASK CITY FIRST:
   - A state or broad region is NOT an airport (e.g. "Andhra Pradesh", "Rajasthan", "Kerala", "Tamil Nadu", "Maharashtra").
   - You MUST ALWAYS ask the user to clarify which specific city or airport they wish to depart from or arrive at!
   - DO NOT automatically search multiple airports across a state unless the user EXPLICITLY asks to compare multiple cities (e.g. "Search both Goa airports" or "Compare flights from Visakhapatnam and Vijayawada").
   - If the user says only "Andhra Pradesh to Goa", DO NOT call any tool. Ask: "Which city in Andhra Pradesh would you like to depart from? (e.g. Visakhapatnam VTZ, Vijayawada VGA, Tirupati TIR)".

3. GROUP OR FAMILY MENTION ➔ ASK PASSENGER COUNT:
   - If the user mentions "family", "friends", "colleagues", "group", "we", or "with my family":
     NEVER default to 1 passenger! Defaulting to 1 passenger for a family/friends trip is completely wrong.
     Ask: "How many passengers (adults and children) will be traveling in your group?"

4. ROUND-TRIP DURATION / STAYING THERE:
   - If the user asks for a vacation or trip (e.g. "Plan a trip to Goa") but does NOT specify how many days:
     Ask: "How many days are you planning to stay? (Most travelers prefer a 3-day weekend or 5-day vacation)."
   - If the user says they want to "stay there" / "relocate" / "one-way":
     Search one-way flights using `search_flights`, and note that return flights can be planned anytime.
   - If the user specifies duration (e.g., "5-day trip"):
     Use `plan_round_trip_flights` with `trip_duration_days=5`. It will automatically calculate the return date and pair outbound + return flights.

## 3. AIRPORT CODES
- Map cities to commercial IATA codes:
  - Major Indian Hubs: Jaipur JAI, Kochi COK, Hyderabad HYD, Mumbai BOM, Delhi DEL, Bangalore BLR, Chennai MAA, Kolkata CCU, Pune PNQ.
  - Andhra Pradesh: Visakhapatnam VTZ, Vijayawada VGA, Tirupati TIR, Rajahmundry RJA.
  - Goa: Distinguish Dabolim (GOI - South Goa) vs Manohar Mopa (GOX - North Goa).
  - Non-airport cities (e.g. Munnar): Inform the user and suggest nearest hubs (COK or IXM).

## 4. PRESENTATION & STRUCTURE (CRITICAL - REPLICATE GOOGLE FLIGHTS UI)
When displaying flight results, format the response with clear visual hierarchy:

### ✈️ Flight Search: [Origin] ➔ [Destination]
📅 **Date:** [Departure Date] | 👤 **[Passengers] Passenger(s) ([Class])**
💡 **Price Summary:** Lowest fare starts at **[lowest_fare_summary]**

---

#### 🌟 Top Flights (Ranked by Price & Convenience)
Present each flight in `top_flights` as a clean structured card:
1. **[Airline] · [Flight Numbers]** [badge if present, e.g. ⭐ Best Overall / 🏷️ Cheapest / ⚡ Fastest]
   - 🕒 **[Departure Time] ➔ [Arrival Time]** ([Duration])
   - 🛑 [Stops Summary, e.g. '1 stop (1h 30m in Delhi DEL)' or 'Non-stop']
   - 💰 **[Price Formatted]**

---

#### 📋 Other Available Options
Brief bullet list of flights from `other_options`:
- **[Airline]** ([Departure Time] ➔ [Arrival Time]) · [Duration] · [Stops Summary] · **[Price Formatted]**

---

### 🔗 Booking Information
CRITICAL LINK INSTRUCTION:
- You must use the exact `booking_url` provided in the MOST RECENT tool call.
- NEVER reuse, recycle, or copy a booking URL from a previous conversation turn. If the route or direction has changed (e.g. from JAI->COK to COK->JAI), the booking URL MUST match the new direction from the latest tool payload.
Provide the link: **[View & Book on Google Flights](booking_url)**
*(Fares change dynamically; final checkout occurs on the provider site.)*

## 5. STYLE & DATA FIDELITY
- Never invent flights, prices, baggage, or refund policies. Use ONLY tool data.
- Maintain route direction fidelity: double check that your printed route (e.g. Origin ➔ Destination) matches the actual `departure_iata` and `arrival_iata` returned by the tool.
- Never output an unreadable raw markdown table; use the structured card layout above.
"""


# ============================================================================
# 6. LLM + LANGGRAPH WORKFLOW
# ============================================================================

def build_agent(checkpointer=None):
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise CustomException("GEMINI_API_KEY / GOOGLE_API_KEY is missing from environment.")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        google_api_key=gemini_key,
    )
    llm_with_tools = llm.bind_tools([search_flights, plan_round_trip_flights])

    async def agent_node(state: MessagesState):
        try:
            history = trim_messages(
                state["messages"],
                max_tokens=MAX_HISTORY_MESSAGES,
                token_counter=len,
                strategy="last",
                start_on="human",
                include_system=False,
                allow_partial=False,
            )
            response = await llm_with_tools.ainvoke([SystemMessage(content=build_system_prompt())] + history)
            return {"messages": [response]}
        except Exception as e:
            err = CustomException("LLM processing failed in agent_node", e)
            logger.exception(str(err))
            raise err

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode([search_flights, plan_round_trip_flights], handle_tool_errors=True))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=checkpointer)


def extract_response_text(messages_or_content):
    """Extract clean string text from messages or content, ensuring route-specific link fidelity."""
    import re
    if isinstance(messages_or_content, list) and messages_or_content and hasattr(messages_or_content[-1], "content"):
        final_msg = messages_or_content[-1]
        raw_text = final_msg.content
        text = extract_response_text(raw_text)

        # Collect tool responses from this turn to match URLs by route
        route_url_map = {}
        single_latest_url = None
        for msg in reversed(messages_or_content):
            if getattr(msg, "type", None) == "human" or msg.__class__.__name__ == "HumanMessage":
                break
            if getattr(msg, "type", None) == "tool" or msg.__class__.__name__ == "ToolMessage":
                try:
                    payload = json.loads(msg.content)
                    if isinstance(payload, dict):
                        b_url = payload.get("booking_url")
                        route_str = payload.get("route", "")
                        if b_url:
                            if not single_latest_url:
                                single_latest_url = b_url
                            if route_str:
                                # e.g. "HYD to GOX" or "VTZ to GOI"
                                parts = route_str.upper().split(" TO ")
                                if len(parts) == 2:
                                    orig, dest = parts[0].strip(), parts[1].strip()
                                    route_url_map[f"{orig}_{dest}"] = b_url
                                    route_url_map[orig] = b_url
                                    route_url_map[dest] = b_url
                except Exception:
                    pass

        # If multiple routes were searched (e.g. GOI and GOX, or VTZ and VGA):
        # Match each markdown link based on the specific airport mentioned in that line!
        if route_url_map and len(set(route_url_map.values())) > 1:
            lines = text.split("\n")
            new_lines = []
            for line in lines:
                if "https://www.google.com/travel/flights" in line:
                    matched_url = None
                    line_upper = line.upper()
                    # Check for specific airport code mentions in this line (e.g. GOX, GOI, VTZ, VGA)
                    for code, url in route_url_map.items():
                        if "_" not in code:  # Individual airport code (e.g. GOX, GOI)
                            if re.search(rf"\b{re.escape(code)}\b", line_upper):
                                matched_url = url
                                break
                    if matched_url:
                        line = re.sub(r'https://www\.google\.com/travel/flights[^\s\)\"\'>]+', matched_url, line)
                new_lines.append(line)
            text = "\n".join(new_lines)
        elif single_latest_url:
            text = re.sub(
                r'https://www\.google\.com/travel/flights[^\s\)\"\'>]+',
                single_latest_url,
                text
            )
        return text

    if isinstance(messages_or_content, str):
        return messages_or_content
    if isinstance(messages_or_content, list):
        return "\n".join(
            part.get("text", "")
            for part in messages_or_content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(messages_or_content)


# ============================================================================
# 7. INTERACTIVE CLI
# ============================================================================

async def cli():
    sys.stdout.reconfigure(encoding="utf-8")
    agent = build_agent(MemorySaver())
    config = {"configurable": {"thread_id": "cli-session-v4"}}

    default_prompt = "Find flights from Jaipur to Kochi on 2026-11-26 for 1 passenger in Economy"

    print("=" * 65)
    print("✈️ SP VOYANTA FLIGHT AGENT v4 (Google Flights Structured UI)")
    print("Type 'exit' to quit.")
    print("=" * 65)

    query = (await asyncio.to_thread(input, f"\nQuery (Enter for default: '{default_prompt}'):\n> ")).strip() or default_prompt

    while True:
        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=config)
        print("\nFlight Agent:\n" + "-" * 65)
        print(extract_response_text(result["messages"]))
        print("-" * 65)

        query = (await asyncio.to_thread(input, "\nFollow-up (Enter/'exit' to quit):\n> ")).strip()
        if not query or query.lower() in ("exit", "quit"):
            print("Session ended. Have a great journey!")
            break


if __name__ == "__main__":
    try:
        asyncio.run(cli())
    except Exception as e:
        err = CustomException("Flight Agent v4 execution failed", e)
        logger.exception(str(err))
        print(f"Execution Error: {err}")
