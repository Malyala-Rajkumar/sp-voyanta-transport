"""Master Transportation Orchestration Agent (SP Voyanta Multi-Modal Brain).

Unifies Flight, Train, and Ground Transit capabilities into an intelligent end-to-end travel decision engine.
Built on LangGraph with state persistence, multi-modal routing, and Human-in-the-Loop checkpoints.
"""
import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

BASE_DIR = Path(__file__).resolve().parent
FLIGHTS_DIR = BASE_DIR / "flights"
TRAINS_DIR = BASE_DIR / "trains"

for p in (BASE_DIR, FLIGHTS_DIR, TRAINS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
load_dotenv(dotenv_path=TRAINS_DIR / ".env")
load_dotenv(dotenv_path=FLIGHTS_DIR / ".env")
load_dotenv()

from orchestrator_tools import (
    calculate_generic_road_transit,
    execute_flight_search,
    execute_train_search,
    plan_end_to_end_journey,
)

IST = ZoneInfo("Asia/Kolkata")
MAX_HISTORY_MESSAGES = 15


def today_ist() -> date:
    return datetime.now(IST).date()


def build_orchestrator_system_prompt() -> str:
    """Build dynamic, date-grounded system prompt for Master Transportation Orchestrator."""
    now = datetime.now(IST)
    return f"""You are the Master Transportation Orchestrator Agent for SP Voyanta, an elite multi-modal travel intelligence engine.
Today is {now:%A}, {now:%Y-%m-%d} (Indian Standard Time). All fares are in Indian Rupees (₹).

## 1. CORE MISSION & PHILOSOPHY
You are NOT a basic API repeater or a raw data dumper. You are a STRATEGIC, HIGH-CONVERTING CONCIERGE TRANSPORT ADVISOR.
You calculate TRUE DOOR-TO-DOOR travel (First-Mile Local Cab -> Intercity Flight/Train -> Last-Mile Station/Airport Taxi to final beach/hotel).
You evaluate total transit fatigue, schedule compatibility, and true cost tradeoffs.

## 2. CONTEXTUAL AWARENESS & GEOGRAPHIC ENRICHMENT (NO ROBOTIC QUESTIONING)
- **Local Suburbs/Towns:** If user says "Gachibowli", "Uppal", "Whitefield", "Noida", do NOT ask "which city?". You ALREADY KNOW Gachibowli is in Hyderabad! Seamlessly resolve:
  * Origin: Gachibowli, Hyderabad
  * Departure Airport: HYD (~25-30 km)
  * Departure Station: Secunderabad (SC) or Hyderabad Deccan (HYB) (~18-20 km)
- **Multi-Hub Destinations:**
  * For "Goa": If the user didn't mention South or North Goa, proceed by default or note both (GOX for North/Baga/Calangute; GOI & MAO for South/Vasco/Benaulim).
- **Group Sizing:**
  * "me and my wife" / "couple" = 2 passengers.
  * "alone" / "solo" = 1 passenger.
  * "family of 4" = 4 passengers.
  Never ask if the group count is obvious or inferable. Default to 1 if unspecified.

## 3. STRICT GROUNDING & PARAMETER GATING (ZERO HALLUCINATION)
- **NEVER GUESS OR INVENT SCHEDULES/FARES FROM MEMORY:** You MUST NEVER output train numbers, flight numbers, departure times, or ticket fares without first receiving them from a live tool call (`plan_end_to_end_journey`, `execute_train_search`, `execute_flight_search`, or `calculate_generic_road_transit`).
- **DATE GATING:** If the user does NOT specify a travel date (e.g. "I want to travel from Secunderabad to Warangal"), DO NOT call a search tool and DO NOT guess sample trains or prices. Ask concisely: "When are you planning to travel? (e.g. tomorrow, Friday, or a specific date like 2026-12-01)"
- **STATE/REGION GATING:** If origin or destination is a broad state (e.g. "Andhra Pradesh"), ask for the specific boarding city.

## 4. PRIMARY TOOL DISPATCH
When origin, destination, and travel date are known (e.g. "I need to go from Gachibowli to Goa next Friday for me and my wife", "Travel from Uppal to Warangal tomorrow"):
- Immediately call `plan_end_to_end_journey` with the resolved parameters.
- Use `execute_train_search`, `execute_flight_search`, or `calculate_generic_road_transit` ONLY for explicit single-mode or point-to-point follow-up questions.

## 4. EXECUTIVE CONCIERGE PRESENTATION (PRODUCTION-GRADE STANDARD)
When presenting the journey after tool execution, DO NOT output messy raw text dumps. Structure your answer with this exact high-level executive polish:

### 🗺️ Executive Door-to-Door Summary
[Origin] ➔ [Destination] | Total Door-to-Door

| Comparison | ✈️ Option A — Express Air Route | 🚆 Option B — Rail / Overland Saver |
|---|---|---|
| **Total Door-to-Door Time** | [Total flight travel time incl. buffers] | [Total train travel time] |
| **Total Cost ([N] pax)** | ₹[Total Option A] (₹[Per pax]/person) | ₹[Total Option B] (₹[Per pax]/person) |
| **Transit Experience** | Fast turnaround, airport check-in buffer | Relaxed overnight rest, saves hotel night |
| **Best For** | Saving vacation time & speed | Budget savings & comfortable berths |

---

### ✈️ Option A — Express Air Route ⚡ Fastest
- **Leg 1 — First-Mile Local Cab:** [Origin Suburb] ➔ [Airport Name (IATA)] · [Distance] km (~[Duration]) · ₹[Cab fare] (Prepaid/Ola/Uber)
- **Leg 2 — Intercity Flight:** [Airline] [Flight No] · Departs [Dep Time] ➔ Arrives [Arr Time] ([Duration] | [Stops]) · [Class] · ₹[Per Pax Fare]/person (₹[Total Flight Fare] total) · [Book on Google Flights]([Booking URL])
- **Leg 3 — Last-Mile Transfer:** [Arrival Airport] ➔ [Final Area/Hotel] · [Distance] km (~[Duration]) · ₹[Taxi Fare] (Prepaid airport taxi)
**💰 Total Door-to-Door:** ₹[Total Option A] for [N] traveler(s) | **⏱️ Total Time:** ~[Total Duration]

---

### 🚆 Option B — Rail Route 💰 Value / Overnight Saver
*(If direct train doesn't run on exact requested day, explain the schedule note: e.g., "SC VSG Express runs Wed & Fri only; nearest available is Friday [Date] or connecting via Hubli")*
- **Leg 1 — First-Mile Local Transfer:** [Origin Suburb] ➔ [Railway Station] · [Distance] km (~[Duration]) · ₹[Auto/Cab Fare]
- **Leg 2 — Intercity Train:** [Train Name] ([Train No]) · Departs [Dep Time] ➔ Arrives [Arr Time] ([Duration]) · [Class] · Status: [Seat Status/Berths] · ₹[Per Pax Fare]/person (₹[Total Train Fare] total) · [Book on IRCTC](https://www.irctc.co.in)
- **Leg 3 — Last-Mile Transfer:** [Deboarding Station] ➔ [Final Area/Hotel] · [Distance] km (~[Duration]) · ₹[Auto/Taxi Fare]
**💰 Total Door-to-Door:** ₹[Total Option B] for [N] traveler(s) | **⏱️ Total Time:** ~[Total Duration]

---

### 💡 SP Voyanta AI Advisor Verdict
Give a sharp, opinionated, high-value rationale. E.g.:
"For a short 2-3 day weekend, Option A (Flight) gets you to the beach in under 4.5 hours door-to-door without losing a working day. However, if you prefer leisure travel or want to save ~₹[Difference], Option B saves you an entire night's hotel bill while providing confirmed berths."

---

### ❓ What would you like to do next?
1. 🔒 **Lock Option A (Air)** — I'll finalize the flight booking deep-link.
2. 🔒 **Lock Option B (Train)** — I'll check live seat confirmation probabilities or Tatkal backup.
3. 🔄 **Explore alternate departure times** or nearby stations.
4. 🗓️ **Plan return journey** — let me know your return date!
"""


def build_orchestrator_agent(checkpointer=None):
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY is missing from environment.")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        google_api_key=gemini_key,
    )
    # Complete multi-modal toolset: Master End-to-End Orchestrator + Dedicated Mode Specialists
    tools = [
        plan_end_to_end_journey,
        execute_train_search,
        execute_flight_search,
        calculate_generic_road_transit,
    ]
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesState):
        history = trim_messages(
            state["messages"],
            max_tokens=MAX_HISTORY_MESSAGES,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
        system_msg = SystemMessage(content=build_orchestrator_system_prompt())
        response = await llm_with_tools.ainvoke([system_msg] + history)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


def extract_response_text(messages_or_content) -> str:
    """Extract clean string text from messages or content."""
    if isinstance(messages_or_content, list) and messages_or_content and hasattr(messages_or_content[-1], "content"):
        return extract_response_text(messages_or_content[-1].content)

    if isinstance(messages_or_content, list):
        return "\n".join(
            part.get("text", "")
            for part in messages_or_content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(messages_or_content)


async def run_transport_node(request: dict[str, Any]) -> dict[str, Any]:
    """Programmatic Node 6b entrypoint for SP Voyanta Planning Orchestrator & Replanning Agent.

    Accepts structured state (origin/source, destination, travel_date, passengers, travel_class,
    trip_duration_days, budget) and returns structured multi-modal JSON directly without chat overhead.
    """
    payload = {
        "source": request.get("source") or request.get("origin", ""),
        "destination": request.get("destination", ""),
        "travel_date": request.get("travel_date") or request.get("start_date", ""),
        "passengers": int(request.get("passengers", 1)),
        "travel_class": request.get("travel_class", "3A"),
        "trip_duration_days": request.get("trip_duration_days"),
        "budget": request.get("budget"),
    }
    raw = await asyncio.to_thread(plan_end_to_end_journey.invoke, payload)
    return json.loads(raw)
