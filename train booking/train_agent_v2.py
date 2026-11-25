"""SP Voyanta Train Agent (v2): LangGraph ReAct Agent with IRCTC Live Data,

Station Cluster Entity Resolution, and Itinerary Team Compatibility.
"""
import asyncio
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from railkit_client import normalize_date_str, railkit
from train_round_trip_engine import plan_round_trip_trains as execute_round_trip_trains
from station_resolver import resolve_city_to_station
from tools.train_tools import check_availability, search_trains

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
load_dotenv()

IST = ZoneInfo("Asia/Kolkata")
MAX_HISTORY_MESSAGES = 25


def today_ist() -> date:
    return datetime.now(IST).date()


# ============================================================================
# ROUND-TRIP TOOL DEFINITION
# ============================================================================

class TrainRoundTripInput(BaseModel):
    source_city: str = Field(description="Origin city or railway station (e.g. Hyderabad, Delhi, Mumbai, Secunderabad)")
    destination_city: str = Field(description="Destination city or railway station (e.g. Goa, Vijayawada, Kochi, Varanasi)")
    start_date: str = Field(description="Earliest departure date in YYYY-MM-DD or DD-MM-YYYY format")
    trip_duration_days: int = Field(default=5, description="Number of days staying at the destination (e.g. 3, 5, 7)")
    search_window_days: int = Field(default=3, description="Days to scan from start date to find confirmed/cheapest seats (1-7)")
    passengers: int = Field(default=1, description="Number of adult passengers")
    travel_class: str = Field(default="3A", description="Travel class: 3A, 2A, 1A, SL, CC, 2S")
    budget: float | None = Field(default=None, description="Optional total round-trip budget in INR")


@tool(args_schema=TrainRoundTripInput)
def plan_round_trip_trains(
    source_city: str,
    destination_city: str,
    start_date: str,
    trip_duration_days: int = 5,
    search_window_days: int = 3,
    passengers: int = 1,
    travel_class: str = "3A",
    budget: float | None = None,
) -> str:
    """Plan a round-trip vacation or holiday by train with automatic return date calculation and window scanning.

    Automatically pairs outbound and return trains and outputs the Itinerary Team's schema.
    """
    try:
        res = execute_round_trip_trains(
            source=source_city,
            destination=destination_city,
            start_date=start_date,
            trip_duration_days=trip_duration_days,
            search_window_days=search_window_days,
            passengers=passengers,
            travel_class=travel_class,
            budget=budget,
        )
        return json.dumps(res, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Round-trip train planning failed: {e}"})


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

def build_system_prompt() -> str:
    now = datetime.now(IST)
    return f"""You are the Lead Train & Railway Specialist in SP Voyanta, a multi-modal trip planner.
Today is {now:%A}, {now:%Y-%m-%d} (IST). All fares are in Indian Rupees (₹).

## 1. SCOPE & TOOL SELECTION
- Use `search_trains`: For one-way journeys, relocations, or general timetable searches between two stations/cities.
- Use `check_availability`: When checking live seat availability (e.g. Confirm vs WL/RAC) and exact fare breakdown for a specific train number.
- Use `plan_round_trip_trains`: When the user asks for a vacation, holiday, round-trip, or flexible trip with a stay duration (e.g. 'Plan a 5-day trip to Goa by train').

## 2. STRICT PARAMETER GATING (MANDATORY CHECKS BEFORE CALLING TOOLS)
You MUST NOT call any tool until you have confirmed: (1) Origin, (2) Destination, and (3) Exact Departure Date.
If ANY parameter is missing or ambiguous, ask a polite clarification question and DO NOT call any tool.

CRITICAL GATING RULES:
1. NO DATE ➔ ALWAYS ASK DATE (NEVER DEFAULT TO TODAY OR TOMORROW):
   - If the user provides origin and destination without specifying ANY date (e.g. "I want to travel from Khammam to Mahabubabad"):
     YOU MUST NOT CALL ANY TOOL!
     Do NOT assume today, tomorrow, or any default date!
     Respond immediately:
     "I would be glad to check trains from [Origin] to [Destination] for you! What date are you planning to travel (e.g. tomorrow, next Friday, or a specific date)? Also, are you looking for a one-way journey or a round-trip trip?"
   - Relative dates ('tomorrow', 'next Friday') must be computed relative to today ({now:%Y-%m-%d}).

2. STATE OR REGION INPUT ➔ ALWAYS ASK CITY FIRST:
   - A state or broad region is NOT a station (e.g. "Andhra Pradesh", "Kerala", "Rajasthan", "Maharashtra").
   - You MUST ALWAYS ask the user to clarify which specific city or boarding station they wish to depart from or arrive at!
   - Example: If the user says "Trains from Andhra Pradesh to Delhi", DO NOT call any tool.
     Ask: "Which city in Andhra Pradesh would you like to board from? (e.g. Vijayawada BZA, Visakhapatnam VSKP, Tirupati TPTY)".

3. GROUP OR FAMILY MENTION ➔ ASK PASSENGER COUNT:
   - If the user mentions "family", "friends", "colleagues", "group", "we", or "with my family":
     NEVER default to 1 passenger! Ask: "How many passengers will be traveling in your group?"

4. ROUND-TRIP STAY DURATION:
   - If the user asks for a vacation (e.g. "Plan a trip to Goa") but does not specify days:
     Ask: "How many days are you planning to stay? (Most travelers prefer a 3-day weekend or 5-day vacation)."
   - If duration is given (e.g., "5-day trip"):
     Use `plan_round_trip_trains` with `trip_duration_days=5`.

## 3. PRESENTATION & STRUCTURE (IRCTC CLEAN CARD LAYOUT)
When displaying train search results, format with clear visual hierarchy:

### 🚆 Train Options: [Origin] ➔ [Destination]
📅 **Date:** [Journey Date] | 👤 **[Passengers] Passenger(s) ([Class])**

For each recommended train:
**[Train Number] · [Train Name]** 🏷️ [Status/Badge]
🕒 **[Departure Time] ([From Station]) ➔ [Arrival Time] ([To Station])** ([Duration])
💺 **Classes:** [Available Classes, e.g. 3A, 2A, SL] | 📅 **Runs:** [Days]
💰 **Fare & Seats:** [Live status if checked, or guidance to check availability]

🔗 **Booking Information:**
- Official tickets can be booked on [IRCTC Official Portal](https://www.irctc.co.in).
"""


# ============================================================================
# AGENT BUILDER (LANGGRAPH + MEMORY)
# ============================================================================

def build_train_agent(checkpointer=None):
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY is missing from environment.")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        google_api_key=gemini_key,
    )
    tools = [search_trains, check_availability, plan_round_trip_trains]
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
        response = await llm_with_tools.ainvoke([SystemMessage(content=build_system_prompt())] + history)
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
        final_msg = messages_or_content[-1]
        raw_text = final_msg.content
        return extract_response_text(raw_text)

    if isinstance(messages_or_content, list):
        return "\n".join(
            part.get("text", "")
            for part in messages_or_content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(messages_or_content)


# ============================================================================
# CLI TESTING INTERFACE
# ============================================================================

async def cli():
    sys.stdout.reconfigure(encoding="utf-8")
    agent = build_train_agent(MemorySaver())
    config = {"configurable": {"thread_id": "train-cli-session-v2"}}

    default_prompt = "Find trains from Hyderabad to Vijayawada on 2026-11-28 for 2 passengers"

    print("=" * 65)
    print("🚆 SP VOYANTA TRAIN AGENT v2 (LangGraph + RailKit Live)")
    print("Type 'exit' to quit.")
    print("=" * 65)

    query = (await asyncio.to_thread(input, f"\nQuery (Enter for default: '{default_prompt}'):\n> ")).strip() or default_prompt

    while True:
        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=config)
        print("\nTrain Agent:\n" + "-" * 65)
        print(extract_response_text(result["messages"]))
        print("-" * 65)

        query = (await asyncio.to_thread(input, "\nFollow-up (Enter/'exit' to quit):\n> ")).strip()
        if not query or query.lower() in ("exit", "quit"):
            print("Session ended. Happy Journey with Indian Railways!")
            break


if __name__ == "__main__":
    asyncio.run(cli())
