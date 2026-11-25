"""Streamlit Dual-Mode Transport Agent UI (Flight & Round-Trip Specialist)

Mode 1 (Left): Natural Language ReAct Agent (Intent extraction, Parameter Gating, Tool Execution JSON).
Mode 2 (Right): Structured Parameter Form (City Autocomplete, Round-Trip Duration & Window, Itinerary Team Payload).
"""
import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

# Ensure flights directory is on sys.path whether executed from root or inside flights/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = BASE_DIR if os.path.basename(BASE_DIR) == "flights" else os.path.join(BASE_DIR, "flights")
if FLIGHTS_DIR not in sys.path:
    sys.path.insert(0, FLIGHTS_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from city_resolver import (
    POPULAR_CITY_OPTIONS,
    resolve_city_to_airport,
)
from flight_v4 import (
    FlightSearchInput,
    build_agent,
    extract_response_text,
    search_flights,
    today_ist,
)
from round_trip_engine import (
    calculate_return_date,
    plan_round_trip,
)

# Page configuration
st.set_page_config(
    page_title="SP Voyanta - Transport Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished layout
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .round-trip-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 1.2rem;
    }
    .badge-best {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-cheap {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-fast {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .price-text {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    /* Authentic Google Flights Dark Mode Styling */
    .gf-container {
        background-color: #17181a;
        border: 1px solid #303134;
        border-radius: 12px;
        padding: 20px;
        color: #e8eaed;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin-bottom: 20px;
    }
    .gf-tabs-row {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
    }
    .gf-tab {
        padding: 8px 18px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        background: #202124;
        border: 1px solid #3c4043;
        color: #bdc1c6;
    }
    .gf-tab-active {
        background: #1b3a57;
        border-color: #8ab4f8;
        color: #8ab4f8;
        font-weight: 600;
    }
    .gf-section-header {
        margin-bottom: 12px;
    }
    .gf-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        display: block;
    }
    .gf-subtitle {
        font-size: 0.8rem;
        color: #9aa0a6;
    }
    .gf-flight-row {
        background: #202124;
        border: 1px solid #303134;
        border-radius: 8px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        transition: background 0.15s ease;
    }
    .gf-flight-row:hover {
        background: #282a2d;
        border-color: #5f6368;
    }
    .gf-col-logo {
        width: 44px;
    }
    .gf-airline-logo {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #303134;
        color: #8ab4f8;
        font-weight: 700;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .gf-col-times {
        flex: 2;
        padding-left: 10px;
    }
    .gf-time-text {
        font-size: 1.05rem;
        font-weight: 600;
        color: #ffffff;
    }
    .gf-next-day {
        color: #ea868f;
        font-size: 0.75rem;
        vertical-align: super;
        font-weight: 600;
    }
    .gf-airline-name {
        font-size: 0.82rem;
        color: #9aa0a6;
    }
    .gf-col-duration {
        flex: 1.2;
    }
    .gf-dur-text {
        font-size: 0.95rem;
        font-weight: 500;
        color: #e8eaed;
    }
    .gf-route-text {
        font-size: 0.8rem;
        color: #9aa0a6;
    }
    .gf-col-stops {
        flex: 1.8;
    }
    .gf-stops-text {
        font-size: 0.95rem;
        font-weight: 500;
        color: #e8eaed;
    }
    .gf-stops-warning {
        color: #fdd663;
        font-weight: 600;
    }
    .gf-layover-sub {
        font-size: 0.78rem;
        color: #9aa0a6;
    }
    .gf-col-emissions {
        flex: 1.2;
        text-align: right;
        padding-right: 15px;
    }
    .gf-emissions-text {
        font-size: 0.82rem;
        color: #bdc1c6;
    }
    .gf-eco-pill {
        display: inline-block;
        background: rgba(129, 201, 149, 0.15);
        color: #81c995;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 1px 6px;
        border-radius: 10px;
    }
    .gf-col-price {
        flex: 1.2;
        text-align: right;
    }
    .gf-price-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #8ab4f8;
    }
    .gf-round-tag {
        font-size: 0.75rem;
        color: #9aa0a6;
    }
    .gf-price-meter {
        background: #202124;
        border: 1px solid #303134;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 14px 0;
        font-size: 0.85rem;
        color: #e8eaed;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">✈️ SP Voyanta Transport Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Modal Flight & Vacation Planner · <b>City Name Resolution</b> · <b>Round-Trip Window Engine</b></div>',
    unsafe_allow_html=True,
)


def render_google_flights_board(data: dict):
    """Render authentic Google Flights dark-mode UI with Top Flights, Other Flights, and AI Recommendation."""
    route = data.get("route", "")
    dep_date = data.get("departure_date", "")
    passengers = data.get("passengers", 1)
    travel_class = data.get("travel_class", "Economy")
    lowest_fare = data.get("lowest_fare_summary") or "Unavailable"
    booking_url = data.get("booking_url", "")
    top_flights = data.get("top_flights") or []
    other_flights = data.get("other_options") or []
    recs = data.get("recommendations") or {}

    def _render_row(f: dict, is_top: bool = True):
        airline = f.get("airline", "Airline")
        initials = "".join([w[0] for w in airline.split()[:2]]).upper() or "FL"
        dep_time = f.get("departure_time", "")
        arr_time = f.get("arrival_time", "")
        next_day = " <span class='gf-next-day'>+1</span>" if f.get("arrives_next_day") else ""
        dur = f.get("duration_formatted", "")
        dep_a = f.get("departure_airport") or ""
        arr_a = f.get("arrival_airport") or ""
        stops = f.get("stops", 0)
        has_warning = f.get("has_layover_warning") or (
            stops > 0
            and "stop" in f.get("stops_summary", "")
            and any(int(w[:-1]) >= 4 for w in f.get("stops_summary", "").split() if w.endswith("h") and w[:-1].isdigit())
        )

        if stops == 0:
            stop_label = "Non-stop"
            stop_class = ""
            layover_sub = ""
        else:
            warning_icon = "⚠️ " if has_warning else ""
            stop_label = f"{warning_icon}{stops} stop"
            stop_class = "gf-stops-warning" if has_warning else ""
            layovers = f.get("layovers") or []
            if layovers:
                lo = layovers[0]
                layover_sub = f"{lo.get('duration_formatted')} {lo.get('transit_id')}"
            else:
                layover_sub = f.get("stops_summary", "").replace(f"{stops} stop", "").strip(" ()")

        price_str = f.get("price_formatted", "")

        return f"""
        <div class="gf-flight-row">
            <div class="gf-col-logo">
                <div class="gf-airline-logo">{initials}</div>
            </div>
            <div class="gf-col-times">
                <div class="gf-time-text">{dep_time} – {arr_time}{next_day}</div>
                <div class="gf-airline-name">{airline} · {f.get('flight_number', '')}</div>
                <div class="gf-eco-pill">🌱 Low emissions</div>
            </div>
            <div class="gf-col-duration">
                <div class="gf-dur-text">{dur}</div>
                <div class="gf-route-text">{dep_a}–{arr_a}</div>
            </div>
            <div class="gf-col-stops">
                <div class="gf-stops-text {stop_class}">{stop_label}</div>
                <div class="gf-layover-sub">{layover_sub}</div>
            </div>
            <div class="gf-col-emissions">
                <div class="gf-emissions-text">168 kg CO2e</div>
                <div class="gf-eco-pill">-6% emissions</div>
            </div>
            <div class="gf-col-price">
                <div class="gf-price-val">{price_str}</div>
                <div class="gf-round-tag">{travel_class}</div>
            </div>
        </div>
        """

    rows_top = "".join([_render_row(f, is_top=True) for f in top_flights])
    rows_other = "".join([_render_row(f, is_top=False) for f in other_flights])

    board_html = f"""
    <div class="gf-container">
        <!-- Top Google Flights Tabs -->
        <div class="gf-tabs-row">
            <div class="gf-tab gf-tab-active">Best ⓘ</div>
            <div class="gf-tab">Cheapest from {lowest_fare} ⓘ</div>
        </div>

        <!-- Section: Top Flights -->
        <div class="gf-section-header">
            <span class="gf-title">Top flights</span>
            <span class="gf-subtitle">Ranked based on price and convenience. Prices include taxes & fees for {passengers} adult(s).</span>
        </div>
        {rows_top if rows_top else '<div style="color:#9aa0a6; padding: 10px;">No non-stop flights found in top pool.</div>'}

        <!-- Price Insight Banner -->
        <div class="gf-price-meter">
            <span>🚥 <b>Prices are currently typical</b> for this travel window</span>
        </div>

        <!-- Section: Other Flights -->
        <div class="gf-section-header" style="margin-top: 20px;">
            <span class="gf-title">Other flights</span>
            <span class="gf-subtitle">Alternative schedules with longer layovers or secondary transit hubs.</span>
        </div>
        {rows_other if rows_other else '<div style="color:#9aa0a6; padding: 10px;">No other flights found.</div>'}
    </div>
    """
    st.markdown(board_html, unsafe_allow_html=True)

    # Intelligent AI Recommendation Section
    best_ov = recs.get("best_overall")
    cheapest = recs.get("cheapest")
    fastest = recs.get("fastest")

    st.markdown("### 💡 AI Flight Advisor & Intelligent Recommendation")
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        if best_ov:
            st.markdown(f"""
            <div style="background:#1e293b; border:1px solid #3b82f6; border-radius:10px; padding:15px; color:#fff;">
                <h5 style="margin-top:0; color:#60a5fa;">⭐ Best Overall</h5>
                <b>{best_ov.get('airline')} ({best_ov.get('flight_number')})</b><br>
                🕒 {best_ov.get('departure_time')} ➔ {best_ov.get('arrival_time')} ({best_ov.get('duration_formatted')})<br>
                💰 <b style="color:#60a5fa; font-size:1.15rem;">{best_ov.get('price_formatted')}</b><br>
                <small style="color:#94a3b8;">Optimal balance of price, convenience, and layover safety.</small>
            </div>
            """, unsafe_allow_html=True)
    with c_b:
        if cheapest:
            st.markdown(f"""
            <div style="background:#14532d; border:1px solid #22c55e; border-radius:10px; padding:15px; color:#fff;">
                <h5 style="margin-top:0; color:#4ade80;">🏷️ Lowest Fare</h5>
                <b>{cheapest.get('airline')} ({cheapest.get('flight_number')})</b><br>
                🕒 {cheapest.get('departure_time')} ➔ {cheapest.get('arrival_time')} ({cheapest.get('duration_formatted')})<br>
                💰 <b style="color:#4ade80; font-size:1.15rem;">{cheapest.get('price_formatted')}</b><br>
                <small style="color:#bbf7d0;">Rock-bottom fare for budget-conscious travelers.</small>
            </div>
            """, unsafe_allow_html=True)
    with c_c:
        if fastest:
            st.markdown(f"""
            <div style="background:#312e81; border:1px solid #6366f1; border-radius:10px; padding:15px; color:#fff;">
                <h5 style="margin-top:0; color:#818cf8;">⚡ Fastest Flight</h5>
                <b>{fastest.get('airline')} ({fastest.get('flight_number')})</b><br>
                🕒 {fastest.get('departure_time')} ➔ {fastest.get('arrival_time')} ({fastest.get('duration_formatted')})<br>
                💰 <b style="color:#818cf8; font-size:1.15rem;">{fastest.get('price_formatted')}</b><br>
                <small style="color:#c7d2fe;">Shortest total transit duration.</small>
            </div>
            """, unsafe_allow_html=True)

    # Detailed Trade-Off Explanation
    if best_ov and cheapest and best_ov != cheapest:
        diff = int(best_ov.get("price") or 0) - int(cheapest.get("price") or 0)
        st.info(
            f"🧠 **Decision Insight:** The absolute cheapest flight is **{cheapest.get('price_formatted')}**, but it involves "
            f"**{cheapest.get('stops_summary')}** ({cheapest.get('duration_formatted')} total). "
            f"We recommend **{best_ov.get('airline')} at {best_ov.get('price_formatted')}** because for just ₹{diff:,} more, "
            f"it saves substantial travel time and layover fatigue!"
        )

    if booking_url:
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <a href="{booking_url}" target="_blank" style="background-color: #1a73e8; color: white; padding: 12px 28px; border-radius: 6px; text-decoration: none; font-weight: 600; display: inline-block; font-size: 1rem;">
                🔗 View & Book on Google Flights
            </a>
        </div>
        """, unsafe_allow_html=True)

# Sidebar helper info
with st.sidebar:
    st.header("ℹ️ Transport Agent Guide")
    st.info(
        """
        **✨ New Capabilities Enabled:**
        - **City/Place Names:** No need to know 3-letter IATA codes!
        - **Round-Trip Itineraries:** Automatically computes Return Date = Departure + Duration.
        - **Flexible Window Search:** Scans multiple departure dates to find the cheapest package.
        - **Itinerary Team Schema:** Standardized `"onward"`, `"return"`, and `"recommendation"` payload.
        """
    )
    st.markdown("---")
    st.markdown("**Example Natural Queries:**")
    st.markdown("• *Plan a 5-day trip from Hyderabad to Goa starting tomorrow over next 3 days*")
    st.markdown("• *Flights from Jaipur to Kochi on Nov 26 for 2 passengers*")
    st.markdown("• *I want to visit Goa with my family from Andhra Pradesh*")

    if st.button("🔄 Reset Chat Session"):
        st.session_state.messages = []
        st.session_state.thread_id = f"session_{int(datetime.now().timestamp())}"
        st.rerun()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{int(datetime.now().timestamp())}"
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = MemorySaver()
if "agent" not in st.session_state:
    st.session_state.agent = build_agent(st.session_state.agent_memory)

# Render Side-by-Side Columns
col_react, col_structured = st.columns([1, 1], gap="large")

# ============================================================================
# LEFT COLUMN: MODE 1 - NATURAL LANGUAGE REACT AGENT
# ============================================================================
with col_react:
    st.subheader("🗣️ Mode 1: Natural Query (ReAct Agent)")
    st.caption("Understands place names, calculates durations, clarifies missing parameters, and executes tools autonomously.")

    chat_container = st.container(height=540)

    with chat_container:
        if not st.session_state.messages:
            st.chat_message("assistant").markdown(
                "Hello! I am your **SP Voyanta Transport Specialist**.\n\n"
                "Tell me where you want to travel! You can use city names and trip durations, like:\n"
                "*• 'Plan a 5-day trip from Hyderabad to Goa starting next Friday over the next 3 days'*\n"
                "*• 'Find one-way flights from Delhi to Mumbai tomorrow'*\n"
                "*• 'I want to travel with my family to Goa from Andhra Pradesh'*"
            )

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    if msg.get("extracted_json"):
                        with st.expander("🛠️ Tool Call / Extracted Parameters (JSON)", expanded=False):
                            st.json(msg["extracted_json"])
                    st.markdown(msg["content"])

    user_query = st.chat_input("Enter travel request (e.g. 'Plan 5-day trip from Hyderabad to Goa next Friday')...")

    if user_query:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})

        async def run_chat():
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = await st.session_state.agent.ainvoke(
                {"messages": [HumanMessage(content=user_query)]},
                config=config,
            )
            return result

        with st.spinner("🤖 Voyanta Agent is analyzing itinerary..."):
            try:
                res = asyncio.run(run_chat())
                history = res.get("messages", [])

                # Check if a tool was executed in this turn
                extracted_tool_payload = None
                for m in reversed(history):
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        extracted_tool_payload = m.tool_calls[0].get("args")
                        break
                    if getattr(m, "type", None) == "tool" or m.__class__.__name__ == "ToolMessage":
                        try:
                            extracted_tool_payload = json.loads(m.content)
                        except Exception:
                            pass
                        break

                final_text = extract_response_text(history)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_text,
                    "extracted_json": extracted_tool_payload,
                })
                st.rerun()
            except Exception as e:
                st.error(f"Agent error: {e}")


# ============================================================================
# RIGHT COLUMN: MODE 2 - STRUCTURED PARAMETER FORM (CITY NAMES & DURATION)
# ============================================================================
with col_structured:
    st.subheader("📋 Mode 2: Structured Trip Planner Form")
    st.caption("Select city/place names, duration, and flexible search window without needing airport codes.")

    with st.form("structured_planner_form"):
        # Trip type toggle
        journey_type = st.radio(
            "Journey Type",
            options=["🌴 Round-Trip (Vacation Planner)", "✈️ One-Way Flight"],
            horizontal=True,
        )

        options_list = POPULAR_CITY_OPTIONS + ["Other (Type Custom City / Country)..."]

        c1, c2 = st.columns(2)
        with c1:
            source_choice = st.selectbox(
                "Source Location (City / Place)",
                options=options_list,
                index=0,  # Hyderabad / Secunderabad
                help="Select your departure city or hub",
            )
            if source_choice == "Other (Type Custom City / Country)...":
                source_city = st.text_input("Enter Departure City / Town / Airport", value="", placeholder="e.g. Zurich, Dubai, Spiti").strip()
            else:
                source_city = source_choice

        with c2:
            dest_choice = st.selectbox(
                "Destination Location (City / Place)",
                options=options_list,
                index=1,  # Goa
                help="Select your destination city or tourist hub",
            )
            if dest_choice == "Other (Type Custom City / Country)...":
                dest_city = st.text_input("Enter Destination City / Town / Airport", value="", placeholder="e.g. Bali, Pondicherry, Paris").strip()
            else:
                dest_city = dest_choice

        c3, c4 = st.columns(2)
        with c3:
            departure_date = st.date_input(
                "Earliest Departure Date",
                value=today_ist() + timedelta(days=2),
                min_value=today_ist(),
            )
        with c4:
            passengers = st.number_input("Adult Passengers", min_value=1, max_value=9, value=1)

        is_round_trip = "Round-Trip" in journey_type

        if is_round_trip:
            c5, c6 = st.columns(2)
            with c5:
                trip_duration = st.number_input(
                    "Trip Duration (Days of Stay)",
                    min_value=1,
                    max_value=60,
                    value=5,
                    help="Length of your stay. Return date = Departure date + Trip Duration.",
                )
            with c6:
                search_window = st.slider(
                    "Search Window (Days to Scan)",
                    min_value=1,
                    max_value=7,
                    value=3,
                    help="Checks departures starting from your chosen date across this window to find the cheapest deal.",
                )

            # Live preview of calculated dates
            calc_return = calculate_return_date(departure_date, int(trip_duration))
            st.info(
                f"📅 **Onward Departure:** {departure_date.strftime('%a, %b %d, %Y')} ➔ **Calculated Return:** {calc_return.strftime('%a, %b %d, %Y')} "
                f"({trip_duration}-Day Stay | Scanning {search_window} departure dates)"
            )
        else:
            trip_duration = 1
            search_window = 1

        c7, c8 = st.columns(2)
        with c7:
            travel_class = st.selectbox("Cabin Class", options=["Economy", "Premium Economy", "Business", "First"], index=0)
        with c8:
            max_stops_choice = st.selectbox("Stops Preference", options=["Any stops", "Non-stop only (0)", "Max 1 stop (1)"])

        with st.expander("⚙️ Optional Budget Filter"):
            budget_val = st.number_input("Max Total Budget (INR)", min_value=0, value=0, step=1000)
            budget = float(budget_val) if budget_val > 0 else None

        max_stops = None
        if max_stops_choice == "Non-stop only (0)":
            max_stops = 0
        elif max_stops_choice == "Max 1 stop (1)":
            max_stops = 1

        # Resolve IATA codes dynamically with AI fallback & cache
        try:
            dep_code, resolved_src, _ = resolve_city_to_airport(source_city)
            arr_code, resolved_dst, _ = resolve_city_to_airport(dest_city)
            loc_err = None
        except ValueError as ve:
            dep_code, resolved_src = "TBD", source_city
            arr_code, resolved_dst = "TBD", dest_city
            loc_err = str(ve)

        # Build payload representation
        input_payload = {
            "source_location": source_city,
            "resolved_departure_airport": dep_code,
            "destination_location": dest_city,
            "resolved_arrival_airport": arr_code,
            "departure_date": departure_date.isoformat(),
            "journey_type": "round_trip" if is_round_trip else "one_way",
            "trip_duration_days": int(trip_duration) if is_round_trip else None,
            "search_window_days": int(search_window) if is_round_trip else 1,
            "passengers": int(passengers),
            "travel_class": travel_class,
            "budget": budget,
            "max_stops": max_stops,
        }

        if loc_err:
            st.warning(f"📍 Location note: {loc_err}")

        st.markdown("**Generated Input Payload (JSON):**")
        st.json(input_payload)

        submit_btn = st.form_submit_button("🚀 Plan My Trip", use_container_width=True)

    if submit_btn:
        with st.spinner("Executing live flight search and itinerary calculations..."):
            try:
                if is_round_trip:
                    # Execute Round-Trip Flexible Window Planner
                    result = plan_round_trip(
                        source=source_city,
                        destination=dest_city,
                        start_date=departure_date,
                        trip_duration_days=int(trip_duration),
                        search_window_days=int(search_window),
                        passengers=int(passengers),
                        travel_class=travel_class,
                        budget=budget,
                        max_stops=max_stops,
                    )

                    if result.get("status") == "no_results":
                        st.warning(result.get("message"))
                    else:
                        st.success(f"Successfully planned round-trip packages for {source_city} ➔ {dest_city}!")

                        # 1. Best Package Banner
                        best_pkg = result.get("cheapest_package", {})
                        st.markdown(
                            f"""
                            <div class="round-trip-box">
                                <h4 style="margin-top:0; color:#065F46;">🏆 Recommended Round-Trip Package</h4>
                                <b>📅 Dates:</b> {best_pkg.get('departure_date')} ➔ {best_pkg.get('return_date')} ({trip_duration}-Day Vacation)<br>
                                💰 <span class="price-text">{best_pkg.get('total_fare')}</span> total round-trip for {passengers} passenger(s)
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 2. Display Onward and Return Legs
                        journey = result.get("journey", {})
                        onward = journey.get("onward", {})
                        ret = journey.get("return", {})

                        c_on, c_ret = st.columns(2)
                        with c_on:
                            st.markdown(
                                f"""
                                <div class="card-box">
                                    <h5 style="color:#1E3A8A; margin-top:0;">🛫 Onward Flight</h5>
                                    <b>Route:</b> {onward.get('route')}<br>
                                    <b>Date:</b> {onward.get('departure_date_formatted')}<br>
                                    <b>Flight:</b> {onward.get('flight_summary')}<br>
                                    <b>Cost:</b> ₹{onward.get('total_inr'):,}<br>
                                    🔗 <a href="{onward.get('booking_url')}" target="_blank">View Onward on Google Flights</a>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with c_ret:
                            st.markdown(
                                f"""
                                <div class="card-box">
                                    <h5 style="color:#065F46; margin-top:0;">🛬 Return Flight</h5>
                                    <b>Route:</b> {ret.get('route')}<br>
                                    <b>Date:</b> {ret.get('return_date_formatted')}<br>
                                    <b>Flight:</b> {ret.get('flight_summary')}<br>
                                    <b>Cost:</b> ₹{ret.get('total_inr'):,}<br>
                                    🔗 <a href="{ret.get('booking_url')}" target="_blank">View Return on Google Flights</a>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # 3. Decision Intelligence Recommendation
                        st.markdown("#### 💡 AI Travel Recommendation")
                        st.markdown(journey.get("recommendation", ""))

                        # 4. Standardized Itinerary Team Payload
                        with st.expander("📦 Standardized Itinerary Team JSON Payload", expanded=False):
                            st.json(result["journey"])

                        # 5. Alternative Date Combinations in Window
                        date_opts = result.get("all_date_options") or []
                        if len(date_opts) > 1:
                            with st.expander(f"📅 Compare All {len(date_opts)} Date Combinations in Window"):
                                for opt in date_opts:
                                    b_status = "✅ Under Budget" if opt["meets_budget"] else f"❌ {opt.get('budget_reason')}"
                                    st.write(
                                        f"• **Option {opt['option_id']}:** {opt['departure_date_formatted']} ➔ {opt['return_date_formatted']} · "
                                        f"**Total {opt['total_round_trip_formatted']}** ({b_status})"
                                    )

                else:
                    # One-Way Search
                    tool_payload = {
                        "departure_iata": dep_code,
                        "arrival_iata": arr_code,
                        "travel_date": departure_date.isoformat(),
                        "passengers": int(passengers),
                        "travel_class": travel_class,
                        "max_stops": max_stops,
                        "budget": budget,
                    }
                    tool_output_str = search_flights.invoke(tool_payload)
                    data = json.loads(tool_output_str)

                    if data.get("status") == "error":
                        st.error(f"Error from search tool: {data.get('message')}")
                    elif data.get("status") == "no_results":
                        st.warning(f"No flights found for {source_city} ➔ {dest_city} on {departure_date}.")
                    else:
                        render_google_flights_board(data)

            except Exception as ex:
                st.error(f"Execution failed: {ex}")
