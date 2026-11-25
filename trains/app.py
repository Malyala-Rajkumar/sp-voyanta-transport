"""Streamlit Dual-Mode Train Booking UI (IRCTC & RailKit Multi-Modal Specialist)

Mode 1 (Left): Natural Language ReAct Train Agent (Strict Parameter Gating, Live Berths & Inline Visual Cards).
Mode 2 (Right): Structured IRCTC Parameter Form (Cluster Autocomplete, Vacation Stay Duration, Flexible Window & Verified Fares).
"""
import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from station_resolver import (
    POPULAR_CITY_STATION_OPTIONS,
    resolve_city_to_station,
)
from train_round_trip_engine import (
    calculate_return_date,
    plan_round_trip_trains,
)
from train_agent_v2 import (
    build_train_agent,
    extract_response_text,
    search_trains,
    today_ist,
)

# Page configuration
st.set_page_config(
    page_title="SP Voyanta - Train Booking Agent",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Authentic IRCTC Dark Theme Styling (Identical to Google Flights Design Architecture)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0284c7;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    /* Authentic Google Flights / IRCTC Dark Mode Container */
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
        background: #0c4a6e;
        border-color: #38bdf8;
        color: #38bdf8;
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
    
    /* Train Row Card */
    .gf-train-row {
        background: #202124;
        border: 1px solid #303134;
        border-radius: 8px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .gf-train-row:hover {
        background: #282a2d;
        border-color: #38bdf8;
    }
    .gf-col-logo {
        width: 60px;
    }
    .gf-train-num-badge {
        background: #0284c7;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 4px 8px;
        border-radius: 5px;
        display: inline-block;
        text-align: center;
    }
    .gf-col-times {
        flex: 2.2;
        padding-left: 10px;
    }
    .gf-time-text {
        font-size: 1.05rem;
        font-weight: 600;
        color: #ffffff;
    }
    .gf-train-name {
        font-size: 0.82rem;
        color: #9aa0a6;
    }
    .gf-pill-superfast {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 1px 7px;
        border-radius: 10px;
        margin-top: 3px;
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
    .gf-seat-avail {
        color: #4ade80;
        font-weight: 700;
        font-size: 0.82rem;
        background: rgba(74, 222, 128, 0.12);
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
    }
    .gf-seat-wl {
        color: #fb923c;
        font-weight: 700;
        font-size: 0.82rem;
        background: rgba(251, 146, 60, 0.12);
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
    }
    .gf-col-price {
        flex: 1.2;
        text-align: right;
    }
    .gf-price-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .gf-class-tag {
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
    .card-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 1rem;
        color: #f8fafc;
    }
    .round-trip-box {
        background-color: #064e3b;
        border: 1px solid #059669;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        color: #ecfdf5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Bar
st.markdown('<div class="main-header">🚆 SP Voyanta · IRCTC Train Booking Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Terminal Cluster Routing (SC/HYB/KCG) · Vacation Stay Planner · Intelligent Live Berth Ranking</div>',
    unsafe_allow_html=True,
)


def render_irctc_train_board(data: dict):
    """Render authentic Google-Flights-style dark mode UI for Indian Railways with Top Trains, Other Trains, and AI Advisor."""
    route = data.get("route", "")
    dep_date = data.get("journey_date") or data.get("departure_date", "")
    lowest_fare = data.get("lowest_fare_summary") or "Unavailable"
    booking_url = data.get("booking_url", "https://www.irctc.co.in/nget/train-search")
    
    top_trains = data.get("top_trains") or data.get("trains", [])[:4]
    other_trains = data.get("other_options") or data.get("trains", [])[4:10]
    recs = data.get("recommendations") or {}

    def _render_row(t: dict, is_top: bool = True):
        t_no = t.get("train_number") or t.get("train_no") or "---"
        t_name = t.get("train_name") or "Express"
        dep_time = t.get("departure_time") or t.get("from_time") or "--:--"
        arr_time = t.get("arrival_time") or t.get("to_time") or "--:--"
        from_code = t.get("from_station_code") or t.get("from_stn_code") or ""
        to_code = t.get("to_station_code") or t.get("to_stn_code") or ""
        dur = t.get("duration") or t.get("travel_time") or ""
        dist = t.get("distance_km") or t.get("distance") or ""
        dist_str = f" · {dist} km" if dist else ""
        runs = t.get("running_days") or t.get("daysOfDeparture") or "Daily"
        
        status_txt = t.get("seat_status") or "Available"
        is_avail = "avl" in status_txt.lower() or "avail" in status_txt.lower()
        status_html = f'<span class="gf-seat-avail">🟢 {status_txt}</span>' if is_avail else f'<span class="gf-seat-wl">🟠 {status_txt}</span>'
        
        price_val = t.get("price")
        price_str = t.get("price_formatted") or (f"₹{price_val:,}" if price_val else "Check on IRCTC")
        target_cls = t.get("target_class") or "3A"

        return f"""
        <div class="gf-train-row">
            <div class="gf-col-logo">
                <span class="gf-train-num-badge">{t_no}</span>
            </div>
            <div class="gf-col-times">
                <div class="gf-time-text">{dep_time} – {arr_time}</div>
                <div class="gf-train-name">{t_name} · {from_code}–{to_code}</div>
                <div class="gf-pill-superfast">⚡ Runs: {runs}</div>
            </div>
            <div class="gf-col-duration">
                <div class="gf-dur-text">{dur}</div>
                <div class="gf-route-text">{from_code}➔{to_code}{dist_str}</div>
            </div>
            <div class="gf-col-stops">
                <div class="gf-stops-text">{status_html}</div>
                <div style="font-size:0.75rem; color:#9aa0a6;">Conf: {t.get('confirmation_prediction', 'High')}</div>
            </div>
            <div class="gf-col-price">
                <div class="gf-price-val">{price_str}</div>
                <div class="gf-class-tag">{target_cls} · General</div>
            </div>
        </div>
        """

    rows_top = "".join([_render_row(t, is_top=True) for t in top_trains])
    rows_other = "".join([_render_row(t, is_top=False) for t in other_trains])

    # board_html = f"""
    # <div class="gf-container">
    #     <!-- Top Google Flights / IRCTC Style Tabs -->
    #     <div class="gf-tabs-row">
    #         <div class="gf-tab gf-tab-active">Best Trains ⓘ</div>
    #         <div class="gf-tab">Cheapest from {lowest_fare} ⓘ</div>
    #     </div>

    #     <!-- Section: Top Trains -->
    #     <div class="gf-section-header">
    #         <span class="gf-title">Top trains</span>
    #         <span class="gf-subtitle">Ranked based on price, convenience, and confirmed berth availability. Route: {route} | 📅 {dep_date}</span>
    #     </div>
    #     {rows_top if rows_top else '<div style="color:#9aa0a6; padding: 10px;">No top trains found.</div>'}

    #     <!-- Price Insight Banner -->
    #     <div class="gf-price-meter">
    #         <span>🚥 <b>Fares are currently typical for Indian Railways (IRCTC)</b> on this route</span>
    #     </div>

    #     <!-- Section: Other Trains -->
    #     <div class="gf-section-header" style="margin-top: 20px;">
    #         <span class="gf-title">Other trains</span>
    #         <span class="gf-subtitle">Alternative schedules and express services throughout the day.</span>
    #     </div>
    #     {rows_other if rows_other else '<div style="color:#9aa0a6; padding: 10px;">No other trains found.</div>'}
    # </div>
    # """
    # st.markdown(board_html, unsafe_allow_html=True)

    # Intelligent AI Recommendation Section (3 Cards)
    best_ov = recs.get("best_overall")
    cheapest = recs.get("cheapest")
    fastest = recs.get("fastest")

    st.markdown("### 💡 AI Train Advisor & Intelligent Recommendation")
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        if best_ov:
            st.markdown(f"""
            <div style="background:#1e293b; border:1px solid #38bdf8; border-radius:10px; padding:15px; color:#fff;">
                <h5 style="margin-top:0; color:#38bdf8;">⭐ Best Overall</h5>
                <b>{best_ov.get('train_number')} · {best_ov.get('train_name')}</b><br>
                🕒 {best_ov.get('departure_time')} ➔ {best_ov.get('arrival_time')} ({best_ov.get('duration')})<br>
                💰 <b style="color:#38bdf8; font-size:1.15rem;">{best_ov.get('price_formatted', 'Typical Fare')}</b> ({best_ov.get('target_class')})<br>
                <small style="color:#94a3b8;">Status: <b>{best_ov.get('seat_status', 'Available')}</b> | Optimal balance of timing & seats.</small>
            </div>
            """, unsafe_allow_html=True)
    with c_b:
        if cheapest:
            st.markdown(f"""
            <div style="background:#14532d; border:1px solid #22c55e; border-radius:10px; padding:15px; color:#fff;">
                <h5 style="margin-top:0; color:#4ade80;">🏷️ Lowest Fare</h5>
                <b>{cheapest.get('train_number')} · {cheapest.get('train_name')}</b><br>
                🕒 {cheapest.get('departure_time')} ➔ {cheapest.get('arrival_time')} ({cheapest.get('duration')})<br>
                💰 <b style="color:#4ade80; font-size:1.15rem;">{cheapest.get('price_formatted', 'Lowest Fare')}</b> ({cheapest.get('target_class')})<br>
                <small style="color:#bbf7d0;">Status: <b>{cheapest.get('seat_status', 'Available')}</b> | Rock-bottom verified fare.</small>
            </div>
            """, unsafe_allow_html=True)
    with c_c:
        if fastest:
            st.markdown(f"""
            <div style="background:#312e81; border:1px solid #818cf8; border-radius:10px; padding:15px; color:#fff;">
                <h5 style="margin-top:0; color:#818cf8;">⚡ Fastest Train</h5>
                <b>{fastest.get('train_number')} · {fastest.get('train_name')}</b><br>
                🕒 {fastest.get('departure_time')} ➔ {fastest.get('arrival_time')} ({fastest.get('duration')})<br>
                💰 <b style="color:#818cf8; font-size:1.15rem;">{fastest.get('price_formatted', 'Verified Fare')}</b> ({fastest.get('target_class')})<br>
                <small style="color:#c7d2fe;">Status: <b>{fastest.get('seat_status', 'Available')}</b> | Shortest journey duration.</small>
            </div>
            """, unsafe_allow_html=True)

    # Detailed Decision Insight
    if best_ov and cheapest:
        c_p = cheapest.get('price') or 0
        b_p = best_ov.get('price') or 0
        diff = b_p - c_p
        diff_str = f" for ₹{diff:,} difference" if diff > 0 else ""
        st.info(
            f"🧠 **Decision Insight:** The fastest option is **{fastest.get('train_name') if fastest else best_ov.get('train_name')}** ({fastest.get('duration') if fastest else ''}). "
            f"The most economical option is **{cheapest.get('train_name')} at {cheapest.get('price_formatted')}** in {cheapest.get('target_class')}. "
            f"We recommend **{best_ov.get('train_name')}**{diff_str} because it offers confirmed availability with convenient departure timings!"
        )

    st.markdown(
        f"""
        <div style="text-align: center; margin: 20px 0;">
            <a href="{booking_url}" target="_blank" style="background-color: #0284c7; color: white; padding: 12px 28px; border-radius: 6px; text-decoration: none; font-weight: 600; display: inline-block; font-size: 1rem;">
                🔗 View & Book Tickets on Official IRCTC Portal
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar helper info
with st.sidebar:
    st.header("🚆 Train Agent Guide")
    st.info(
        """
        **✨ Exact Flight-Symmetric Capabilities:**
        - **Strict Parameter Gating:** Asks date & trip type if missing.
        - **Cluster Routing:** Automatically scans `SC, HYB, KCG` or `NDLS, NZM, DLI`.
        - **Google Flights Style Board:** Top Trains, Other Trains, Typical Price Meter.
        - **3 AI Advisor Picks:** ⭐ Best Overall, 🏷️ Lowest Fare, ⚡ Fastest Train.
        - **Live Berth & Fare Verification:** 100% verified RailKit fares and seat counts.
        """
    )
    st.markdown("---")
    st.markdown("**Example Natural Queries:**")
    st.markdown("• *Plan a 5-day round trip from Hyderabad to Vijayawada next Friday for 2 people in 2A*")
    st.markdown("• *Find trains from Khammam to Mahabubabad tomorrow in CC*")
    st.markdown("• *Trains from Delhi to Mumbai next Monday in 3A*")

    if st.button("🔄 Reset Chat Session"):
        st.session_state.train_messages = []
        st.session_state.train_thread_id = f"train_session_{int(datetime.now().timestamp())}"
        st.rerun()

# Initialize Session State
if "train_messages" not in st.session_state:
    st.session_state.train_messages = []
if "train_thread_id" not in st.session_state:
    st.session_state.train_thread_id = f"train_session_{int(datetime.now().timestamp())}"
if "train_agent_memory" not in st.session_state:
    st.session_state.train_agent_memory = MemorySaver()
if "train_agent" not in st.session_state:
    st.session_state.train_agent = build_train_agent(st.session_state.train_agent_memory)

# Render Side-by-Side Columns
col_react, col_structured = st.columns([1, 1], gap="large")

# ============================================================================
# LEFT COLUMN: MODE 1 - NATURAL LANGUAGE REACT TRAIN AGENT
# ============================================================================
with col_react:
    st.subheader("🗣️ Mode 1: Natural Query (IRCTC ReAct Agent)")
    st.caption("Understands Indian cities, enforces strict date gating, and renders authentic IRCTC cards inline.")

    chat_container = st.container(height=580)

    with chat_container:
        if not st.session_state.train_messages:
            st.chat_message("assistant").markdown(
                "Namaste! I am your **IRCTC & RailKit Train Specialist**.\n\n"
                "Tell me where you want to travel! Examples:\n"
                "*• 'Find trains from Khammam to Mahabubabad tomorrow in CC'*\n"
                "*• 'Plan a 4-day trip from Hyderabad to Vijayawada next Friday for 2 people in 2A'*\n"
                "*• 'Check train options to Goa from Secunderabad'*"
            )

        for msg in st.session_state.train_messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    if msg.get("extracted_json"):
                        with st.expander("🛠️ Tool Call / Extracted Parameters (JSON)", expanded=False):
                            st.json(msg["extracted_json"])
                    st.markdown(msg["content"])
                    
                    # Auto-Render Visual Train Board right inside the chat bubble!
                    if msg.get("tool_result") and isinstance(msg["tool_result"], dict):
                        t_res = msg["tool_result"]
                        if "top_trains" in t_res or "trains" in t_res:
                            render_irctc_train_board(t_res)

    user_query = st.chat_input("Enter train request (e.g. 'Trains from Khammam to Mahabubabad tomorrow in CC')...")

    if user_query:
        st.session_state.train_messages.append({"role": "user", "content": user_query})

        async def run_chat():
            config = {"configurable": {"thread_id": st.session_state.train_thread_id}}
            result = await st.session_state.train_agent.ainvoke(
                {"messages": [HumanMessage(content=user_query)]},
                config=config,
            )
            return result

        with st.spinner("🚆 IRCTC Agent is analyzing railway schedules and verifying berths..."):
            try:
                res = asyncio.run(run_chat())
                history = res.get("messages", [])

                extracted_tool_payload = None
                extracted_tool_result = None
                for m in reversed(history):
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        extracted_tool_payload = m.tool_calls[0].get("args")
                    if getattr(m, "type", None) == "tool" or m.__class__.__name__ == "ToolMessage":
                        try:
                            extracted_tool_result = json.loads(m.content)
                        except Exception:
                            pass
                        break

                final_text = extract_response_text(history)
                st.session_state.train_messages.append({
                    "role": "assistant",
                    "content": final_text,
                    "extracted_json": extracted_tool_payload,
                    "tool_result": extracted_tool_result,
                })
                st.rerun()
            except Exception as e:
                st.error(f"Agent error: {e}")


# ============================================================================
# RIGHT COLUMN: MODE 2 - STRUCTURED IRCTC PARAMETER FORM
# ============================================================================
with col_structured:
    st.subheader("📋 Mode 2: Structured Train Planner Form")
    st.caption("Select railway clusters, stay duration, class, and flexible search window without needing manual station codes.")

    with st.form("structured_train_form"):
        journey_type = st.radio(
            "Journey Type",
            options=["🌴 Round-Trip (Vacation Stay Planner)", "🚆 One-Way Train"],
            horizontal=True,
        )

        options_list = POPULAR_CITY_STATION_OPTIONS + ["Other (Type Custom City / Station)..."]

        c1, c2 = st.columns(2)
        with c1:
            source_choice = st.selectbox(
                "Source City / Station Cluster",
                options=options_list,
                index=0,  # Hyderabad / Secunderabad
                help="Select your departure city or railway hub",
            )
            if source_choice.startswith("Other"):
                source_city = st.text_input("Enter Departure City / Town / Station", value="Khammam", placeholder="e.g. Khammam, Hampi, Gokarna").strip()
            else:
                source_city = source_choice

        with c2:
            dest_choice = st.selectbox(
                "Destination City / Station Cluster",
                options=options_list,
                index=1,  # Vijayawada
                help="Select your destination city or railway junction",
            )
            if dest_choice.startswith("Other"):
                dest_city = st.text_input("Enter Destination City / Town / Station", value="Mahabubabad", placeholder="e.g. Mahabubabad, Madgaon, Tirupati").strip()
            else:
                dest_city = dest_choice

        c3, c4 = st.columns(2)
        with c3:
            departure_date = st.date_input(
                "Earliest Departure Date",
                value=today_ist() + timedelta(days=1),
                min_value=today_ist(),
            )
        with c4:
            passengers = st.number_input("Passengers", min_value=1, max_value=6, value=1)

        is_round_trip = "Round-Trip" in journey_type

        if is_round_trip:
            c5, c6 = st.columns(2)
            with c5:
                trip_duration = st.number_input(
                    "Trip Duration (Days of Stay)",
                    min_value=1,
                    max_value=60,
                    value=4,
                    help="Length of your stay. Return date = Departure date + Trip Duration.",
                )
            with c6:
                search_window = st.slider(
                    "Search Window (Days to Scan)",
                    min_value=1,
                    max_value=7,
                    value=3,
                    help="Scans departure dates across this window to prioritize confirmed seats (AVL over WL) and verified fares.",
                )

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
            travel_class = st.selectbox(
                "Travel Class",
                options=["2S (Second Sitting)", "CC (AC Chair Car)", "3A (AC 3 Tier)", "2A (AC 2 Tier)", "SL (Sleeper)", "1A (First AC)", "3E (3 AC Economy)"],
                index=1,
            )
            class_code = travel_class.split()[0]
        with c8:
            quota_choice = st.selectbox("Booking Quota", options=["GN (General Quota)", "TQ (Tatkal Quota)", "LD (Ladies Quota)"], index=0)
            quota_code = quota_choice.split()[0]

        with st.expander("⚙️ Optional Budget Filter"):
            budget_val = st.number_input("Max Total Budget (INR)", min_value=0, value=0, step=200)
            budget = float(budget_val) if budget_val > 0 else None

        # Resolve stations dynamically
        try:
            src_primary, src_clusters, resolved_src, _ = resolve_city_to_station(source_city)
            dst_primary, dst_clusters, resolved_dst, _ = resolve_city_to_station(dest_city)
            loc_err = None
        except ValueError as ve:
            src_primary, src_clusters, resolved_src = "TBD", [], source_city
            dst_primary, dst_clusters, resolved_dst = "TBD", [], dest_city
            loc_err = str(ve)

        input_payload = {
            "source_location": source_city,
            "resolved_primary_station": src_primary,
            "station_clusters_origin": src_clusters,
            "destination_location": dest_city,
            "resolved_destination_station": dst_primary,
            "station_clusters_destination": dst_clusters,
            "departure_date": departure_date.isoformat(),
            "journey_type": "round_trip" if is_round_trip else "one_way",
            "trip_duration_days": int(trip_duration) if is_round_trip else None,
            "search_window_days": int(search_window) if is_round_trip else 1,
            "passengers": int(passengers),
            "travel_class": class_code,
            "quota": quota_code,
            "budget": budget,
        }

        if loc_err:
            st.warning(f"📍 Location note: {loc_err}")

        st.markdown("**Generated Input Payload (JSON):**")
        st.json(input_payload)

        submit_btn = st.form_submit_button("🚀 Plan My Train Journey", use_container_width=True)

    if submit_btn:
        with st.spinner("Querying live RailKit IRCTC schedules and verifying fares..."):
            try:
                if is_round_trip:
                    result = plan_round_trip_trains(
                        source=source_city,
                        destination=dest_city,
                        start_date=departure_date,
                        trip_duration_days=int(trip_duration),
                        search_window_days=int(search_window),
                        travel_class=class_code,
                        quota=quota_code,
                        passengers=int(passengers),
                        budget=budget,
                    )

                    if result.get("status") == "no_results":
                        st.warning(result.get("message"))
                    else:
                        st.success(f"Successfully planned round-trip train packages for {resolved_src} ➔ {resolved_dst}!")

                        best_pkg = result.get("cheapest_package", {})
                        st.markdown(
                            f"""
                            <div class="round-trip-box">
                                <h4 style="margin-top:0; color:#a7f3d0;">🏆 Recommended Round-Trip Train Package</h4>
                                <b>📅 Travel Dates:</b> {best_pkg.get('departure_date')} ➔ {best_pkg.get('return_date')} ({trip_duration}-Day Stay)<br>
                                💰 <span style="font-size:1.35rem; font-weight:700; color:#38bdf8;">{best_pkg.get('total_fare')}</span> total round-trip for {passengers} passenger(s) in {class_code}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        journey = result.get("journey", {})
                        onward = journey.get("onward", {})
                        ret = journey.get("return", {})

                        c_on, c_ret = st.columns(2)
                        with c_on:
                            on_avail = onward.get("seat_status", "Available")
                            st.markdown(
                                f"""
                                <div class="card-box">
                                    <h5 style="color:#38bdf8; margin-top:0;">🚆 Onward Train ({onward.get('departure_date_formatted')})</h5>
                                    <b>Route:</b> {onward.get('from_station')} ➔ {onward.get('to_station')}<br>
                                    <b>Train:</b> {onward.get('train_name')} ({onward.get('train_number')})<br>
                                    <b>Timings:</b> {onward.get('departure_time')} ➔ {onward.get('arrival_time')} (⏱️ {onward.get('duration')})<br>
                                    <b>Class & Quota:</b> {class_code} · {quota_code}<br>
                                    <b>Seat Status:</b> <span class="gf-seat-avail">{on_avail}</span> ({onward.get('confirmation_prediction', 'Confirmed')})<br>
                                    <b>Cost:</b> ₹{onward.get('total_fare_inr', 0):,}<br>
                                    <div style="margin-top:8px;">
                                        <a href="{onward.get('booking_url')}" target="_blank" style="color:#38bdf8; font-weight:600; font-size:0.85rem;">🔗 Book Onward on IRCTC</a>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with c_ret:
                            ret_avail = ret.get("seat_status", "Available")
                            st.markdown(
                                f"""
                                <div class="card-box">
                                    <h5 style="color:#34d399; margin-top:0;">🚆 Return Train ({ret.get('return_date_formatted')})</h5>
                                    <b>Route:</b> {ret.get('from_station')} ➔ {ret.get('to_station')}<br>
                                    <b>Train:</b> {ret.get('train_name')} ({ret.get('train_number')})<br>
                                    <b>Timings:</b> {ret.get('departure_time')} ➔ {ret.get('arrival_time')} (⏱️ {ret.get('duration')})<br>
                                    <b>Class & Quota:</b> {class_code} · {quota_code}<br>
                                    <b>Seat Status:</b> <span class="gf-seat-avail">{ret_avail}</span> ({ret.get('confirmation_prediction', 'Confirmed')})<br>
                                    <b>Cost:</b> ₹{ret.get('total_fare_inr', 0):,}<br>
                                    <div style="margin-top:8px;">
                                        <a href="{ret.get('booking_url')}" target="_blank" style="color:#34d399; font-weight:600; font-size:0.85rem;">🔗 Book Return on IRCTC</a>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("#### 💡 AI Train Travel Advisor")
                        st.markdown(journey.get("recommendation", ""))

                        with st.expander("📦 Standardized Itinerary Team JSON Payload", expanded=False):
                            st.json(result["journey"])

                else:
                    # One-Way Train Search
                    tool_payload = {
                        "source": source_city,
                        "destination": dest_city,
                        "journey_date": departure_date.isoformat(),
                        "travel_class": class_code,
                        "quota": quota_code,
                    }
                    tool_output_str = search_trains.invoke(tool_payload)
                    data = json.loads(tool_output_str)

                    if data.get("status") == "error":
                        st.error(f"Error from search tool: {data.get('message')}")
                    elif data.get("status") == "no_results":
                        st.warning(f"No direct trains found for {source_city} ➔ {dest_city} on {departure_date}.")
                    else:
                        render_irctc_train_board(data)

            except Exception as ex:
                st.error(f"Execution failed: {ex}")
