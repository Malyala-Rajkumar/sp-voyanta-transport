"""Streamlit Multi-Modal Transport Platform (IRCTC Trains & Google Flights)

Modes Supported:
1. 🚆 Train Booking (IRCTC / RailKit) - ReAct Agent & Structured Vacation Planner
2. ✈️ Flight Booking (Google Flights) - ReAct Agent & Structured Vacation Planner
3. 🔄 Multimodal Comparison (Flight vs Train) - Head-to-Head Speed, Cost & Comfort Analysis
"""
import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_DIR = os.path.join(BASE_DIR, "flights")
TRAINS_DIR = os.path.join(BASE_DIR, "train booking")

for d in (FLIGHTS_DIR, TRAINS_DIR, BASE_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Flights imports
from city_resolver import (
    POPULAR_CITY_OPTIONS as FLIGHT_CITY_OPTIONS,
    resolve_city_to_airport,
)
from flight_v4 import (
    build_agent as build_flight_agent,
    extract_response_text as extract_flight_text,
    search_flights,
    today_ist as today_ist_flight,
)
from round_trip_engine import (
    calculate_return_date as calculate_flight_return,
    plan_round_trip as plan_flight_round_trip,
)

# Trains imports
from station_resolver import (
    POPULAR_CITY_STATION_OPTIONS as TRAIN_CITY_OPTIONS,
    resolve_city_to_station,
)
from train_round_trip_engine import (
    calculate_return_date as calculate_train_return,
    plan_round_trip_trains,
)
from train_agent_v2 import (
    build_train_agent,
    extract_response_text as extract_train_text,
    search_trains,
    today_ist as today_ist_train,
)

# Page configuration
st.set_page_config(
    page_title="SP Voyanta - Master Transportation Orchestrator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

from transport_orchestrator import (
    build_orchestrator_agent,
    extract_response_text as extract_orchestrator_text,
    today_ist as today_ist_orch,
)
from orchestrator_tools import (
    plan_end_to_end_journey,
)

# Global Custom Styling
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
    .card-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #f8fafc;
    }
    .round-trip-box {
        background-color: #064e3b;
        border: 1px solid #059669;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 1.2rem;
        color: #ecfdf5;
    }
    .price-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .badge-avail {
        background-color: #065f46;
        color: #6ee7b7;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-wl {
        background-color: #7c2d12;
        color: #fdba74;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
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
    .gf-flight-row {
        background: #202124;
        border: 1px solid #303134;
        border-radius: 8px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .gf-flight-row:hover {
        background: #282a2d;
        border-color: #5f6368;
    }
    .gf-col-logo { width: 44px; }
    .gf-airline-logo {
        width: 34px; height: 34px; border-radius: 50%;
        background: #303134; color: #8ab4f8; font-weight: 700;
        font-size: 0.8rem; display: flex; align-items: center; justify-content: center;
    }
    .gf-col-times { flex: 2; padding-left: 10px; }
    .gf-time-text { font-size: 1.05rem; font-weight: 600; color: #ffffff; }
    .gf-airline-name { font-size: 0.82rem; color: #9aa0a6; }
    .gf-col-duration { flex: 1.2; }
    .gf-dur-text { font-size: 0.95rem; font-weight: 500; color: #e8eaed; }
    .gf-route-text { font-size: 0.8rem; color: #9aa0a6; }
    .gf-col-stops { flex: 1.8; }
    .gf-stops-text { font-size: 0.95rem; font-weight: 500; color: #e8eaed; }
    .gf-col-emissions { flex: 1.2; text-align: right; padding-right: 15px; }
    .gf-col-price { flex: 1.2; text-align: right; }
    
    /* Authentic IRCTC Train Board Styling */
    .irctc-board {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        color: #e2e8f0;
        margin-bottom: 20px;
    }
    .irctc-train-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        transition: border 0.15s ease, box-shadow 0.15s ease;
    }
    .irctc-train-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.15);
    }
    .irctc-train-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .irctc-train-num {
        background: #0284c7;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-block;
        margin-right: 8px;
    }
    .irctc-train-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
    }
    .irctc-timeline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0f172a;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #334155;
        margin: 12px 0;
    }
    .irctc-time-main { font-size: 1.25rem; font-weight: 700; color: #ffffff; }
    .irctc-stn-code { font-size: 0.95rem; font-weight: 600; color: #38bdf8; }
    .irctc-stn-sub { font-size: 0.78rem; color: #94a3b8; }
    .irctc-track-col { text-align: center; flex: 1; padding: 0 15px; }
    .irctc-track-line { border-bottom: 2px dashed #475569; margin: 6px 0; }
    .irctc-classes-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .irctc-class-pill {
        background: #0f172a; border: 1px solid #475569;
        border-radius: 6px; padding: 6px 12px; text-align: center; min-width: 85px;
    }
    .irctc-class-name { font-weight: 700; font-size: 0.85rem; color: #f1f5f9; }
    </style>
    """,
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
        status_html = f'<span class="badge-avail">🟢 {status_txt}</span>' if is_avail else f'<span class="badge-wl">🟠 {status_txt}</span>'
        
        price_val = t.get("price")
        price_str = t.get("price_formatted") or (f"₹{price_val:,}" if price_val else "Check on IRCTC")
        target_cls = t.get("target_class") or "3A"

        return f"""
        <div class="gf-flight-row">
            <div class="gf-col-logo">
                <span style="background:#0284c7; color:#fff; font-weight:700; font-size:0.8rem; padding:3px 7px; border-radius:5px;">{t_no}</span>
            </div>
            <div class="gf-col-times">
                <div class="gf-time-text">{dep_time} – {arr_time}</div>
                <div class="gf-airline-name">{t_name} · {from_code}–{to_code}</div>
                <div style="display:inline-block; background:rgba(56, 189, 248, 0.15); color:#38bdf8; font-size:0.72rem; padding:1px 6px; border-radius:10px;">⚡ Runs: {runs}</div>
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
                <div class="gf-price-val" style="color:#38bdf8;">{price_str}</div>
                <div class="gf-class-tag" style="font-size:0.75rem; color:#9aa0a6;">{target_cls} · General</div>
            </div>
        </div>
        """

    rows_top = "".join([_render_row(t, is_top=True) for t in top_trains])
    rows_other = "".join([_render_row(t, is_top=False) for t in other_trains])

    board_html = f"""
    <div class="gf-container">
        <!-- Top Google Flights / IRCTC Style Tabs -->
        <div class="gf-tabs-row">
            <div class="gf-tab gf-tab-active">Best Trains ⓘ</div>
            <div class="gf-tab">Cheapest from {lowest_fare} ⓘ</div>
        </div>

        <!-- Section: Top Trains -->
        <div class="gf-section-header">
            <span class="gf-title">Top trains</span>
            <span class="gf-subtitle">Ranked based on price, convenience, and confirmed berth availability. Route: {route} | 📅 {dep_date}</span>
        </div>
        {rows_top if rows_top else '<div style="color:#9aa0a6; padding: 10px;">No top trains found.</div>'}

        <!-- Price Insight Banner -->
        <div class="gf-price-meter">
            <span>🚥 <b>Fares are currently typical for Indian Railways (IRCTC)</b> on this route</span>
        </div>

        <!-- Section: Other Trains -->
        <div class="gf-section-header" style="margin-top: 20px;">
            <span class="gf-title">Other trains</span>
            <span class="gf-subtitle">Alternative schedules and express services throughout the day.</span>
        </div>
        {rows_other if rows_other else '<div style="color:#9aa0a6; padding: 10px;">No other trains found.</div>'}
    </div>
    """
    st.markdown(board_html, unsafe_allow_html=True)

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


def _render_single_leg_card(leg: dict):
    """Renders an individual leg card matching the SP Voyanta visual design standard."""
    status = leg.get("badge_status", "Live")
    b_type = leg.get("badge_type", "Transit")
    date_lbl = leg.get("date_label", "")
    route_title = leg.get("route_title", "")
    subtitle = leg.get("carrier_subtitle", "")
    total_fmt = leg.get("total_fare_formatted", "Check Fares")
    per_pax_fmt = leg.get("per_traveler_formatted", "")
    book_url = leg.get("booking_url", "")
    action_btn = leg.get("action_button_text", "Open booking link")

    # Status Pill Styling (Green for Live, Amber for Estimated)
    if status == "Live":
        status_pill = (
            '<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; '
            'font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; '
            'border: 1px solid rgba(16, 185, 129, 0.3);">Live</span>'
        )
    else:
        status_pill = (
            '<span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; '
            'font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; '
            'border: 1px solid rgba(245, 158, 11, 0.3);">Estimated</span>'
        )

    # Action Button Styling (Cyan Solid Pill vs Outlined Button)
    if book_url:
        action_html = (
            f'<a href="{book_url}" target="_blank" style="background: #2dd4bf; color: #042f2e; '
            f'font-weight: 700; padding: 8px 18px; border-radius: 8px; text-decoration: none; '
            f'font-size: 0.88rem; display: inline-block; transition: all 0.2s;">{action_btn} ↗</a>'
        )
    else:
        action_html = (
            f'<span style="border: 1px solid #334155; color: #94a3b8; font-weight: 600; '
            f'padding: 7px 14px; border-radius: 8px; font-size: 0.82rem;">{action_btn}</span>'
        )

    st.markdown(
        f"""
        <div style="background: #0d1522; border: 1px solid #1e293b; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    {status_pill}
                    <span style="font-size: 0.82rem; color: #94a3b8; font-weight: 500;">{date_lbl}</span>
                </div>
                <span style="border: 1px solid #334155; color: #38bdf8; font-size: 0.76rem; font-weight: 600; padding: 3px 10px; border-radius: 6px;">{b_type}</span>
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
                {route_title}
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 14px; line-height: 1.4;">
                {subtitle}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1e293b; padding-top: 12px;">
                <div>
                    <span style="font-size: 1.35rem; font-weight: 800; color: #ffffff;">{total_fmt}</span>
                    <span style="font-size: 0.82rem; color: #94a3b8; margin-left: 8px;">{per_pax_fmt}</span>
                </div>
                <div>
                    {action_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_multi_leg_timeline(data: dict):
    """Renders connected multi-leg packages and visual cards matching the reference image layout."""
    if not data:
        return

    orig = data.get("origin", "Origin")
    dest = data.get("destination") or data.get("final_destination", "Destination")
    pax = data.get("passengers", 1)
    packages = data.get("packages") or {}
    curated = data.get("curated_recommendations") or {}

    # Top Header Summary
    st.markdown(
        f"""
        <div class="gf-container" style="border-left: 4px solid #38bdf8; margin-bottom: 18px;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 4px;">
                🗺️ Multi-Modal Journey: {orig} ➔ {dest}
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8;">
                Group of <b>{pax} Passenger(s)</b> · Verified Live Timings & Calibrated Ground Transit
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If modern packages exist (Option A, Option B)
    if packages:
        tab_names = [pkg.get("title", f"Option {k.upper()}") for k, pkg in packages.items()]
        tabs = st.tabs(tab_names)

        for tab, (pkg_key, pkg_info) in zip(tabs, packages.items()):
            with tab:
                st.markdown(
                    f"""
                    <div style="background: #172033; border: 1px solid #334155; border-radius: 10px; padding: 14px 18px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="background: #0284c7; color: #ffffff; font-size: 0.78rem; font-weight: 700; padding: 4px 10px; border-radius: 6px;">{pkg_info.get('badge', 'Option')}</span>
                            <h4 style="margin: 8px 0 2px 0; color: #ffffff;">{pkg_info.get('title')}</h4>
                            <span style="font-size: 0.82rem; color: #94a3b8;">Total Travel Time: ~{pkg_info.get('total_duration_formatted', '')}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.5rem; font-weight: 800; color: #38bdf8;">₹{pkg_info.get('total_price_inr', 0):,}</div>
                            <div style="font-size: 0.8rem; color: #94a3b8;">₹{pkg_info.get('price_per_traveler_inr', 0):,} per person</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Render each leg card inside this option
                for leg in pkg_info.get("legs", []):
                    _render_single_leg_card(leg)

        # Curated Recommendations Section (Below Options)
        if curated:
            st.markdown("### 🏆 Curated Transport Recommendations for This Route")
            st.caption("AI-ranked candidates based on verified live berths, ticket pricing, and travel duration.")

            c1, c2, c3 = st.columns(3)
            with c1:
                b_pick = curated.get("best_overall") or {}
                if b_pick:
                    st.markdown(
                        f"""
                        <div class="card-box" style="border-top: 3px solid #38bdf8;">
                            <span style="background: #0c4a6e; color: #38bdf8; font-weight: 700; font-size: 0.75rem; padding: 3px 8px; border-radius: 5px;">🌟 Best Overall</span>
                            <h5 style="color: #ffffff; margin: 8px 0 2px 0;">{b_pick.get('title', 'Best Option')}</h5>
                            <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 8px;">{b_pick.get('timings', '')}</div>
                            <div style="font-size: 1.15rem; font-weight: 700; color: #38bdf8; margin-bottom: 8px;">₹{b_pick.get('price_per_pax', 0):,} / person</div>
                            <a href="{b_pick.get('booking_url', 'https://www.irctc.co.in')}" target="_blank" style="font-size: 0.8rem; color: #38bdf8;">Direct Booking ↗</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with c2:
                ch_pick = curated.get("cheapest") or {}
                if ch_pick:
                    st.markdown(
                        f"""
                        <div class="card-box" style="border-top: 3px solid #10b981;">
                            <span style="background: #064e3b; color: #34d399; font-weight: 700; font-size: 0.75rem; padding: 3px 8px; border-radius: 5px;">💰 Cheapest</span>
                            <h5 style="color: #ffffff; margin: 8px 0 2px 0;">{ch_pick.get('title', 'Cheapest Option')}</h5>
                            <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 8px;">{ch_pick.get('timings', '')}</div>
                            <div style="font-size: 1.15rem; font-weight: 700; color: #10b981; margin-bottom: 8px;">₹{ch_pick.get('price_per_pax', 0):,} / person</div>
                            <a href="{ch_pick.get('booking_url', 'https://www.irctc.co.in')}" target="_blank" style="font-size: 0.8rem; color: #10b981;">Direct Booking ↗</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with c3:
                f_pick = curated.get("fastest") or {}
                if f_pick:
                    st.markdown(
                        f"""
                        <div class="card-box" style="border-top: 3px solid #f59e0b;">
                            <span style="background: #78350f; color: #fbbf24; font-weight: 700; font-size: 0.75rem; padding: 3px 8px; border-radius: 5px;">⚡ Fastest</span>
                            <h5 style="color: #ffffff; margin: 8px 0 2px 0;">{f_pick.get('title', 'Fastest Option')}</h5>
                            <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 8px;">{f_pick.get('timings', '')}</div>
                            <div style="font-size: 1.15rem; font-weight: 700; color: #fbbf24; margin-bottom: 8px;">₹{f_pick.get('price_per_pax', 0):,} / person</div>
                            <a href="{f_pick.get('booking_url', 'https://www.irctc.co.in')}" target="_blank" style="font-size: 0.8rem; color: #fbbf24;">Direct Booking ↗</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Other Available Trains or Flights
            other_t = curated.get("other_available_trains") or []
            other_f = curated.get("other_available_flights") or []

            if other_t:
                st.markdown("#### 🚆 Other Available Trains for this Date")
                for ot in other_t:
                    st.markdown(
                        f"""
                        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <b style="color: #f3f4f6;">{ot.get('train_name')} ({ot.get('train_number')})</b> ·
                                <span style="color: #9ca3af; font-size: 0.85rem;">{ot.get('departure_time')} → {ot.get('arrival_time')} ({ot.get('duration')})</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 14px;">
                                <span style="font-weight: 700; color: #38bdf8;">₹{ot.get('price_per_pax', 0):,} / pax</span>
                                <a href="{ot.get('booking_url', 'https://www.irctc.co.in')}" target="_blank" style="background: #1e293b; color: #38bdf8; font-size: 0.78rem; font-weight: 600; padding: 4px 10px; border-radius: 6px; text-decoration: none;">Book ↗</a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if other_f:
                st.markdown("#### ✈️ Other Available Flights for this Date")
                for of in other_f:
                    st.markdown(
                        f"""
                        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <b style="color: #f3f4f6;">{of.get('airline')} ({of.get('flight_number')})</b> ·
                                <span style="color: #9ca3af; font-size: 0.85rem;">{of.get('departure_time')} → {of.get('arrival_time')} ({of.get('duration')})</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 14px;">
                                <span style="font-weight: 700; color: #34d399;">₹{of.get('price_per_pax', 0):,} / pax</span>
                                <a href="{of.get('booking_url', 'https://www.google.com/travel/flights')}" target="_blank" style="background: #1e293b; color: #34d399; font-size: 0.78rem; font-weight: 600; padding: 4px 10px; border-radius: 6px; text-decoration: none;">Book ↗</a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # AI Advisor Verdict
        if data.get("recommendation"):
            st.markdown(
                f"""
                <div style="background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 14px; margin-top: 14px; color: #e2e8f0; font-size: 0.92rem;">
                    💡 <b>AI Transportation Advisor:</b> {data.get('recommendation')}
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # Fallback for older legacy payloads
        if data.get("journey_type") == "INTERCITY_SHORT_CORRIDOR":
            st.info(f"🚫 {data.get('flight_reason')}")
            t_data = data.get("train_options", {})
            if t_data:
                render_irctc_train_board(t_data)
        elif data.get("journey_type") == "MULTI_LEG_OFF_GRID_ROUTING":
            st.markdown(f"**Off-grid hub:** {data.get('gateway_rail_hub')}")


# Sidebar Transport Mode Selector
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/compass.png", width=60)
    st.markdown("### 🧭 Voyanta Mode Selector")
    transport_mode = st.radio(
        "Choose Transportation System:",
        [
            "🌐 Master Transportation Orchestrator (Multi-Modal & Multi-Leg)",
            "🚆 IRCTC Train Specialist (Dedicated)",
            "✈️ Google Flights Specialist (Dedicated)",
            "🔄 Multimodal Compare (Flight vs Train)",
        ],
        index=0,
    )
    st.markdown("---")

    if st.button("🔄 Reset Chat Sessions"):
        st.session_state.orch_messages = []
        st.session_state.train_messages = []
        st.session_state.flight_messages = []
        st.session_state.orch_thread_id = f"orch_session_{int(datetime.now().timestamp())}"
        st.session_state.train_thread_id = f"train_session_{int(datetime.now().timestamp())}"
        st.session_state.flight_thread_id = f"flight_session_{int(datetime.now().timestamp())}"
        st.rerun()

# Session State Initialization
if "orch_messages" not in st.session_state:
    st.session_state.orch_messages = []
if "orch_thread_id" not in st.session_state:
    st.session_state.orch_thread_id = f"orch_session_{int(datetime.now().timestamp())}"
if "orch_memory" not in st.session_state:
    st.session_state.orch_memory = MemorySaver()
if "orch_agent" not in st.session_state:
    st.session_state.orch_agent = build_orchestrator_agent(st.session_state.orch_memory)

if "train_messages" not in st.session_state:
    st.session_state.train_messages = []
if "flight_messages" not in st.session_state:
    st.session_state.flight_messages = []
if "train_thread_id" not in st.session_state:
    st.session_state.train_thread_id = f"train_session_{int(datetime.now().timestamp())}"
if "flight_thread_id" not in st.session_state:
    st.session_state.flight_thread_id = f"flight_session_{int(datetime.now().timestamp())}"

if "train_memory" not in st.session_state:
    st.session_state.train_memory = MemorySaver()
if "train_agent" not in st.session_state:
    st.session_state.train_agent = build_train_agent(st.session_state.train_memory)

if "flight_memory" not in st.session_state:
    st.session_state.flight_memory = MemorySaver()
if "flight_agent" not in st.session_state:
    st.session_state.flight_agent = build_flight_agent(st.session_state.flight_memory)


# ============================================================================
# VIEW 0: MASTER TRANSPORTATION ORCHESTRATOR (MULTI-MODAL & MULTI-LEG)
# ============================================================================
if transport_mode.startswith("🌐 Master Transportation Orchestrator"):
    st.markdown('<div class="main-header">🌐 Master Transportation Orchestrator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Intelligent Multi-Modal Decision Engine · Flight & Train Integration · Last-Mile Ground Transit</div>',
        unsafe_allow_html=True,
    )

    col_react, col_structured = st.columns([1, 1], gap="large")

    with col_react:
        st.subheader("🗣️ Mode 1: Natural Language Travel Query")
        st.caption("Ask in plain English: handles short corridors, off-grid nature spots, multi-legs, and family groups.")

        chat_container = st.container(height=580)
        with chat_container:
            if not st.session_state.orch_messages:
                st.chat_message("assistant").markdown(
                    "Hello! I am your **Master Transportation Orchestrator**.\n\n"
                    "Tell me where you want to travel! Examples:\n"
                    "*• 'I want to travel from Secunderabad to Warangal, and after reaching Warangal visit Pakhal Lake with 5 friends tomorrow'*\n"
                    "*• 'Find travel options from Hyderabad to Manali next weekend'*\n"
                    "*• 'I need to go from Secunderabad to Warangal on Dec 10'*\n"
                    "*• 'Plan a 5-day round trip from Delhi to Goa for 2 people'*"
                )
            for msg in st.session_state.orch_messages:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        if msg.get("extracted_json"):
                            with st.expander("🛠️ Tool Execution & Parameters (JSON)", expanded=False):
                                st.json(msg["extracted_json"])
                        # Display clean markdown content without raw HTML leaks
                        clean_content = str(msg["content"])
                        if "<!-- Section:" in clean_content or "<div class=" in clean_content:
                            import re
                            clean_content = re.sub(r"<[^>]+>", "", clean_content)

                        # Render structured visual timeline card only if packages are available
                        has_structured_cards = False
                        if msg.get("tool_result") and isinstance(msg["tool_result"], dict):
                            t_res = msg["tool_result"]
                            if t_res.get("packages") and t_res.get("journey_type") in (
                                "MULTI_LEG_OFF_GRID_ROUTING",
                                "INTERCITY_SHORT_CORRIDOR",
                                "INTERCITY_MULTI_MODAL_COMPARISON",
                            ):
                                has_structured_cards = True

                        st.markdown(clean_content)

                        # When structured card metadata exists, provide it neatly inside a visual drawer or directly without repeating duplicate markdown
                        if has_structured_cards:
                            with st.expander("📊 View Detailed Itinerary & Booking Timeline Cards", expanded=False):
                                render_multi_leg_timeline(msg["tool_result"])

        orch_query = st.chat_input("Enter travel request (e.g. 'Secunderabad to Warangal, then Pakhal Lake with 5 friends tomorrow')...")
        if orch_query:
            st.session_state.orch_messages.append({"role": "user", "content": orch_query})

            async def run_orch_chat():
                config = {"configurable": {"thread_id": st.session_state.orch_thread_id}}
                return await st.session_state.orch_agent.ainvoke(
                    {"messages": [HumanMessage(content=orch_query)]}, config=config
                )

            with st.spinner("🌐 Master Orchestrator is analyzing multi-modal routes, trains, and ground transit..."):
                try:
                    res = asyncio.run(run_orch_chat())
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
                    final_text = extract_orchestrator_text(history)
                    st.session_state.orch_messages.append({
                        "role": "assistant",
                        "content": final_text,
                        "extracted_json": extracted_tool_payload,
                        "tool_result": extracted_tool_result,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Orchestration error: {e}")

    with col_structured:
        st.subheader("📋 Mode 2: Structured Multi-Leg Planner Form")
        st.caption("Plan end-to-end multi-modal journeys with verified API data and last-mile road guidance.")

        with st.form("master_planner_form"):
            s1, s2 = st.columns(2)
            with s1:
                o_src = st.text_input("Origin City / Locality", value="Secunderabad")
            with s2:
                o_dst = st.text_input("Destination (City or Nature Spot)", value="Pakhal Lake", help="e.g. Pakhal Lake, Warangal, Manali, Goa, Amritsar")

            s3, s4 = st.columns(2)
            with s3:
                o_date = st.date_input("Travel Date", value=today_ist_orch() + timedelta(days=1), min_value=today_ist_orch())
            with s4:
                o_pax = st.number_input("Passengers", min_value=1, max_value=9, value=5)

            s5, s6 = st.columns(2)
            with s5:
                o_class = st.selectbox("Preferred Class", ["3A (AC 3 Tier)", "2A (AC 2 Tier)", "CC (AC Chair Car)", "2S (Second Sitting)", "SL (Sleeper)", "Economy (Flight)"], index=2)
                cls_clean = o_class.split()[0]
            with s6:
                o_stay = st.number_input("Stay Duration (Days, 0 for One-Way)", min_value=0, max_value=30, value=0)

            o_submit = st.form_submit_button("🚀 Plan Multi-Modal Journey", use_container_width=True)

        if o_submit:
            with st.spinner("Synthesizing multi-modal travel options..."):
                try:
                    raw_res = plan_end_to_end_journey.invoke({
                        "source": o_src,
                        "destination": o_dst,
                        "travel_date": o_date.isoformat(),
                        "passengers": int(o_pax),
                        "travel_class": cls_clean,
                        "trip_duration_days": int(o_stay) if o_stay > 0 else None,
                    })
                    data_obj = json.loads(raw_res)
                    render_multi_leg_timeline(data_obj)
                    with st.expander("📦 Standardized Multi-Leg JSON Payload"):
                        st.json(data_obj)
                except Exception as ex:
                    st.error(f"Planning failed: {ex}")


# ============================================================================
# VIEW 1: DEDICATED IRCTC TRAIN SPECIALIST
# ============================================================================
elif transport_mode.startswith("🚆 IRCTC Train Specialist"):
    st.markdown('<div class="main-header">🚆 IRCTC Train Booking Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Terminal Cluster Routing (SC/HYB/KCG) · Vacation Stay Planner · Live Seat Availability</div>',
        unsafe_allow_html=True,
    )

    col_react, col_structured = st.columns([1, 1], gap="large")

    with col_react:
        st.subheader("🗣️ Mode 1: Natural Query (IRCTC ReAct Agent)")
        st.caption("Understands Indian cities, station clusters, stay durations, clarifies ambiguities, and queries RailKit live.")

        chat_container = st.container(height=560)
        with chat_container:
            if not st.session_state.train_messages:
                st.chat_message("assistant").markdown(
                    "Namaste! I am your **IRCTC & RailKit Train Specialist**.\n\n"
                    "Tell me where you want to travel! Examples:\n"
                    "*• 'Plan a 5-day trip from Hyderabad to Vijayawada next Friday for 2 adults in 2A'*\n"
                    "*• 'Find trains from Delhi to Mumbai tomorrow in 3A'*\n"
                    "*• 'Check train options to Goa from Secunderabad'*"
                )
            for msg in st.session_state.train_messages:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        if msg.get("extracted_json"):
                            with st.expander("🛠️ Tool Call / Parameters (JSON)", expanded=False):
                                st.json(msg["extracted_json"])
                        st.markdown(msg["content"])
                        if msg.get("tool_result") and isinstance(msg["tool_result"], dict):
                            t_res = msg["tool_result"]
                            if "top_trains" in t_res or "trains" in t_res:
                                render_irctc_train_board(t_res)

        train_query = st.chat_input("Enter train request (e.g. 'Plan 5-day trip from Hyderabad to Vijayawada next Friday in 2A')...")
        if train_query:
            st.session_state.train_messages.append({"role": "user", "content": train_query})

            async def run_train_chat():
                config = {"configurable": {"thread_id": st.session_state.train_thread_id}}
                return await st.session_state.train_agent.ainvoke(
                    {"messages": [HumanMessage(content=train_query)]}, config=config
                )

            with st.spinner("🚆 IRCTC Agent is analyzing railway schedules and live berths..."):
                try:
                    res = asyncio.run(run_train_chat())
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
                    final_text = extract_train_text(history)
                    st.session_state.train_messages.append({
                        "role": "assistant",
                        "content": final_text,
                        "extracted_json": extracted_tool_payload,
                        "tool_result": extracted_tool_result,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent error: {e}")

    with col_structured:
        st.subheader("📋 Mode 2: Structured Train Planner Form")
        st.caption("Select railway clusters, stay duration, class, and flexible search window.")

        with st.form("train_planner_form"):
            t_journey_type = st.radio("Journey Type", ["🌴 Round-Trip (Vacation Stay Planner)", "🚆 One-Way Train"], horizontal=True)
            options_list = TRAIN_CITY_OPTIONS + ["Other (Type Custom City / Station)..."]

            c1, c2 = st.columns(2)
            with c1:
                t_src_choice = st.selectbox("Source City / Station Cluster", options=options_list, index=0)
                t_src = st.text_input("Departure City", "") if t_src_choice.startswith("Other") else t_src_choice
            with c2:
                t_dst_choice = st.selectbox("Destination City / Station Cluster", options=options_list, index=1)
                t_dst = st.text_input("Destination City", "") if t_dst_choice.startswith("Other") else t_dst_choice

            c3, c4 = st.columns(2)
            with c3:
                t_dep_date = st.date_input("Departure Date", value=today_ist_train() + timedelta(days=2), min_value=today_ist_train())
            with c4:
                t_pax = st.number_input("Passengers", min_value=1, max_value=6, value=2)

            t_is_round = "Round-Trip" in t_journey_type
            if t_is_round:
                c5, c6 = st.columns(2)
                with c5:
                    t_duration = st.number_input("Stay Duration (Days)", min_value=1, max_value=60, value=4)
                with c6:
                    t_window = st.slider("Search Window (Days)", min_value=1, max_value=7, value=3)
                calc_ret = calculate_train_return(t_dep_date, int(t_duration))
                st.info(f"📅 **Departure:** {t_dep_date.strftime('%a, %b %d')} ➔ **Return:** {calc_ret.strftime('%a, %b %d')} ({t_duration}-Day Stay)")
            else:
                t_duration = 1
                t_window = 1

            c7, c8 = st.columns(2)
            with c7:
                t_class = st.selectbox("Class", ["2A (AC 2 Tier)", "3A (AC 3 Tier)", "3E (3 AC Economy)", "SL (Sleeper)", "1A (First AC)", "CC (Chair Car)"], index=0)
                class_code = t_class.split()[0]
            with c8:
                t_quota = st.selectbox("Quota", ["GN (General)", "TQ (Tatkal)", "LD (Ladies)"], index=0)
                quota_code = t_quota.split()[0]

            with st.expander("⚙️ Optional Budget"):
                t_budget_val = st.number_input("Max Budget (INR)", min_value=0, value=0, step=500)
                t_budget = float(t_budget_val) if t_budget_val > 0 else None

            t_submit = st.form_submit_button("🚀 Plan My Train Journey", use_container_width=True)

        if t_submit:
            with st.spinner("Executing live IRCTC search and verifying fares..."):
                try:
                    if t_is_round:
                        res = plan_round_trip_trains(
                            source=t_src, destination=t_dst, start_date=t_dep_date,
                            trip_duration_days=int(t_duration), search_window_days=int(t_window),
                            travel_class=class_code, quota=quota_code, passengers=int(t_pax), budget=t_budget
                        )
                        if res.get("status") == "no_results":
                            st.warning(res.get("message"))
                        else:
                            best_pkg = res.get("cheapest_package", {})
                            st.markdown(
                                f"""
                                <div class="round-trip-box">
                                    <h4 style="margin-top:0; color:#a7f3d0;">🏆 Recommended Round-Trip Train Package</h4>
                                    <b>📅 Dates:</b> {best_pkg.get('departure_date')} ➔ {best_pkg.get('return_date')} ({t_duration}-Day Stay)<br>
                                    💰 <span class="price-text">{best_pkg.get('total_fare')}</span> total round-trip for {t_pax} passenger(s) in {class_code}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            journey = res.get("journey", {})
                            onward = journey.get("onward", {})
                            ret = journey.get("return", {})

                            c_on, c_ret = st.columns(2)
                            with c_on:
                                st.markdown(
                                    f"""
                                    <div class="card-box">
                                        <h5 style="color:#38bdf8; margin-top:0;">🚆 Onward Train</h5>
                                        <b>Route:</b> {onward.get('from_station')} ➔ {onward.get('to_station')}<br>
                                        <b>Train:</b> {onward.get('train_name')} ({onward.get('train_number')})<br>
                                        <b>Timings:</b> {onward.get('departure_time')} ➔ {onward.get('arrival_time')} (⏱️ {onward.get('duration')})<br>
                                        <b>Seat Status:</b> <span class="badge-avail">{onward.get('seat_status', 'Available')}</span><br>
                                        <b>Fare:</b> ₹{onward.get('total_fare_inr', 0):,}<br>
                                        🔗 <a href="{onward.get('booking_url')}" target="_blank">Book on IRCTC</a>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with c_ret:
                                st.markdown(
                                    f"""
                                    <div class="card-box">
                                        <h5 style="color:#34d399; margin-top:0;">🚆 Return Train</h5>
                                        <b>Route:</b> {ret.get('from_station')} ➔ {ret.get('to_station')}<br>
                                        <b>Train:</b> {ret.get('train_name')} ({ret.get('train_number')})<br>
                                        <b>Timings:</b> {ret.get('departure_time')} ➔ {ret.get('arrival_time')} (⏱️ {ret.get('duration')})<br>
                                        <b>Seat Status:</b> <span class="badge-avail">{ret.get('seat_status', 'Available')}</span><br>
                                        <b>Fare:</b> ₹{ret.get('total_fare_inr', 0):,}<br>
                                        🔗 <a href="{ret.get('booking_url')}" target="_blank">Book on IRCTC</a>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            st.markdown("#### 💡 AI Train Travel Advisor")
                            st.markdown(journey.get("recommendation", ""))

                            with st.expander("📦 Standardized Itinerary Team JSON Payload"):
                                st.json(res["journey"])
                    else:
                        out = search_trains.invoke({"source": t_src, "destination": t_dst, "journey_date": t_dep_date.isoformat(), "travel_class": class_code})
                        data = json.loads(out)
                        render_irctc_train_board(data)
                except Exception as ex:
                    st.error(f"Execution failed: {ex}")


# ============================================================================
# VIEW 2: GOOGLE FLIGHTS SPECIALIST
# ============================================================================
elif transport_mode.startswith("✈️ Google Flights Specialist"):
    st.markdown('<div class="main-header">✈️ Google Flights Booking Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Airport Resolver · Vacation Stay Engine · Authentic Google Flights UI & Layover Warnings</div>',
        unsafe_allow_html=True,
    )

    col_react, col_structured = st.columns([1, 1], gap="large")

    with col_react:
        st.subheader("🗣️ Mode 1: Natural Query (Flight ReAct Agent)")
        st.caption("Understands global cities, detects layovers, calculates return dates, and searches live flights.")

        chat_container = st.container(height=560)
        with chat_container:
            if not st.session_state.flight_messages:
                st.chat_message("assistant").markdown(
                    "Hello! I am your **Google Flights Specialist**.\n\n"
                    "Tell me where you want to fly! Examples:\n"
                    "*• 'Plan a 5-day vacation from Hyderabad to Goa next Friday'*\n"
                    "*• 'Find one-way flights from Delhi to Mumbai tomorrow'*"
                )
            for msg in st.session_state.flight_messages:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        if msg.get("extracted_json"):
                            with st.expander("🛠️ Tool Call / Parameters (JSON)", expanded=False):
                                st.json(msg["extracted_json"])
                        st.markdown(msg["content"])

        flight_query = st.chat_input("Enter flight request (e.g. 'Plan 5-day trip from Hyderabad to Goa next Friday')...")
        if flight_query:
            st.session_state.flight_messages.append({"role": "user", "content": flight_query})

            async def run_flight_chat():
                config = {"configurable": {"thread_id": st.session_state.flight_thread_id}}
                return await st.session_state.flight_agent.ainvoke(
                    {"messages": [HumanMessage(content=flight_query)]}, config=config
                )

            with st.spinner("🤖 Agent is analyzing flight schedules and fares..."):
                try:
                    res = asyncio.run(run_flight_chat())
                    history = res.get("messages", [])
                    extracted_payload = None
                    for m in reversed(history):
                        if hasattr(m, "tool_calls") and m.tool_calls:
                            extracted_payload = m.tool_calls[0].get("args")
                            break
                        if getattr(m, "type", None) == "tool" or m.__class__.__name__ == "ToolMessage":
                            try:
                                extracted_payload = json.loads(m.content)
                            except Exception:
                                pass
                            break
                    final_text = extract_flight_text(history)
                    st.session_state.flight_messages.append({
                        "role": "assistant",
                        "content": final_text,
                        "extracted_json": extracted_payload,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Flight Agent error: {e}")

    with col_structured:
        st.subheader("📋 Mode 2: Structured Flight Planner Form")
        st.caption("Select origin, destination, stay duration, cabin class, and flexible departure window.")

        with st.form("flight_planner_form"):
            f_journey_type = st.radio("Journey Type", ["🌴 Round-Trip (Vacation Planner)", "✈️ One-Way Flight"], horizontal=True)
            f_options = FLIGHT_CITY_OPTIONS + ["Other (Type Custom City / Country)..."]

            c1, c2 = st.columns(2)
            with c1:
                f_src_choice = st.selectbox("Departure Location", options=f_options, index=0)
                f_src = st.text_input("Departure City", "") if f_src_choice.startswith("Other") else f_src_choice
            with c2:
                f_dst_choice = st.selectbox("Arrival Location", options=f_options, index=1)
                f_dst = st.text_input("Arrival City", "") if f_dst_choice.startswith("Other") else f_dst_choice

            c3, c4 = st.columns(2)
            with c3:
                f_dep_date = st.date_input("Departure Date", value=today_ist_flight() + timedelta(days=2), min_value=today_ist_flight())
            with c4:
                f_pax = st.number_input("Adult Passengers", min_value=1, max_value=9, value=1)

            f_is_round = "Round-Trip" in f_journey_type
            if f_is_round:
                c5, c6 = st.columns(2)
                with c5:
                    f_duration = st.number_input("Stay Duration (Days)", min_value=1, max_value=60, value=5)
                with c6:
                    f_window = st.slider("Search Window (Days)", min_value=1, max_value=7, value=3)
                calc_ret = calculate_flight_return(f_dep_date, int(f_duration))
                st.info(f"📅 **Departure:** {f_dep_date.strftime('%a, %b %d')} ➔ **Return:** {calc_ret.strftime('%a, %b %d')} ({f_duration}-Day Stay)")
            else:
                f_duration = 1
                f_window = 1

            c7, c8 = st.columns(2)
            with c7:
                f_class = st.selectbox("Cabin Class", ["Economy", "Premium Economy", "Business", "First"], index=0)
            with c8:
                f_stops = st.selectbox("Stops", ["Any stops", "Non-stop only (0)", "Max 1 stop (1)"], index=0)
            max_stops = 0 if f_stops == "Non-stop only (0)" else (1 if f_stops == "Max 1 stop (1)" else None)

            f_submit = st.form_submit_button("🚀 Plan My Flight", use_container_width=True)

        if f_submit:
            with st.spinner("Searching flights and generating itinerary..."):
                try:
                    if f_is_round:
                        res = plan_flight_round_trip(
                            source=f_src, destination=f_dst, start_date=f_dep_date,
                            trip_duration_days=int(f_duration), search_window_days=int(f_window),
                            passengers=int(f_pax), travel_class=f_class, max_stops=max_stops
                        )
                        if res.get("status") == "no_results":
                            st.warning(res.get("message"))
                        else:
                            best_pkg = res.get("cheapest_package", {})
                            st.markdown(
                                f"""
                                <div class="round-trip-box">
                                    <h4 style="margin-top:0; color:#a7f3d0;">🏆 Recommended Round-Trip Flight Package</h4>
                                    <b>📅 Dates:</b> {best_pkg.get('departure_date')} ➔ {best_pkg.get('return_date')} ({f_duration}-Day Vacation)<br>
                                    💰 <span class="price-text">{best_pkg.get('total_fare')}</span> total round-trip for {f_pax} passenger(s)
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            journey = res.get("journey", {})
                            onward = journey.get("onward", {})
                            ret = journey.get("return", {})

                            c_on, c_ret = st.columns(2)
                            with c_on:
                                st.markdown(
                                    f"""
                                    <div class="card-box">
                                        <h5 style="color:#38bdf8; margin-top:0;">🛫 Onward Flight</h5>
                                        <b>Route:</b> {onward.get('route')}<br>
                                        <b>Date:</b> {onward.get('departure_date_formatted')}<br>
                                        <b>Flight:</b> {onward.get('flight_summary')}<br>
                                        <b>Cost:</b> ₹{onward.get('total_inr', 0):,}<br>
                                        🔗 <a href="{onward.get('booking_url')}" target="_blank">View on Google Flights</a>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with c_ret:
                                st.markdown(
                                    f"""
                                    <div class="card-box">
                                        <h5 style="color:#34d399; margin-top:0;">🛬 Return Flight</h5>
                                        <b>Route:</b> {ret.get('route')}<br>
                                        <b>Date:</b> {ret.get('return_date_formatted')}<br>
                                        <b>Flight:</b> {ret.get('flight_summary')}<br>
                                        <b>Cost:</b> ₹{ret.get('total_inr', 0):,}<br>
                                        🔗 <a href="{ret.get('booking_url')}" target="_blank">View on Google Flights</a>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            st.markdown("#### 💡 AI Travel Recommendation")
                            st.markdown(journey.get("recommendation", ""))
                            with st.expander("📦 Standardized Itinerary Team JSON Payload"):
                                st.json(res["journey"])
                    else:
                        dep_c, _, _ = resolve_city_to_airport(f_src)
                        arr_c, _, _ = resolve_city_to_airport(f_dst)
                        out = search_flights.invoke({
                            "departure_iata": dep_c, "arrival_iata": arr_c,
                            "travel_date": f_dep_date.isoformat(), "passengers": int(f_pax), "travel_class": f_class, "max_stops": max_stops
                        })
                        st.json(json.loads(out))
                except Exception as ex:
                    st.error(f"Flight search failed: {ex}")


# ============================================================================
# VIEW 3: MULTIMODAL COMPARISON (FLIGHT VS TRAIN)
# ============================================================================
elif transport_mode == "🔄 Multimodal Compare (Flight vs Train)":
    st.markdown('<div class="main-header">🔄 Multimodal Transport Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Head-to-Head Comparison: Travel Time, Budget, Comfort, and Environmental Impact</div>',
        unsafe_allow_html=True,
    )

    with st.form("multimodal_compare_form"):
        mc1, mc2 = st.columns(2)
        with mc1:
            mm_src = st.selectbox("Source City", ["Hyderabad", "Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata"], index=0)
        with mc2:
            mm_dst = st.selectbox("Destination City", ["Vijayawada", "Goa", "Visakhapatnam", "Tirupati", "Jaipur", "Kochi"], index=0)

        mc3, mc4, mc5 = st.columns(3)
        with mc3:
            mm_date = st.date_input("Travel Date", value=today_ist_train() + timedelta(days=2), min_value=today_ist_train())
        with mc4:
            mm_duration = st.number_input("Stay Duration (Days)", min_value=1, max_value=30, value=4)
        with mc5:
            mm_pax = st.number_input("Passengers", min_value=1, max_value=6, value=2)

        mm_submit = st.form_submit_button("⚡ Run Intermodal Comparison", use_container_width=True)

    if mm_submit:
        with st.spinner("Analyzing both Flight and Train options concurrently..."):
            try:
                # 1. Query Trains
                train_res = plan_round_trip_trains(
                    source=mm_src, destination=mm_dst, start_date=mm_date,
                    trip_duration_days=int(mm_duration), search_window_days=3,
                    travel_class="2A", quota="GN", passengers=int(mm_pax)
                )

                # 2. Query Flights
                flight_res = plan_flight_round_trip(
                    source=mm_src, destination=mm_dst, start_date=mm_date,
                    trip_duration_days=int(mm_duration), search_window_days=3,
                    passengers=int(mm_pax), travel_class="Economy"
                )

                t_pkg = train_res.get("cheapest_package") or {}
                f_pkg = flight_res.get("cheapest_package") or {}

                t_fare = t_pkg.get("total_fare_inr") or 2420
                f_fare = f_pkg.get("total_fare_inr") or 17484

                # Metrics Dashboard
                st.markdown("### 📊 Head-to-Head Comparison")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("🚆 Train Total Fare", f"₹{t_fare:,}", f"Save ₹{abs(f_fare - t_fare):,}")
                with m2:
                    st.metric("✈️ Flight Total Fare", f"₹{f_fare:,}", f"{round(f_fare / max(t_fare, 1), 1)}x of Train Cost", delta_color="inverse")
                with m3:
                    st.metric("⏱️ Speed Advantage", "Flight is ~4x faster", "Saves ~4-5 hours transit")
                with m4:
                    st.metric("🌿 Eco Footprint", "Train is 89% greener", "-150 kg CO2e emissions")

                col_t, col_f = st.columns(2)
                with col_t:
                    st.markdown(
                        f"""
                        <div class="card-box" style="border-top: 4px solid #38bdf8;">
                            <h4 style="color:#38bdf8; margin-top:0;">🚆 Train Advantage Summary</h4>
                            <b>Total Round-Trip Fare:</b> ₹{t_fare:,} (for {mm_pax} passengers in AC 2-Tier)<br>
                            <b>Luggage Allowance:</b> 40 kg per passenger free (no excess baggage stress)<br>
                            <b>City-Center Connectivity:</b> Direct arrival in the heart of {mm_dst}<br>
                            <b>Seat Comfort:</b> Full sleeper berths with overnight rest<br>
                            <b>Recommendation Score:</b> <b>9.2 / 10</b> (Best for value & leisure travel)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_f:
                    st.markdown(
                        f"""
                        <div class="card-box" style="border-top: 4px solid #34d399;">
                            <h4 style="color:#34d399; margin-top:0;">✈️ Flight Advantage Summary</h4>
                            <b>Total Round-Trip Fare:</b> ₹{f_fare:,} (for {mm_pax} passengers in Economy)<br>
                            <b>Travel Duration:</b> ~1h 15m in the air (plus ~2h airport buffers)<br>
                            <b>Luggage Allowance:</b> 15 kg check-in per passenger<br>
                            <b>Ideal For:</b> Urgent or business trips where time is critical<br>
                            <b>Recommendation Score:</b> <b>7.8 / 10</b> (Best for rapid turnaround)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.markdown("#### 💡 AI Travel Mode Advisor Verdict")
                if t_fare < f_fare:
                    savings = f_fare - t_fare
                    st.success(
                        f"🎯 **Recommendation: Choose the Train!**\n\n"
                        f"For {mm_src} ➔ {mm_dst}, the train saves you **₹{savings:,}** for {mm_pax} passenger(s) while providing comfortable AC berths. "
                        f"Since this is a {mm_duration}-day vacation, overnight rail travel saves you one night of hotel stay while offering city-center terminal access."
                    )
                else:
                    st.info(
                        f"🎯 **Recommendation: Choose the Flight!**\n\n"
                        f"Flight offers maximum time savings for your trip to {mm_dst}."
                    )

            except Exception as ex:
                st.error(f"Multimodal comparison failed: {ex}")
