"""Unit tests for SP Voyanta Transport Agent (fast, offline, deterministic)."""
import os
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "flights", ROOT_DIR / "trains"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from city_resolver import resolve_city_to_airport
from ground_estimator import calculate_generic_road_transit
from maps_service import _format_address
from orchestrator_tools import (
    _build_leg_card,
    _format_minutes_to_duration,
    _format_train_seat_status,
    _get_train_fare,
    _parse_duration_minutes,
)
from railkit_client import normalize_date_str
from station_resolver import resolve_city_to_station


class TestMapsAndAddressFormatting(unittest.TestCase):
    def test_format_address_preserves_international(self):
        self.assertEqual(_format_address("Zurich"), "Zurich")
        self.assertEqual(_format_address("  Dubai, UAE  "), "Dubai, UAE")

    def test_format_address_preserves_indian_places(self):
        self.assertEqual(_format_address("Gachibowli, Hyderabad"), "Gachibowli, Hyderabad")


class TestOrchestratorHelpers(unittest.TestCase):
    def test_parse_duration_minutes_formats(self):
        self.assertEqual(_parse_duration_minutes("1h 39m"), 99)
        self.assertEqual(_parse_duration_minutes("02:44 hrs"), 164)
        self.assertEqual(_parse_duration_minutes("45 mins"), 45)
        self.assertEqual(_parse_duration_minutes({"duration_minutes": 120}), 120)
        self.assertEqual(_parse_duration_minutes(""), 9999)

    def test_format_minutes_to_duration(self):
        self.assertEqual(_format_minutes_to_duration(99), "1h 39m")
        self.assertEqual(_format_minutes_to_duration(120), "2h")
        self.assertEqual(_format_minutes_to_duration(45), "45 mins")
        self.assertEqual(_format_minutes_to_duration(0), "Direct Transit")

    def test_get_train_fare(self):
        self.assertEqual(_get_train_fare({"price": 525}), 525)
        self.assertEqual(_get_train_fare({"totalFare": "1650.0"}), 1650)
        self.assertEqual(_get_train_fare({}), 0)
        self.assertEqual(_get_train_fare(None), 0)

    def test_format_train_seat_status(self):
        self.assertIn("Available", _format_train_seat_status({"seat_status": "AVL 171"}, "CC"))
        self.assertIn("Waiting List", _format_train_seat_status({"seat_status": "WL 12"}, "3A"))
        self.assertIn("RAC", _format_train_seat_status({"seat_status": "RAC 4"}, "3A"))

    def test_build_leg_card_per_pax_math(self):
        card = _build_leg_card(
            badge_status="Live",
            badge_type="Train",
            date_label="Leg 1",
            route_title="SC ➔ WL",
            carrier_subtitle="Vande Bharat",
            total_fare_inr=1050,
            passengers=2,
        )
        self.assertEqual(card["total_fare_inr"], 1050)
        self.assertEqual(card["per_traveler_fare_inr"], 525)
        self.assertEqual(card["total_fare_formatted"], "₹1,050")


class TestGroundEstimator(unittest.TestCase):
    @patch("ground_estimator.get_exact_route")
    def test_sedan_vs_suv_pricing(self, mock_route):
        mock_route.return_value = {
            "distance_km": 100.0,
            "duration_minutes": 120,
            "duration_formatted": "2h 0m",
        }
        sedan_res = calculate_generic_road_transit("A", "B", passengers=2)
        suv_res = calculate_generic_road_transit("A", "B", passengers=4)

        self.assertTrue(sedan_res["is_live_api"])
        self.assertEqual(sedan_res["options"]["private_cab"]["total_fare_inr"], int(100.0 * 18.0 + 150))
        self.assertEqual(suv_res["options"]["private_cab"]["total_fare_inr"], int(100.0 * 24.0 + 250))
        self.assertEqual(sedan_res["options"]["bus"]["total_fare_inr"], int(100.0 * 1.60) * 2)


class TestDateAndEntityResolvers(unittest.TestCase):
    def test_normalize_date_str(self):
        self.assertEqual(normalize_date_str("2026-12-04"), "04-12-2026")
        self.assertEqual(normalize_date_str("04/12/2026"), "04-12-2026")
        self.assertEqual(normalize_date_str(date(2026, 11, 27)), "27-11-2026")

    def test_city_and_station_resolvers_offline(self):
        iata, city, _ = resolve_city_to_airport("Hyderabad", allow_llm=False)
        self.assertEqual(iata, "HYD")
        self.assertEqual(city, "Hyderabad")

        stn, clusters, _, _ = resolve_city_to_station("Hyderabad", allow_llm=False)
        self.assertEqual(stn, "SC")
        self.assertIn("SC", clusters)

    def test_broad_state_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_city_to_airport("Andhra Pradesh", allow_llm=False)
        with self.assertRaises(ValueError):
            resolve_city_to_station("Andhra Pradesh", allow_llm=False)


if __name__ == "__main__":
    unittest.main()
